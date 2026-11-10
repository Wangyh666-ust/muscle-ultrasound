"""Attach visible curves to frozen measurement candidates and audit local PA support."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import time

from scripts.round48_length_replay import (
    ROOT, R46, R41, CFG, KINDS, SEEDS, SPLIT, SCOPES, key, rows, j, sha, write, gzread,
    gzwrite, Path, np, source_path, read_mask, extract, initialize, check_equal,
)
from scripts.round48_evaluate import status, paired_ci, error, summary
from scripts.round50_curved_tracks import matching, REF_WIDTH, axis_delta, decode_shapes, component_pixels
from scripts.round29_core import line_at, value_at

DEST=ROOT/'runs/round51_dual_fascicle_v1'
R49=ROOT/'runs/round49_short_fascicle_v1'
R50=ROOT/'runs/round50_curved_tracks_v1'
PLAN=ROOT/'thinking/ROUND51_DUAL_FASCICLE_PA_FL_ATTRIBUTION_PLAN_20261105.txt'
NATURAL=ROOT/'runs/round47_boundary_selection_v1/evaluation/natural_pairs.json.gz'
PARAMETERS=((4,20),(4,30),(8,20),(8,30))
DEV_MIN=15
HIST_MIN=20
PA_SLOPE_TOL=3.


def reg():
    f=DEST/'protocol.json'
    paths=[PLAN,Path(__file__),CFG,R49/'success.json',R50/'success.json',R50/'lock.json',
           R41/'partitions.json',NATURAL]
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    protocol=dict(sources=sources, pa_candidates=PARAMETERS, min_development_images_each_c1=DEV_MIN,
       min_confirmation_images_each_c1=HIST_MIN, max_terminal_axis_spread_deg=PA_SLOPE_TOL,
       pa_lock_rule='Highest minimum eligible independent development image count over C1 seeds. '
                    'Tie uses smaller gap, then longer terminal evidence. '
                    'Full PA replay requires at least 15 development and 20 historical original images per C1, '
                    'with a nonzero eligible count under each boundary condition.',
       bounds='8 reference pixels maximum candidate PA approach; no P contribution without existing PA weight. '
              'V is exactly read-only for PA, FL, MT, and weights.',
       training=False,default_changed=False,subagents=False)
    if f.exists():
        assert j(f)==j_normalized(protocol)
    else:
        write(f,protocol)
    return protocol


def j_normalized(v):
    import json
    return json.loads(json.dumps(v))


def path50(r,kind,tr):
    stage='original' if tr=='identity' else 'perturbation'
    return R50/stage/kind/r['dataset']/(Path(r['image_id']).stem+'.json.gz')


def paths_from_cache(r,kind):
    old=gzread(path50(r,kind,'identity'))
    other=path50(r,kind,'panel')
    more=gzread(other)['variants'] if other.exists() else {}
    threshold=str(j(R50/'lock.json')['threshold'])
    return old['human'],{tr:v['thresholds'][threshold] for tr,v in {**old['variants'],**more}.items()}


def at_arc(path,s,where):
    return np.column_stack([np.interp(where,s,path[:,i]) for i in range(2)])


def actual_crossings(points, curve, side):
    fit=curve[side]
    knots=np.asarray(fit['knots_x'])
    values=np.asarray(fit['knots_y'])
    q=[]
    elapsed=0.
    for a,b in zip(points[:-1],points[1:]):
        dx=b[0]-a[0]
        fractions=[0.,1.]
        if abs(dx)>1e-12:
            fr=(knots-a[0])/dx
            fractions.extend(fr[(fr>0)&(fr<1)].tolist())
        fractions=np.array(sorted(fractions))
        p=a+fractions[:,None]*(b-a)
        signed=p[:,1]-np.interp(p[:,0],knots,values)
        length=float(np.linalg.norm(b-a))
        for i in range(len(signed)-1):
            if abs(signed[i])<1e-10 and abs(signed[i+1])<1e-10:
                return []
            if (signed[i]<0)==(signed[i+1]<0):
                continue
            weight=signed[i]/(signed[i]-signed[i+1])
            hit=p[i]+weight*(p[i+1]-p[i])
            dist=elapsed+(fractions[i]+weight*(fractions[i+1]-fractions[i]))*length
            if not q or abs(dist-q[-1][0])>1e-7:
                q.append((dist,hit))
        elapsed+=length
    return q


def in_observed_band(p,cv):
    x,y=map(float,p)
    if not cv['support'][0]<=x<=cv['support'][1]:
        return False
    return value_at(cv['sup'],x)-1e-6<=y<=value_at(cv['deep'],x)+1e-6


def measured_x(x,cv):
    return cv['support'][0]<=x<=cv['support'][1] and (
        min(abs(np.asarray(cv['paired_x'])-x))<=cv['max_sample_gap'])


def end_tangent(path,s,at,orientation,win,ref):
    directions=[]
    vector=None
    for width,offset in ((win,0.),(.8*win,0.),(win,2*ref)):
        ds=at-orientation*(offset+np.linspace(0,width,13))
        if ds.min()<-1e-8 or ds.max()>s[-1]+1e-8:
            return None
        points=at_arc(path,s,ds)
        diff=points-points.mean(axis=0)
        _,_,axes=np.linalg.svd(diff,full_matrices=False)
        u=axes[0]
        if u@(points[0]-points[-1])<0:
            u=-u
        if vector is None:
            vector=u
        directions.append(math.degrees(math.atan2(u[1],u[0]))%180)
    spread=max(axis_delta(a,b) for a in directions for b in directions)
    if spread>PA_SLOPE_TOL:
        return None
    return vector,directions[0],spread


def endpoint(track,cv,span,max_gap,window):
    if cv is None:
        return None
    path=np.asarray(track['path'],float)
    s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    ref=span/REF_WIDTH
    win=window*ref
    full=actual_crossings(path,cv,'deep')
    if len(full)>1:
        return None
    origins=[]
    if len(full)==1:
        at,p=full[0]
        if min(at,s[-1]-at)>max_gap*ref:
            return None
        origins=[(at,p,1,'direct'),(at,p,-1,'direct')]
    else:
        origins=[(0.,path[0],-1,'near'),(s[-1],path[-1],1,'near')]
    good=[]
    for at,p,ori,mode in origins:
        ds=at-ori*np.linspace(.05*ref,win,11)
        if ds.min()<0 or ds.max()>s[-1]:
            continue
        if not all(in_observed_band(x,cv) for x in at_arc(path,s,ds)):
            continue
        fit=end_tangent(path,s,at,ori,win,ref)
        if fit is None:
            continue
        vector,axis,spread=fit
        distance=0.
        end=p
        if mode=='near':
            if not in_observed_band(p,cv):
                continue
            continuation=p+vector*max_gap*ref
            values=[v[1]-value_at(cv['deep'],float(v[0])) for v in (p,continuation)]
            if values[0]>1e-8 or values[1]<=0.:
                continue
            roots=actual_crossings(np.stack((p,continuation)),cv,'deep')
            if len(roots)!=1:
                continue
            distance,end=roots[0]
        if not measured_x(float(end[0]),cv):
            continue
        angle=axis_delta(axis,math.degrees(math.atan(line_at(cv['deep'],float(end[0]))[0])))
        if angle<3.:
            continue
        good.append(dict(mode=mode,pa_deg=angle,tangent_deg=axis,spread_deg=spread,
                         extension_ref=distance/ref,location=end.tolist(),observed_end=p.tolist()))
    return good[0] if len(good)==1 else None


def control_tests():
    def cv(deep):
        return dict(support=[0.,160.],paired_x=list(range(161)),max_sample_gap=1.,
                    sup=dict(knots_x=[0.,160.],knots_y=[30.,30.]),
                    deep=dict(knots_x=[0.,160.],knots_y=deep))
    def tr(x,y):
        return dict(path=np.column_stack((x,y)).tolist())
    x=np.linspace(20.,104.,45)
    straight=tr(x,42.+.6*(x-20.))
    near=endpoint(straight,cv([95.,95.]),1046,8,20)
    assert near and near['mode']=='near' and abs(near['pa_deg']-math.degrees(math.atan(.6)))<1e-8
    assert endpoint(straight,cv([100.,100.]),1046,8,20) is None
    assert endpoint(straight,cv([90.,90.]),1046,8,20)['mode']=='direct'
    flat=tr(x,np.full(len(x),94.))
    assert endpoint(flat,cv([95.,95.]),1046,8,20) is None
    x2=np.linspace(48,109,51)
    bent=tr(x2,55.+.58*(x2-48.)+.0011*(x2-48.)**2)
    e=endpoint(bent,cv([96.,96.]),1046,8,20)
    assert e and e['mode']=='near'
    short=tr(x[-5:],42.+.6*(x[-5:]-20.))
    assert endpoint(short,cv([95.,95.]),1046,8,20) is None
    return dict(near_line=near,direct=True,too_far=True,parallel=True,curve=e,short_window=True)


def pa_value(source, qualified):
    value=source['values']['PA']
    if value is None:
        assert not qualified
        return None
    if not qualified:
        return value
    weights=source['PA_weights']
    # Baseline directions and centroid evaluation are exactly the frozen PA definition.
    cv=source['curve']
    segments={str(s['label_id']):s for s in source['display']['segments']}
    delta=0.
    for track,e in qualified.items():
        s=segments[track]
        m=line_at(cv['deep'],float(s['centroid'][0]))[0]
        old=axis_delta(s['direction_deg'],math.degrees(math.atan(m)))
        delta+=weights[track]*(e['pa_deg']-old)
    return float(value+delta)


def result_path(r,phase):
    return DEST/phase/r['dataset']/(Path(r['image_id']).stem+'.json.gz')


def lock_settings():
    p=DEST/'pa_lock.json'
    dev=[gzread(result_path(r,'development')) for r in rows('development')]
    audit=eligibility_counts(dev)
    scores={}
    for gap,window in PARAMETERS:
        t=f'{gap}|{window}'
        scores[t]=min(audit.get(t,{}).get(k,{}).get('identity_images',0) for k in KINDS)
    best=sorted(PARAMETERS,key=lambda q:(-scores[f'{q[0]}|{q[1]}'],q[0],-q[1]))[0]
    write(p,dict(parameters=best,scores=scores,development_gate=scores[f'{best[0]}|{best[1]}']>=DEV_MIN,
                 development_counts=audit,controls=control_tests(),
                 protocol_sha256=sha(DEST/'protocol.json')))
    return j(p)


def review_row(r, phase):
    dest=result_path(r,phase)
    if dest.exists():
        return
    old={s:gzread(source_path(r,s)) for s in SEEDS}
    historic=gzread(R49/'replay'/r['dataset']/(Path(r['image_id']).stem+'.json.gz'))
    settings=PARAMETERS if phase=='development' else [tuple(j(DEST/'pa_lock.json')['parameters'])]
    cfg=j(CFG)
    result={}
    shape_cache={}
    for kind in KINDS:
        labels,paths=paths_from_cache(r,kind)
        for tr,record in paths.items():
            a={p['id']:p for p in record['paths']['A']}
            c={p['id']:p for p in record['paths']['C'] if p['mode']=='curve'}
            assert set(a)<=set(p['id'] for p in record['paths']['C'])
            before=old[SEEDS[0]]['variants'][tr]
            t=before['transform']
            width=before['measurement_roi']['box'][2]-before['measurement_roi']['box'][0]
            span=width/t['sx']
            ids_only=set(c)-set(a)
            audits={}
            if ids_only:
                raw={p['id']:p['gate']['flags'] for p in record['shape']['base']}
                if not ids_only<=raw.keys():
                    mask,_=read_mask(r,tr,kind,before)
                    _,_,audit=decode_shapes([(sid,p) for sid,p in component_pixels(mask) if sid in ids_only],
                                           before['measurement_roi'],cfg,True)
                    raw={p['id']:p['gate']['flags'] for p in audit}
                assert ids_only<=raw.keys()
                audits={str(i):raw[i] for i in ids_only}
            dual={str(i):dict(straight=a.get(i),curve=c.get(i),base_geometry=bool(i in a),
                              reject_reasons=audits.get(str(i),[])) for i in sorted(set(a)|set(c))}
            assert len(dual)==len(a)+len(ids_only)
            by_seed={}
            for seed,row in old.items():
                q=row['variants'][tr]['arms'][kind+'|ID']
                base=historic['variants'][tr][f'{seed}|{kind}']['arms']['A']
                assert q['values']==base['values']
                assert set(a)==set(s['label_id'] for s in q['display']['segments'])
                segment_index={s['label_id']:s for s in q['display']['segments']}
                for i,p in a.items():
                    check_equal(p['path'],[segment_index[i]['p0'],segment_index[i]['p1']],'paired original path')
                values=q['values']
                reference=dict(values=values,PA_weights=q['PA_weights'],FL_weights=q['FL_weights'])
                candidates={str(i):c[i] for i in c.keys()&a.keys() if str(i) in q['PA_weights']}
                estimates={}
                for gap,window in settings:
                    usable={i:e for i,p in candidates.items()
                            if (e:=endpoint(p,q['curve'],span,gap,window)) is not None}
                    for i,e in usable.items():
                        if labels:
                            rec=matching([c[int(i)]],labels,span)['references'][0]
                            if rec['matched']:
                                ref=next(p for p in labels if p['id']==rec['candidate'])
                                pp=np.asarray(ref['path'])
                                location=np.array(e['observed_end'])
                                idx=int(np.argmin(np.linalg.norm(pp-location,axis=1)))
                                lo,hi=max(0,idx-2),min(len(pp)-1,idx+2)
                                v=pp[hi]-pp[lo]
                                if np.linalg.norm(pp[idx]-location)<=3*span/REF_WIDTH and np.linalg.norm(v)>=6*span/REF_WIDTH:
                                    e['reference_tangent_error_deg']=axis_delta(
                                        e['tangent_deg'],math.degrees(math.atan2(v[1],v[0])))
                    estimates[f'{gap}|{window}']=dict(PA=pa_value(q,usable),eligible=usable)
                by_seed[str(seed)]=dict(R=reference,V=reference.copy(),P=estimates,
                                        curve_contributors=len(candidates),observed_boundary=q['curve'] is not None)
            result.setdefault(tr,{})[kind]=dict(dual=dual,measurement=by_seed,
                    reference=record['reference'],span=span)
    gzwrite(dest,dict(key=key(r),partition=SPLIT['assignment'][key(r)],component=SPLIT['component_ids'][key(r)],
                      variants=result,legacy_source_sha256={str(s):sha(source_path(r,s)) for s in SEEDS}))


def eligibility_counts(all_rows):
    counts=defaultdict(Counter)
    seen=defaultdict(set)
    for r in all_rows:
        for tr,view in r['variants'].items():
            for kind,v in view.items():
                for b,m in v['measurement'].items():
                    for param,p in m['P'].items():
                        unit=counts[param,kind]
                        unit['conditions']+=1
                        unit['curve_pa_candidates']+=m['curve_contributors']
                        if not p['eligible']:
                            continue
                        unit['eligible_conditions']+=1
                        unit['tracks']+=len(p['eligible'])
                        unit['direct']+=sum(q['mode']=='direct' for q in p['eligible'].values())
                        unit['near']+=sum(q['mode']=='near' for q in p['eligible'].values())
                        unit['reference_comparisons']+=sum('reference_tangent_error_deg' in q for q in p['eligible'].values())
                        seen[param,kind,'all_images'].add(r['key'])
                        if tr=='identity':
                            seen[param,kind,'identity_images'].add(r['key'])
                            seen[param,kind,'boundary_'+b].add(r['key'])
    for (param,kind,typ),ids in seen.items():
        counts[param,kind][typ]=len(ids)
    out=defaultdict(dict)
    for (p,k),v in counts.items():
        out[p][k]=dict(v)
    return out


def run_rows(phase,workers):
    selected=rows('development') if phase=='development' else rows()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        tasks=[pool.submit(review_row,r,phase) for r in selected]
        for i,f in enumerate(as_completed(tasks),1):
            f.result()
            if i%50==0 or i==len(tasks):
                print('Round51',phase,i,'/',len(tasks),flush=True)


def summarize():
    data=[gzread(result_path(r,'panel')) for r in rows()]
    lock=j(DEST/'pa_lock.json')
    param='|'.join(map(str,lock['parameters']))
    hist=[r for r in data if r['partition']=='confirmation']
    historical=eligibility_counts(hist)
    tests=historical[param]
    full=lock['development_gate'] and all(
         tests[k].get('identity_images',0)>=HIST_MIN and all(tests[k].get('boundary_'+str(s),0)>0 for s in SEEDS)
         for k in KINDS)
    counts=Counter()
    for r in data:
        for tr,v in r['variants'].items():
            for kind,q in v.items():
                scope='original' if tr=='identity' else 'perturbation'
                counts[scope,kind,'views']+=1
                counts[scope,kind,'same_id_dual']+=sum(p['curve'] is not None and p['straight'] is not None
                                                       for p in q['dual'].values())
                counts[scope,kind,'curve_only']+=sum(p['curve'] is not None and p['straight'] is None
                                                     for p in q['dual'].values())
                for m in q['measurement'].values():
                    assert m['R']==m['V']
                    counts[scope,kind,'exact_measurement_parity']+=1
    result=dict(eligibility_all=eligibility_counts(data),eligibility_historical=historical,
                locked_pa=lock['parameters'],measurement_gate=full,
                groups=[dict(scope=s,kind=k,metric=f,n=n) for (s,k,f),n in sorted(counts.items())])
    write(DEST/'dual_pa_summary.json',result)
    print('Round51 PA full measurement gate',full,tests,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('development','panel','summary'))
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    reg()
    if a.stage=='development':
        control_tests()
        run_rows('development',a.workers)
        print('Round51 PA locked',lock_settings(),flush=True)
    elif a.stage=='panel':
        assert j(DEST/'pa_lock.json')['protocol_sha256']==sha(DEST/'protocol.json')
        run_rows('panel',a.workers)
        summarize()
    else:
        summarize()
