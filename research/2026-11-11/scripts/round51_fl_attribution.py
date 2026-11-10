"""Paired FL replay with frozen boundaries, frozen C1 and joint negative control."""
import argparse
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

from scipy.optimize import linear_sum_assignment
from scripts.round48_length_replay import (
    ROOT, R46, R40, CFG, KINDS, SEEDS, SPLIT, SCOPES, key, rows, j, sha, write,
    gzread, gzwrite, Path, np, source_path, read_mask, extract, initialize, check_equal,
    measure, reduced, map_curve, map_segment,
)
from scripts.round38_evaluate import relative
from scripts.round29_core import value_at
from scripts.round50_curved_tracks import REF_WIDTH, axis_delta, vectors
from scripts.round51_support import DEST, R49, PLAN

ARMS=('N','FB','FM','BOTH')


@lru_cache(maxsize=35)
def saved(image_key):
    ds,name=image_key.split('|')
    return gzread(R49/'replay'/ds/(Path(name).stem+'.json.gz'))


def difference(a,b):
    return None if a is None or b is None else relative(a,b)


def event_manifest():
    file=DEST/'fl_event_manifest.json'
    if file.exists():
        return j(file)
    q=gzread(R49/'worst_by_image.json.gz')
    hist=[r for r in q if r['partition']=='confirmation' and r['scope']=='all26']
    severe=[r for r in hist if r['arms']['A']['FL']['severe']]
    assert len(severe)==38
    severe_images={r['key'] for r in severe}
    severe_groups={r['component'] for r in severe}
    extreme=[]
    controls=[]
    used=set()
    for r in severe:
        k=r['key']
        cond=r['condition']
        src=saved(k)['variants']
        x=src['identity'][cond]['arms']['A']['values']['FL']
        for tr in SCOPES['all26']:
            y=src[tr][cond]['arms']['A']['values']['FL']
            if (dr:=difference(x,y)) is not None and dr>.2:
                extreme.append(dict(key=k,condition=cond,transform=tr,original=x,factual=y,
                                    original_difference=dr,component=r['component']))
        transform=r['arms']['A']['FL']['worst_at']
        data=r['key'].split('|')[0]
        candidates=[c for c in hist if c['condition']==cond and c['key'] not in severe_images
                     and c['component'] not in severe_groups and c['key'] not in used
                     and (v:=c['arms']['A']['FL']['worst']) is not None and 0<v<=.2]
        candidates.sort(key=lambda c:(c['key'].split('|')[0]!=data,
                                        c['arms']['A']['FL']['worst_at']!=transform,
                                        abs(c['arms']['A']['FL']['worst']-.2),c['key']))
        chosen=[]
        for c in candidates[:120]:
            vr=saved(c['key'])['variants']
            x=vr['identity'][cond]['arms']['A']['values']['FL']
            y=vr[transform][cond]['arms']['A']['values']['FL']
            dr=difference(x,y)
            if dr is None or dr<=0:
                continue
            chosen.append((c['key'].split('|')[0]!=data,abs(dr-.2),c,x,y,dr))
            if dr>=.14 and c['key'].split('|')[0]==data:
                break
        assert chosen,(r,transform)
        _,_,c,x,y,dr=min(chosen,key=lambda z:z[:2])
        used.add(c['key'])
        controls.append(dict(key=c['key'],condition=cond,transform=transform,original=x,factual=y,
                             original_difference=dr,component=c['component'],control_for=k))
    assert len(extreme)==106 and len(controls)==38
    p=dict(severe=extreme,controls=controls,condition_count=len(severe),unique_images=len(severe_images),
           rule='All >20% perturbations of the 38 severe historical conditions. One control per condition '
                'matched to the worst perturbation, same boundary/C1 seeds, from a disjoint image and component. '
                'Prefer same dataset and closest nonsevere change within the prespecified search.',
           bindings={p.relative_to(ROOT).as_posix():sha(p) for p in
                     [PLAN,Path(__file__),R49/'worst_by_image.json.gz',R49/'success.json',
                      R46/'lock.json',CFG,R40/'calibration.json']})
    write(file,p)
    return p


def compact(output):
    q=reduced(output,detailed=True)
    risks=output.get('boundary_sensitivity',{}).get('tracks',{})
    q['risks']={i:{f:d[f] for f in ('u','p','reliable','root_switches')} for i,d in risks.items()}
    return q


def compare_boundary(a,b):
    if a is None or b is None:
        return dict(common=False)
    lo=max(a['support'][0],b['support'][0])
    hi=min(a['support'][1],b['support'][1])
    width=max(a['support'][1],b['support'][1])-min(a['support'][0],b['support'][0])
    if hi<=lo:
        return dict(common=False,support_overlap=0.)
    xs=np.linspace(lo,hi,41)
    changes={}
    thickness=np.median([value_at(a['deep'],x)-value_at(a['sup'],x) for x in xs])
    for side in ('sup','deep'):
        delta=np.array([value_at(b[side],x)-value_at(a[side],x) for x in xs])
        offset=float(np.median(delta))
        f,g=a[side],b[side]
        rays={ray:axis_delta(math.degrees(math.atan(f[ray][0])),math.degrees(math.atan(g[ray][0])))
              for ray in ('left_ray','right_ray')}
        changes[side]=dict(median_shift_px=offset,max_shift_px=float(np.max(abs(delta))),
           max_ray_angle_difference_deg=max(rays.values()),ray_angle_difference_deg=rays,
           large_consistent_shift=abs(offset)>.25*thickness and np.mean(np.sign(delta)==np.sign(offset))>=.8)
    return dict(common=True,support_overlap=(hi-lo)/max(width,1.),original_thickness_px=float(thickness),
                side=changes)


def fixed_weight(a,n,span,scale):
    a_weights=a['weights']['FL']
    b_weights=n['weights']['FL']
    aa=[s for s in a['segments'] if str(s['label_id']) in a_weights]
    bb=[s for s in n['segments'] if str(s['label_id']) in b_weights]
    cost=np.full((len(aa),len(bb)),1000.)
    ref=span/REF_WIDTH
    for i,s in enumerate(aa):
        u,v=vectors(s['direction_deg'])
        ax=sorted(np.asarray([s['p0'],s['p1']])@u)
        for k,t in enumerate(bb):
            angle=axis_delta(s['direction_deg'],t['direction_deg'])
            cross=abs(np.subtract(s['centroid'],t['centroid'])@v)
            bx=sorted(np.asarray([t['p0'],t['p1']])@u)
            overlap=max(0.,min(ax[1],bx[1])-max(ax[0],bx[0]))/max(1.,min(ax[1]-ax[0],bx[1]-bx[0]))
            if angle<=6 and cross<=3*ref and overlap>=.3:
                cost[i,k]=angle/6+cross/max(1e-9,3*ref)+(1-overlap)
    pairs=[(aa[i],bb[k]) for i,k in zip(*linear_sum_assignment(cost)) if cost[i,k]<1000.]
    a_mass=sum(a_weights[str(s['label_id'])] for s,t in pairs)
    b_mass=sum(b_weights[str(t['label_id'])] for s,t in pairs)
    result=dict(original_mass=a_mass,perturbation_mass=b_mass,pairs=len(pairs))
    if a_mass<.8 or b_mass<.8:
        return result
    orig=sum(a_weights[str(s['label_id'])]*s['FL_px'] for s,t in pairs)/a_mass/scale
    now=sum(b_weights[str(t['label_id'])]*t['FL_px'] for s,t in pairs)/b_mass/scale
    fixed=sum(a_weights[str(s['label_id'])]*t['FL_px'] for s,t in pairs)/a_mass/scale
    return dict(**result,matched_original=orig,matched_factual=now,matched_fixed_weights=fixed,
                factual_change=difference(orig,now),fixed_weight_change=difference(orig,fixed),
                angles=[axis_delta(s['direction_deg'],t['direction_deg']) for s,t in pairs])


def diagnose(x,arms):
    delta={a:difference(x,arms[a]['values']['FL']) for a in ARMS}
    if delta['FB'] is None or delta['FM'] is None:
        cat='single_swap_missing'
    elif delta['FB']<=.2 and delta['FM']>.2:
        cat='boundary_freeze_only'
    elif delta['FM']<=.2 and delta['FB']>.2:
        cat='fascicle_freeze_only'
    elif delta['FB']<=.2 and delta['FM']<=.2:
        cat='either_freeze'
    elif delta['BOTH'] is not None and delta['BOTH']<=.2:
        cat='joint_freeze_only'
    else:
        cat='unresolved'
    values={a:arms[a]['values']['FL'] for a in ARMS}
    parts=None
    if all(v is not None for v in values.values()):
        n,fb,fm,both=(values[a] for a in ARMS)
        boundary=((fm-both)+(n-fb))/2
        fascicle=((fb-both)+(n-fm))/2
        residual=both-x
        assert math.isclose(boundary+fascicle+residual,n-x,abs_tol=1e-8)
        parts=dict(boundary=boundary,fascicle=fascicle,remaining_frame=residual,
                   interaction=n-fb-fm+both)
    return delta,cat,parts


def run_image(r, events):
    cfg,cal=j(CFG),j(R40/'calibration.json')
    read={s:gzread(source_path(r,s)) for s in SEEDS}
    row=read[SEEDS[0]]
    masks={}
    for e in events:
        tr=e['transform']
        kind=e['condition'].split('|')[1]
        for name in ('identity',tr):
            if (name,kind) in masks:
                continue
            view=row['variants'][name]
            mask,meta=read_mask(r,name,kind,view)
            assert meta['sha256']==row['sources'][name][kind+'_I']
            raw,kept,_,_,_=extract(mask,view['measurement_roi'],cfg)
            masks[name,kind]=(len(raw),kept)
    records=[]
    for e in events:
        seed,kind=e['condition'].split('|')
        seed=int(seed)
        name=e['transform']
        src=read[seed]['variants']
        original=src['identity']
        moved=src[name]
        a=original['arms'][kind+'|ID']
        b=moved['arms'][kind+'|ID']
        assert a['values']['FL']==e['original'] and b['values']['FL']==e['factual']
        assert a['curve'] is not None and b['curve'] is not None
        t,roi=moved['transform'],moved['measurement_roi']
        raw0,s0=masks['identity',kind]
        raw1,s1=masks[name,kind]
        assert len(s0)==a['filter_count'] and len(s1)==b['filter_count']
        assert raw0==a['raw_components'] and raw1==b['raw_components']
        old=measure(a['curve'],a['selected_pair_id'],s0,raw0,original['measurement_roi'],
                    original['transform'],row,cfg,cal)
        check_equal(old['values'],a['values'],'original replay')
        check_equal(old['FL_weights'],a['FL_weights'],'original weights')
        c0=map_curve(a['curve'],t)
        c1=map_curve(b['curve'],t)
        frozen=[map_segment(s,t) for s in s0]
        decoded={}
        for arm,curve,pair,segments,n in [
            ('N',c1,b['selected_pair_id'],s1,raw1),
            ('FB',c0,a['selected_pair_id'],s1,raw1),
            ('FM',c1,b['selected_pair_id'],frozen,raw0),
            ('BOTH',c0,a['selected_pair_id'],frozen,raw0),
        ]:
            out=measure(curve,pair,segments,n,roi,t,row,cfg,cal)
            if arm=='N':
                for field in ('values','PA_weights','FL_weights'):
                    check_equal(out[field],b[field],f'factual {e["key"]} {e["condition"]} {name}')
            decoded[arm]=compact(out)
        delta,cat,parts=diagnose(e['original'],decoded)
        width=original['measurement_roi']['box'][2]-original['measurement_roi']['box'][0]
        audit=compare_boundary(a['curve'],b['curve'])
        weights=fixed_weight(compact(old),decoded['N'],width,row['px_per_mm'] or 1.)
        moved_span=roi['box'][2]-roi['box'][0]
        records.append(dict(**e,arms=decoded,differences=delta,category=cat,parts=parts,
                boundary=audit,weight_probe=weights,
                curves=dict(original=a['curve'],factual=b['curve']),
                roi_widths=dict(original=width,mapped=moved_span/t['sx']),
                canonical_both=old['values']['FL'],
                old_contributors=compact(old),original_image_path=row['image_path']))
    gzwrite(DEST/'fl_replay'/r['dataset']/(Path(r['image_id']).stem+'.json.gz'),records)
    return records


def conclude(records,manifest):
    classes={}
    for cohort in ('severe','controls'):
        items=[r for r in records if r['cohort']==cohort]
        q=dict(n=len(items),events=Counter(r['category'] for r in items),
               arms={a:dict(severe=sum(r['differences'][a] is not None and r['differences'][a]>.2 for r in items),
                            missing=sum(r['differences'][a] is None for r in items),
                            median=quant([r['differences'][a] for r in items],.5)) for a in ARMS})
        weights=[r for r in items if 'fixed_weight_change' in r['weight_probe']]
        q['weights']=dict(comparable=len(weights),fixed_changes_severe_to_nonsevere=sum(
            r['weight_probe']['factual_change']>.2>=r['weight_probe']['fixed_weight_change'] for r in weights),
            fixed_removes_half_change=sum(r['weight_probe']['fixed_weight_change']<
                                         r['weight_probe']['factual_change']*.5 for r in weights))
        q['large_boundary_shift']=sum(any(s['large_consistent_shift'] for s in r['boundary'].get('side',{}).values()) for r in items)
        q['max_both_residual']=max((r['differences']['BOTH'] for r in items if r['differences']['BOTH'] is not None),default=None)
        q['both_residual_gt_1pct']=sum(r['differences']['BOTH'] is not None and r['differences']['BOTH']>.01 for r in items)
        q['by_seed']={}
        for kind in KINDS:
            rr=[r for r in items if r['condition'].endswith('|'+kind)]
            q['by_seed'][kind]=dict(n=len(rr),categories=Counter(r['category'] for r in rr))
        classes[cohort]=q
    groups=defaultdict(list)
    for r in records:
        if r['cohort']=='severe':
            groups[r['key'],r['condition']].append(r)
    rescue={}
    for arm in ARMS:
        rescue[arm]=sum(all(r['differences'][arm] is not None and r['differences'][arm]<=.2 for r in rr)
                       for rr in groups.values())
    worst={k:max(rr,key=lambda r:r['original_difference']) for k,rr in groups.items()}
    case_groups={}
    for category in ('boundary_freeze_only','fascicle_freeze_only','either_freeze','joint_freeze_only','single_swap_missing','unresolved'):
        matches=sorted((r for r in worst.values() if r['category']==category),
                       key=lambda r:r['original_difference'],reverse=True)
        case_groups[category]=[dict(key=r['key'],condition=r['condition'],transform=r['transform'],
                                    differences=r['differences'],original=r['original'],
                                    values={a:r['arms'][a]['values']['FL'] for a in ARMS}) for r in matches[:8]]
    write(DEST/'fl_attribution_summary.json',dict(cohorts=classes,
          worst_event_categories=Counter(r['category'] for r in worst.values()),
          all_original_severe_events_removed_by_condition=rescue,examples=case_groups,
          control_same_dataset=sum(r['key'].split('|')[0]==r['control_for'].split('|')[0] for r in manifest['controls']),
          definition='A rescue means all originally severe views in that condition are now finite and <=20%. '
                     'This is targeted attribution, not a full variant coverage check or a deployment method. '
                     'BOTH uses transformed frame; the canonical original geometry+frame is independently replayed.'))


def quant(a,q):
    v=[x for x in a if x is not None]
    return float(np.quantile(v,q)) if v else None


def main(workers):
    event=event_manifest()
    for p,h in event['bindings'].items():
        assert sha(ROOT/p)==h,p
    index={key(r):r for r in rows()}
    work=defaultdict(list)
    for cohort in ('severe','controls'):
        for r in event[cohort]:
            work[r['key']].append(dict(r,cohort=cohort))
    outputs=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        tasks=[pool.submit(run_image,index[k],rs) for k,rs in work.items()]
        for i,f in enumerate(as_completed(tasks),1):
            outputs.extend(f.result())
            if i%10==0 or i==len(tasks):
                print('Round51 FL',i,'/',len(tasks),flush=True)
    assert len(outputs)==144
    conclude(outputs,event)
    print('Round51 FL completed',len(outputs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    main(a.workers)
