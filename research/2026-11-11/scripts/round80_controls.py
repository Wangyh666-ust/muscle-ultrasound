"""Label-based geometric controls, with selection on48 and frozen confirmation on624."""
from scripts.round80_common import *
from copy import deepcopy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time, math
from scipy.ndimage import gaussian_filter
from scripts.round75_controls import source_pair, synthetic_segments
from scripts.round63_paths import pack_path, rebuild, image_evidence, path_quality, choose_pair
from scripts.round64_seams import reconstruct as seams
from scripts.round65_core import keep
from scripts.round80_core import connect, fair_rank, fidelity_curve
from scripts.round77_layer import rank as oldrank, signature
from scripts.round56_boundary import fit_side
from scripts.round75_core import install_exact_geometry
from scripts.round29_core import extended_fl


def mirrored(p):
    x=511-np.asarray(p['x'])[::-1];y=np.asarray(p['y'])[::-1]
    return pack_path(x,y,np.asarray(p['confidence'])[::-1],np.asarray(p['observed'])[::-1],p['ids'])


def ridge(path,width=512):
    yy,xx=np.indices((width,width));curve=np.interp(np.arange(width),path['x'],path['y'])
    m=(np.arange(width)>=path['x'][0])&(np.arange(width)<=path['x'][-1])
    return .95*np.exp(-.5*((yy-curve[None,:])/1.5)**2)*m[None,:]


def path_slice(p,lo,hi,ids):
    m=(p['x']>=lo)&(p['x']<=hi)
    return pack_path(p['x'][m],p['y'][m],p['confidence'][m],p['observed'][m],ids)


def c_scenes(source,gray):
    for side in ('upper','lower'):
        for flip in (False,True):
            p=mirrored(source[side]) if flip else deepcopy(source[side]);lo,hi=p['x'][[0,-1]];span=hi-lo
            image=gray[:,::-1].copy() if flip else gray
            for kind in ('normal','clean_split','weak_middle','overlap_quantization','wrong_endpoint','parallel_layer','disjoint_layer'):
                field=ridge(p);valid=np.ones(field.shape,bool)
                if kind=='normal':paths=[p];negative=False
                else:
                    a=path_slice(p,lo,lo+.56*span,(11,));b=path_slice(p,lo+.46*span,hi,(12,));negative=kind.endswith('layer')
                    if kind=='weak_middle':
                        a=path_slice(p,lo,lo+.40*span,(11,));b=path_slice(p,lo+.60*span,hi,(12,));field[:,int(a['x'][-1])+1:int(b['x'][0])]*=.13
                    if kind=='overlap_quantization':
                        ea=a['x']>=lo+.44*span;eb=b['x']<=lo+.58*span
                        a['y'][ea]+=np.where(np.arange(ea.sum())%2,1.,-1.);b['y'][eb]+=np.where(np.arange(eb.sum())%2,-1.,1.)
                    if kind=='wrong_endpoint':
                        ea=a['x']>=lo+.48*span;eb=b['x']<=lo+.54*span
                        a['y'][ea]+=.025*512*np.linspace(0,1,ea.sum());b['y'][eb]-=.025*512*np.linspace(1,0,eb.sum())
                        field=np.maximum(field,np.maximum(ridge(a),ridge(b)))
                    if kind=='parallel_layer':b['y']+=.025*512;field=np.maximum(field,ridge(b))
                    if kind=='disjoint_layer':
                        a=path_slice(p,lo,lo+.42*span,(11,));b=path_slice(p,lo+.58*span,hi,(12,));b['y']+=.06*512;field=np.maximum(field,ridge(b))
                    paths=[a,b]
                yield side,flip,kind,p,paths,field,image,valid,negative


def coverage(path,truth):
    x=truth['x'];inside=(x>=path['x'][0])&(x<=path['x'][-1]);error=abs(np.interp(x,path['x'],path['y'])-truth['y'])
    return float(np.mean(inside&(error<=2.)))


def c_control(pair,gray,configs,fixed,base,scfg):
    rows=[]
    for side,flip,kind,truth,paths,field,image,valid,negative in c_scenes(pair,gray):
        rebuilt,_=rebuild(paths,field,image,valid,512,base);extra,_=seams(rebuilt,field,image,valid,512,scfg,base);oldpool=keep(rebuilt,extra,512)
        old=max(oldpool,key=lambda p:path_quality(p,512));before=coverage(old,truth)
        for ci,cfg in enumerate(configs):
            added,au=connect(rebuilt,field,image,valid,512,base,cfg,fixed,extra)
            best=max(oldpool+added,key=lambda p:path_quality(p,512));after=coverage(best,truth)
            bridge_fractions=[]
            for p in added:
                e=next(e for e in p['events'][::-1] if e['type']=='R80_C');xx=p['x'];mask=(xx>e['interval'][0])&(xx<e['interval'][1]);inside=(xx>=truth['x'][0])&(xx<=truth['x'][-1]);error=abs(p['y']-np.interp(xx,truth['x'],truth['y']))
                bridge_fractions.append(float(np.mean(inside[mask]&(error[mask]<=2.))) if mask.any() else 0.)
            rows.append(dict(side=side,flip=flip,kind=kind,config=ci,negative=negative,normal=kind in ('normal','clean_split'),before=before,after=after,added=len(added),
                             wrong_new_bridge=any(f<.95 for f in bridge_fractions),bridge_min_correct=min(bridge_fractions,default=None),audit=au))
    return rows


def r_control(pair,base):
    rows=[];lo,hi=pair['interval'];span=hi-lo;T=np.mean(np.interp(np.linspace(lo,hi,31),pair['lower']['x'],pair['lower']['y'])-np.interp(np.linspace(lo,hi,31),pair['upper']['x'],pair['upper']['y']))
    for kind in ('normal','partial_bonus_trap','partial_sparse_unknown','real_common_evidence'):
        u,d=deepcopy(pair['upper']),deepcopy(pair['lower']);u['ids']=(1,);d['ids']=(2,);u['confidence'][:]=.9;d['confidence'][:]=.9
        wrong=deepcopy(d);wrong['ids']=(3,);wrong['y']+=.2*T
        segs=synthetic_segments(pair)
        if kind in ('partial_bonus_trap','partial_sparse_unknown'):
            wrong=path_slice(wrong,lo,lo+.40*span,(3,))
            # Dense valid left fragments; right contains two extra fragments above the true band.
            segs=[]
            for i in range(10):
                cx=lo+(.08+.22*(i%5)/4)*span if i<8 else lo+.85*span
                y0=np.interp(cx,u['x'],u['y']);y1=np.interp(cx,d['x'],d['y']);cy=y0+.5*(y1-y0)
                if i>=8:cy=y0-.12*T
                half=.015*span
                segs.append(dict(label_id=i+1,centroid=np.array([cx,cy]),p0=np.array([cx-half,cy+half*.15]),p1=np.array([cx+half,cy-half*.15]),direction_deg=-8.5))
            if kind=='partial_sparse_unknown':segs=segs[:2]
        if kind=='real_common_evidence':
            # A shallower lower boundary excludes much of the true fragment body.
            wrong=deepcopy(d);wrong['ids']=(3,);wrong['y']-=.45*T
        _,pairs=choose_pair([[u],[d,wrong]],512,base,segs,None,True,False)
        if len(pairs)<2:continue
        # Deliberate small quality differences, matching the observed sub-.1 bonus sensitivity.
        for p in pairs:
            correct=p['lower']['ids']==(2,);bonus=.2*(p['c1']-.5) if p['c1_active'] else 0.
            p['score']=(.70 if correct else .66)+bonus
        if kind=='real_common_evidence':
            for p in pairs:
                bonus=.2*(p['c1']-.5) if p['c1_active'] else 0.;p['score']=(.66 if p['lower']['ids']==(2,) else .70)+bonus
        selected,au=fair_rank(pairs,[[u],[d,wrong]],np.zeros((3,512,512)),segs,512,[lo,hi],False)
        old,_=oldrank(pairs,[],None,segs,512,[lo,hi],None,False)
        correct=selected['lower']['ids']==(2,);oldgood=old['lower']['ids']==(2,)
        rows.append(dict(kind=kind,base_good=oldgood,new_good=correct,normal=kind=='normal',positive=kind=='partial_bonus_trap',unknown=au['reason'].startswith('C1_unknown'),
                         changed=signature(selected)!=signature(old),audit=au))
    return rows


def g_control(pair):
    install_exact_geometry();rows=[]
    for flip in (False,True):
        true={s:mirrored(pair[s]) if flip else deepcopy(pair[s]) for s in ('upper','lower')}
        lo=max(p['x'][0] for p in true.values());hi=min(p['x'][-1] for p in true.values())
        for kind in ('normal','upper_sparse','lower_sparse','both_sparse','unqualified_hole'):
            work=deepcopy(true);fitted={};unqualified=kind=='unqualified_hole'
            for side,role in [('upper','sup'),('lower','deep')]:
                p=work[side];p['qualified']=np.ones(len(p['x']),bool);p['observed']=np.ones(len(p['x']),bool)
                sparse=(side in kind) or kind in ('both_sparse','unqualified_hole')
                if sparse:
                    gap=(p['x']>lo+.15*(hi-lo))&(p['x']<lo+.37*(hi-lo));p['observed'][gap]=False
                    if unqualified:p['qualified'][gap]=False
                fitted[role]=fit_side(np.c_[p['x'][p['observed']],p['y'][p['observed']]],512,512)
            if any(f is None for f in fitted.values()):rows.append(dict(kind=kind,flip=flip,skipped=True));continue
            cv=dict(**fitted,support=[lo,hi],paired_x=np.intersect1d(true['upper']['x'],true['lower']['x']).tolist(),max_sample_gap=8.)
            got,au=fidelity_curve(work,cv,512,512);before=[];after=[];exact=True;rays=True
            for side,role in [('upper','sup'),('lower','deep')]:
                from scripts.round78_mt_core import qualified_at
                p=work[side];xx=np.unique(np.r_[p['x'],cv[role]['knots_x'],.5*(np.asarray(got[role]['knots_x'])[:-1]+np.asarray(got[role]['knots_x'])[1:])])
                q=np.array([qualified_at(p,float(x),float(np.median(np.diff(p['x']))))[0] for x in xx]);x=xx[q];y=np.interp(x,p['x'],p['y'])
                before.extend(abs(np.interp(x,cv[role]['knots_x'],cv[role]['knots_y'])-y));after.extend(abs(np.interp(x,got[role]['knots_x'],got[role]['knots_y'])-y))
                exact&=bool(np.max(abs(np.interp(x,got[role]['knots_x'],got[role]['knots_y'])-y),initial=0)<1e-8)
                rays&=all(got[role][k]==cv[role][k] for k in ('left_ray','right_ray'))
            missing=0
            # The true known curves use the same original terminal rays: isolate finite-geometry fidelity.
            truthcv=deepcopy(cv)
            for side,role in [('upper','sup'),('lower','deep')]:
                p=true[side];truthcv[role].update(knots_x=p['x'].tolist(),knots_y=p['y'].tolist(),_round75_exact=True,_round75_width=512,_round75_np_x=p['x'],_round75_np_y=p['y'])
            fl=[]
            for x in np.linspace(lo+.25*(hi-lo),hi-.25*(hi-lo),3):
                cy=.5*(np.interp(x,true['upper']['x'],true['upper']['y'])+np.interp(x,true['lower']['x'],true['lower']['y']))
                for angle in (-35.,35.):
                    seg=dict(centroid=[x,cy],direction_deg=angle,p0=[x-1,cy],p1=[x+1,cy],length_px=2.)
                    old=extended_fl(seg,cv,512,512).get('fl_px_raw');new=extended_fl(seg,got,512,512).get('fl_px_raw');truth=extended_fl(seg,truthcv,512,512).get('fl_px_raw')
                    missing+=(old is not None and new is None)
                    if old is not None and new is not None and truth is not None:fl.append(dict(before=abs(old-truth),after=abs(new-truth)))
            rows.append(dict(kind=kind,flip=flip,skipped=False,geometry_before=float(np.mean(before)),geometry_after=float(np.mean(after)),exact=exact,rays_unchanged=rays,new_missing_FL=missing,FL=fl,audit=au))
    return rows


def worker(sample,phase,configs):
    dst=OUT/f"controls/{phase}/{sample['image_id']}.json.gz"
    if dst.exists():return readgz(dst)
    pair=source_pair(sample)
    if pair is None:return dict(id=sample['image_id'],skipped='source_pair_not_available')
    pair,binding=pair
    with np.load(ROOT/binding['path']) as z:im=z['image'];im=im.mean(0) if im.ndim==3 and im.shape[0]==3 else im
    # Photometric evidence is the actual training image; only probability ridges/fragments are controlled.
    if im.ndim!=2:raise ValueError(('unexpected image shape',im.shape))
    gray=image_evidence(im,512);proto=j(OUT/'protocol.json');base=j(ROOT/'runs/round63_boundary_paths_v1/config.json');scfg=j(ROOT/'runs/round64_boundary_continuation_v1/controls/S_confirmation_lock.json')['selected']
    r=dict(id=sample['image_id'],group=sample['group_id'],binding=binding,C=c_control(pair,gray,configs['C'],proto['fixed_C'],base,scfg) if configs['C'] else [],
           R=r_control(pair,base) if configs['R'] else [],G=g_control(pair) if configs['G'] else [])
    write(dst,r);return r


def summarize(rows,configs):
    out={}
    for arm in ('C','R','G'):
        out[arm]=[]
        for ci,cfg in enumerate(configs[arm]):
            rs=[r for source in rows if not source.get('skipped') for r in source[arm] if r.get('config',ci)==ci]
            if arm=='C':
                normal=[r for r in rs if r['normal']];pos=[r for r in rs if not r['normal'] and not r['negative']];neg=[r for r in rs if r['negative']]
                gain=float(np.mean([r['after']-r['before'] for r in pos])) if pos else 0.;false=sum(r['added']>0 for r in neg);wrong=sum(r['wrong_new_bridge'] for r in pos);nc=sum(abs(r['after']-r['before'])>1e-10 for r in normal)
                summary=dict(config=cfg,cases=len(rs),positive=len(pos),mean_correct_coverage_gain=gain,negative_false_connections=false,wrong_positive_bridges=wrong,normal_changes=nc,added_cases=sum(r['added']>0 for r in rs),passed=bool(pos and gain>=.05 and false==0 and wrong==0 and nc==0))
            elif arm=='R':
                pos=[r for r in rs if r['positive'] and not r['base_good']];protected=[r for r in rs if r['base_good']];rec=sum(r['new_good'] for r in pos);reg=sum(not r['new_good'] for r in protected);nc=sum(r['changed'] for r in rs if r['normal'])
                summary=dict(config=cfg,cases=len(rs),positive=len(pos),recovered=rec,protected_regressions=reg,normal_changes=nc,unknown_cases=sum(r['unknown'] for r in rs),passed=bool(pos and rec>=.8*len(pos) and reg==0 and nc==0))
            else:
                valid=[r for r in rs if not r['skipped']];missing=sum(r['new_missing_FL'] for r in valid);exact=all(r['exact'] for r in valid);rays=all(r['rays_unchanged'] for r in valid)
                before=float(np.mean([r['geometry_before'] for r in valid]));after=float(np.mean([r['geometry_after'] for r in valid]));fl=[z for r in valid for z in r['FL']]
                summary=dict(config=cfg,cases=len(valid),finite_geometry_exact=exact,external_rays_unchanged=rays,geometry_MAE_before=before,geometry_MAE_after=after,new_missing_FL=missing,
                             controlled_FL_error_before=float(np.mean([r['before'] for r in fl])),controlled_FL_error_after=float(np.mean([r['after'] for r in fl])),passed=bool(valid and exact and rays and after<=before+1e-10 and missing==0))
            out[arm].append(summary)
    return out


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);args=ap.parse_args();proto=register();samples=j(ROOT/'runs/round60_boundary_objective_v1/payload/training_manifest.json')['samples'];phases={r['id']:r['phase'] for r in proto['controls']};selected={};start=time.monotonic()
    for phase in ('calibration','confirmation'):
        configs=proto['grids'] if phase=='calibration' else {a:([selected[a]['config']] if selected[a]['passed'] else []) for a in ('C','R','G')}
        wanted=[s for s in samples if phases[s['image_id']]==phase];rows=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(worker,s,phase,configs) for s in wanted]
            for f in as_completed(futures):
                rows.append(f.result())
                if len(rows)%24==0:print('R80_CONTROLS_PROGRESS',phase,len(rows),len(wanted),round(time.monotonic()-start),flush=True)
        summary=summarize(rows,configs);write(OUT/f'controls/{phase}.json',dict(sources=len(rows),skipped=[r for r in rows if r.get('skipped')],configs=configs,summary=summary))
        if phase=='calibration':
            for a,ss in summary.items():
                selected[a]=min(ss,key=lambda r:(not r['passed'],-r.get('mean_correct_coverage_gain',0),r['config'].get('endpoint_window',0)))
            write(OUT/'parameter_lock.json',dict(selected=selected,before_confirmation=True,validation112_used=False))
        else:
            admitted={a:bool(selected[a]['passed'] and summary[a] and summary[a][0]['passed']) for a in ('C','R','G')}
            arms=[a for a in ARMS if a=='B' or all(admitted[f] for f in a)]
            write(OUT/'execution_lock.json',dict(admitted=admitted,parameters={a:selected[a]['config'] for a in selected},arms=arms,parameter_lock_sha=sha(OUT/'parameter_lock.json'),validation112_used=False,control_files_verified=True))
        print('R80_CONTROLS',phase,summary,flush=True)
    verify_default();print('R80_LOCKED',arms,admitted,flush=True)


if __name__=='__main__':main()
