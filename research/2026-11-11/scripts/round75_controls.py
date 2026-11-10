"""Training-source geometric controls, parameter choice before validation replay."""
from copy import deepcopy
import math,time
from pathlib import Path
import numpy as np
from scripts.round75_common import *
from scripts.round53_experiment import initialize_worker
from scripts.round63_paths import pack_path,choose_pair
from scripts.round56_boundary import fit_side
from scripts.round29_core import extended_fl
from scripts.round75_core import rank_density,exact_adapter,install_exact_geometry

def source_pair(sample):
    v=sample['variants'][0];path=ROOT/'runs/round60_boundary_objective_v1/payload'/v['path'];assert sha(path)==v['sha256']
    with np.load(path) as z:t=z['target'].astype(bool)&z['known'].astype(bool)[None]
    sides=[]
    for i in (0,1):
        x=np.flatnonzero(t[i].any(0));y=(t[i].argmax(0)[x]+511-t[i,::-1].argmax(0)[x])/2
        if len(x)<32:return None
        sides.append(pack_path(x.astype(float),y,.95+np.zeros(len(x)),np.ones(len(x),bool),[i+1]))
    lo=max(p['x'][0] for p in sides);hi=min(p['x'][-1] for p in sides)
    if hi-lo<80:return None
    xx=np.linspace(lo,hi,81);T=np.interp(xx,sides[1]['x'],sides[1]['y'])-np.interp(xx,sides[0]['x'],sides[0]['y'])
    if np.min(T)<=2:return None
    return dict(upper=sides[0],lower=sides[1],interval=[lo,hi]),dict(path=path.relative_to(ROOT).as_posix(),sha=v['sha256'])

def synthetic_segments(pair,cluster=False,outliers=False):
    lo,hi=pair['interval'];out=[]
    for i in range(12):
        cx=lo+(.08+.84*(i%6)/5)*(hi-lo);u=np.interp(cx,pair['upper']['x'],pair['upper']['y']);d=np.interp(cx,pair['lower']['x'],pair['lower']['y'])
        f=.42+.16*(i//6) if cluster else .15+.7*(i%4)/3
        if outliers and i in (0,5):f=-.2
        cy=u+f*(d-u);half=.035*(hi-lo);delta=np.array([half,-.15*half])
        out.append(dict(label_id=i+1,centroid=np.array([cx,cy]),p0=np.array([cx,cy])-delta,p1=np.array([cx,cy])+delta,direction_deg=math.degrees(math.atan(-.15))%180))
    return out

def scores_control(pair,cfg,kind):
    lo,hi=pair['interval'];width=hi-lo;xx=np.linspace(lo,hi,101);T=np.median(np.interp(xx,pair['lower']['x'],pair['lower']['y'])-np.interp(xx,pair['upper']['x'],pair['upper']['y']));scale=T/20
    upper=deepcopy(pair['upper']);lower=deepcopy(pair['lower']);upper['confidence'][:]=.90;lower['confidence'][:]=.90
    if kind=='normal':other=deepcopy(upper);other['y']+=.2;other['ids']=(3,);pool=[[upper,other],[lower]];segments=synthetic_segments(pair)
    elif kind=='narrow_cluster':
        nu=deepcopy(upper);nd=deepcopy(lower);nu['y']+=.25*T;nd['y']-=.25*T;nu['ids']=(3,);nd['ids']=(4,)
        pool=[[upper,nu],[lower,nd]];segments=synthetic_segments(pair,True)
    else:
        other=deepcopy(upper);other['y']-=.65*T;other['ids']=(3,);other['confidence'][:]=.96
        if kind=='short_correct':
            keep=(upper['x']>=lo+.12*width)&(upper['x']<=hi-.12*width)
            for k in ('x','y','confidence','observed','kind'):upper[k]=upper[k][keep]
        pool=[[upper,other],[lower]];segments=synthetic_segments(pair,outliers=True)
    base,pairs=choose_pair(pool,width,j(ROOT/'runs/round63_boundary_paths_v1/config.json'),segments,scale,True,True)
    selected,audit=rank_density(pairs,segments,[lo,hi],cfg)
    # Existing upper/lower source IDs define the target band; near-normal shift is tolerated.
    def good(p):return (kind=='normal') or (p['upper']['ids']==(1,) and p['lower']['ids']==(2,))
    identity_cfg={**cfg,'identity_D':True};same,au=rank_density(pairs,segments,[lo,hi],identity_cfg)
    assert all(abs(q['new_score']-q['original']['score'])<1e-12 for q in au)
    assert same['upper']['ids']==base['upper']['ids'] and same['lower']['ids']==base['lower']['ids']
    return dict(kind=kind,base_good=good(base),new_good=good(selected),changed=(base['upper']['ids'],base['lower']['ids'])!=(selected['upper']['ids'],selected['lower']['ids']),identity_exact=True)

def Q_control(pair):
    # Real labelled shape; only provenance/support flags are intervened on.
    p=deepcopy(pair);lo,hi=p['interval'];pos=lo+.75*(hi-lo)
    for side in ('upper','lower'):
        a=p[side];a['qualified']=np.ones(len(a['x']),bool)
    m=abs(p['lower']['x']-pos)<.12*(hi-lo);p['lower']['observed'][m]=False;p['lower']['kind'][m]=3
    old=[]
    for side in ('upper','lower'):
        a=pair[side];fit=fit_side(np.c_[a['x'],a['y']],512,512)
        if fit is None:return dict(skipped=True)
        old.append(fit)
    legacy=dict(sup=old[0],deep=old[1],support=[lo,hi],paired_x=np.intersect1d(pair['upper']['x'],pair['lower']['x']).tolist(),max_sample_gap=7.68)
    q,audit=exact_adapter(p,legacy,512,512)
    if q is None:return dict(skipped=True)
    probe=np.linspace(lo,hi,5)[1:-1];newgood=bool(np.all(np.min(abs(probe[:,None]-np.asarray(q['paired_x'])),axis=1)<=7.68))
    raw=np.intersect1d(p['upper']['x'][p['upper']['observed']],p['lower']['x'][p['lower']['observed']]);oldgood=bool(len(raw) and np.all(np.min(abs(probe[:,None]-raw),axis=1)<=7.68))
    # Unsupported continuation must not become support merely because it is drawable.
    bad=deepcopy(p);bad['lower']['qualified'][m]=False;b,_=exact_adapter(bad,legacy,512,512)
    badpassed=bool(b and np.min(abs(probe[-1]-np.asarray(b['paired_x'])))<=7.68)
    identical=all(np.array_equal(q[k]['knots_y'],p[s]['y']) and np.array_equal(q[k]['knots_x'],p[s]['x']) for k,s in [('sup','upper'),('deep','lower')])
    return dict(skipped=False,recovered=newgood and not oldgood,eligible=not oldgood,negative_false_support=badpassed,
                exact_display_geometry=identical,source_flags_unchanged=True)

def F_control(pair,taus):
    # Geometry errors are measured against labelled finite boundaries, not synthetic hidden anatomy.
    from scripts.round27_geometry import roots_on_curve
    lo,hi=pair['interval'];width=hi-lo;fitted=[]
    for side in ('upper','lower'):
        p=pair[side];fit=fit_side(np.c_[p['x'],p['y']],512,512)
        if fit is None:return None
        fitted.append(fit)
    curve=dict(sup=fitted[0],deep=fitted[1],support=[lo,hi],paired_x=np.intersect1d(pair['upper']['x'],pair['lower']['x']).tolist(),max_sample_gap=7.68)
    # Crop observed support so held-out source boundary columns provide a known continuation answer.
    partial=deepcopy(curve)
    for role,side in [('sup','upper'),('deep','lower')]:
        p=pair[side];m=(p['x']>=lo+.12*width)&(p['x']<=hi-.12*width);fit=fit_side(np.c_[p['x'][m],p['y'][m]],512,512)
        if fit is None:return None
        partial[role]=fit
    segments=[];truth=[]
    for i,theta in enumerate((-20.,-30.,-40.,20.,30.,40.)):
        cx=lo+(i%3+1)/4*width;cy=.5*(np.interp(cx,pair['upper']['x'],pair['upper']['y'])+np.interp(cx,pair['lower']['x'],pair['lower']['y']))
        seg=dict(label_id=i+1,centroid=[cx,cy],p0=[cx-3,cy],p1=[cx+3,cy],direction_deg=theta%180)
        rs=[roots_on_curve(seg,dict(knots_x=pair[s]['x'],knots_y=pair[s]['y']))['roots'] for s in ('upper','lower')]
        if all(len(q)==1 for q in rs):
            length=np.linalg.norm(np.subtract(rs[0][0]['point'],rs[1][0]['point']));segments.append(seg);truth.append(length)
    if len(segments)<3:return None
    result=[]
    for damaged in (False,True):
        cv=deepcopy(partial)
        if damaged:
            m,b=cv['sup']['left_ray'];x=cv['sup']['knots_x'][0];y=cv['sup']['knots_y'][0];m=math.tan(math.atan(m)+math.radians(5));cv['sup']['left_ray']=[m,y-m*x]
        vals=[];true=[];uncert=[]
        for s,t in zip(segments,truth):
            base=extended_fl(s,cv,512,512)['fl_px_raw']
            if base is None:continue
            probes=[]
            for da in (-1.,1.):
                p=deepcopy(s);p['direction_deg']+=da;v=extended_fl(p,cv,512,512)['fl_px_raw']
                if v is not None:probes.append(v)
            if not probes:continue
            vals.append(base);true.append(t);uncert.append(max(abs(np.asarray(probes)-base)))
        if len(vals)<3:continue
        vals,true,u=map(np.asarray,(vals,true,uncert));old=1/(1+(u/max(width,1)/.03)**2);old/=old.sum()
        for tau in taus:
            weights=old/(1+(u/np.maximum(vals,1)/tau)**2);weights/=weights.sum()
            result.append(dict(tau=tau,damaged=damaged,old_error=float(old@abs(vals-true)),new_error=float(weights@abs(vals-true)),
                 normal_relative_output_change=float(abs(weights@vals-old@vals)/max(abs(old@vals),1)),samples=len(vals)))
    return result

def run():
    initialize_worker();install_exact_geometry();tic=time.monotonic()
    samples=j(ROOT/'runs/round60_boundary_objective_v1/payload/training_manifest.json')['samples']
    reviewed={r['image_id'] for r in j(ROOT/'thinking/ROUND62_PILOT_REVIEW_QUEUE_20261109.json')['cases']}
    grids=[dict(normalizer=n,contrast_guard=g,minimum_common=.12) for n in ('ratio','contrast') for g in (1.10,1.25)]
    taus=(.03,.06,.10);allrows=[];locks=None
    for phase in ('calibration','confirmation'):
        S=[];Q=[];F=[];sourceids=[]
        for s in samples:
            if (s['image_id'] in reviewed)!=(phase=='calibration'):continue
            material=source_pair(s)
            if material is None:continue
            pair,binding=material;sourceids.append(dict(id=s['image_id'],group=s['group_id'],**binding))
            for ci,cfg in enumerate(grids if locks is None else ([locks['S']['config']] if locks['S']['passed'] else [])):
                for kind in ('normal','broad','short_correct','narrow_cluster'):
                    S.append(dict(id=s['image_id'],config=ci,**scores_control(pair,cfg,kind)))
            Q.append(dict(id=s['image_id'],**Q_control(pair)))
            f=F_control(pair,taus if locks is None else [locks['F']['tau']]) if locks is None or locks['F']['passed'] else None
            F.extend([dict(id=s['image_id'],**x) for x in (f or [])])
        summaries=[]
        for ci,cfg in enumerate(grids if locks is None else ([locks['S']['config']] if locks['S']['passed'] else [])):
            ss=[q for q in S if q['config']==ci];neg=[q for q in ss if q['kind'] in ('normal','narrow_cluster') and q['base_good']];pos=[q for q in ss if not q['base_good']]
            rate=sum(q['new_good'] for q in pos)/len(pos) if pos else 1.
            summaries.append(dict(config=cfg,positive=len(pos),recovered=sum(q['new_good'] for q in pos),rate=rate,normal_regressions=sum(not q['new_good'] for q in neg),
                  normal_changes=sum(q['changed'] for q in ss if q['kind']=='normal'),passed=rate>=.8 and all(q['new_good'] for q in neg) and not any(q['changed'] for q in ss if q['kind']=='normal')))
        qq=[q for q in Q if not q.get('skipped')];eligible=[q for q in qq if q['eligible']]
        qs=dict(eligible=len(eligible),recovered=sum(q['recovered'] for q in eligible),negative_false_support=sum(q['negative_false_support'] for q in qq),
           exact_geometry=all(q['exact_display_geometry'] for q in qq),passed=bool(eligible) and np.mean([q['recovered'] for q in eligible])>=.95 and not any(q['negative_false_support'] for q in qq))
        fs=[]
        for tau in (taus if locks is None else ([locks['F']['tau']] if locks['F']['passed'] else [])):
            ff=[q for q in F if q['tau']==tau];normal=[q for q in ff if not q['damaged']];damage=[q for q in ff if q['damaged']]
            before=float(np.mean([q['old_error'] for q in damage]));after=float(np.mean([q['new_error'] for q in damage]));change=float(np.quantile([q['normal_relative_output_change'] for q in normal],.95))
            fs.append(dict(tau=tau,n=len(ff),damaged_old_error=before,damaged_new_error=after,normal_p95_relative_change=change,passed=after<before and change<=.01))
        report=dict(phase=phase,sources=sourceids,S=summaries,Q=qs,F=fs,seconds=time.monotonic()-tic)
        write(OUT/f'controls/{phase}.json',report);write(OUT/f'controls/{phase}_rows.json.gz',dict(S=S,Q=Q,F=F))
        print('R75_CONTROLS',phase,'sources',len(sourceids),'S',summaries,'Q',qs,'F',fs,flush=True)
        if phase=='calibration':
            bestS=sorted(summaries,key=lambda q:(not q['passed'],q['normal_regressions'],-q['rate'],q['config']['normalizer']!='ratio'))[0]
            bestF=sorted(fs,key=lambda q:(not q['passed'],q['normal_p95_relative_change'],q['damaged_new_error']))[0]
            locks=dict(S=bestS,Q=qs,F=bestF,training=False,default_changed=False,selected_before_confirmation=True)
            write(OUT/'parameter_lock.json',locks)
        else:
            locks['confirmation']=dict(S=summaries[0]['passed'] if summaries else False,Q=qs['passed'],F=fs[0]['passed'] if fs else False)
            locks['admitted']={k:bool(locks[k]['passed'] and locks['confirmation'][k]) for k in ('S','Q','F')}
            write(OUT/'execution_lock.json',locks)
    print('R75_PARAMETERS_LOCKED',locks['admitted'],flush=True)

if __name__=='__main__':run()
