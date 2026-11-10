"""Excluded tips cannot fit, validate or anchor a continuation. No GT inputs."""
from copy import deepcopy
import numpy as np
from scripts.round66_geometry import fit
from scripts.round71_geometry import forecast
from scripts.round72_geometry import complete as r72_complete,long_models,crop

FIELDS=('x','y','kind','observed','confidence','anchor_eligible','source_qualified')
BASE_CFG=dict(endpoint_z=3.,decay_width=.1)

def baseline(path,domain,width):return r72_complete(path,domain,width,BASE_CFG,True,True)

def core(path,right,width,cfg):
    x,y=np.asarray(path['x']),np.asarray(path['y']);direction=1 if right else -1
    edge=x[-1 if right else 0];distance=direction*(edge-x)/width
    eligible=np.asarray(path.get('anchor_eligible',path.get('observed',np.ones(len(x),bool))),bool)
    conf=np.asarray(path.get('confidence',np.full(len(x),.9)))
    tail=eligible&(distance<=.005)
    if tail.sum()<3:tail=eligible&(distance<=.015)
    weak=bool(tail.any() and np.mean(conf[tail])<.5)
    excluded=(distance<cfg['quarantine_fraction']-1e-10) if weak else np.zeros(len(x),bool)
    allowed=eligible&~excluded&(conf>=.5)
    indices=np.flatnonzero(allowed)
    if len(indices)<24:return None,dict(reason='no_reliable_core',weak=weak)
    index=indices[-1 if right else 0];start=x[index]
    # Preserve the whole array for source auditing, but models receive reliable points only.
    q=crop(path,allowed)
    assert not np.any(excluded[allowed])
    return q,dict(reason='weak_tip_isolated' if weak else 'confident_core',weak=weak,original_edge=float(edge),
        stable_x=float(start),stable_y=float(y[index]),excluded_indices=np.flatnonzero(excluded).tolist(),
        fit_source_indices=indices.tolist(),quarantined=int(excluded.sum()),direction=direction)

def reliable_long_models(path,right,width,cfg):
    x,y=np.asarray(path['x']),np.asarray(path['y']);direction=1 if right else -1
    edge=x[-1 if right else 0];origin=y[-1 if right else 0];t=direction*(x-edge)/width
    order=np.argsort(t);t,y=t[order],y[order]
    conf=np.asarray(path.get('confidence',np.full(len(x),.9)))[order]
    sigma=max(.35,2*width/1046.);bank=[];seen=set()
    for window in (.4,.6,.8,1.):
        use=t>=-window-1e-10
        if use.sum()<24 or np.ptp(t[use])<.65*window:continue
        key=tuple(np.flatnonzero(use))
        if key in seen:continue
        seen.add(key);tt,yy,ww=t[use],y[use],conf[use]
        for degree in (1,2):
            coef=fit(tt,yy,ww,degree);res=float(np.quantile(abs(np.polynomial.polynomial.polyval(tt,coef)-yy),.9))
            cvs=[];cv_indices=[]
            for fraction in (.1,.2):
                cut=tt.max()-fraction*np.ptp(tt);train=tt<=cut;held=~train
                if train.sum()<12 or held.sum()<3:continue
                at=tt[train][-1];value=yy[train][-1]
                c=fit(tt[train]-at,yy[train],ww[train],degree)
                anchor=dict(x=0.,y=float(value),direction=1,width=float(width),models=[dict(coef=c)],sigma=sigma,anchor_error=0.)
                predicted,_=forecast(anchor,(tt[held]-at)*width,cfg)
                cvs.append(float(np.quantile(abs(predicted-yy[held]),.9)))
                cv_indices.append(dict(training=int(train.sum()),held=int(held.sum())))
            if len(cvs)!=2:continue
            cv=max(cvs);score=cv+.25*res+.1*sigma*(degree-1)
            bank.append(dict(coef=coef,window=window,actual_span=float(np.ptp(tt)),degree=degree,hold=cv,residual=res,score=score,
                cv_errors=cvs,cv_sizes=cv_indices,qualified=bool(cv<=2*sigma and res<=2*sigma)))
    if not bank:return None
    valid=[b for b in bank if b['qualified']];pool=valid or bank;best=min(m['score'] for m in pool)
    near=[m for m in pool if m['score']<=best+.5*sigma]
    chosen=min(near,key=lambda m:(-round(m['actual_span'],10),m['score'],m['degree']))
    return dict(x=float(edge),y=float(origin),direction=direction,width=float(width),models=[chosen],sigma=sigma,
        anchor_error=max(chosen['hold'],chosen['residual']),stable=bool(valid),long_core=True,
        scoring='two_actual_forecast_holdouts_then_longest_compatible',bank=bank)

def complete(path,domain,width,cfg,arm):
    out,old=baseline(path,domain,width)
    if out is None or arm=='R':return out,dict(reason='frozen_R72_PH',stable_core_preserved=True,endpoint=[])
    original_x=np.asarray(path['x']);original_y=np.asarray(path['y']);target=np.asarray(out['x']);events=[]
    allowed=np.zeros(len(original_x),bool)
    for right in (False,True):
        oldedge=original_x[-1 if right else 0]
        outside=(target>oldedge) if right else (target<oldedge)
        if not outside.any():continue
        clean,proof=core(path,right,width,cfg)
        if clean is None:events.append(dict(reason='no_reliable_core',direction=1 if right else -1));continue
        anchor=(reliable_long_models if arm=='SL' else long_models)(clean,right,width,dict(cfg,endpoint_z=3.))
        if anchor is None:events.append(dict(reason='no_model',direction=1 if right else -1,proof=proof));continue
        stable_edge=anchor['x'];edit=(target>stable_edge) if right else (target<stable_edge)
        if arm=='A':
            # Deliberately isolate fitting while retaining the wrong old anchor: mechanism ablation only.
            shift=anchor['direction']*(oldedge-anchor['x'])/width
            for m in anchor['models']:
                c=m['coef'].copy();c[1]+=2*c[2]*shift;m['coef']=c
            anchor.update(x=float(oldedge),y=float(original_y[-1 if right else 0]));edit=outside
        predicted,uncertainty=forecast(anchor,target[edit],cfg)
        out['y'][edit]=predicted;out['uncertainty'][edit]=np.maximum(uncertainty,100.*anchor['sigma'] if not anchor['stable'] else 0.)
        prior=(target>=original_x[0])&(target<=original_x[-1]);rebuilt=edit&prior
        out['kind'][edit&~prior]=5
        out['kind'][rebuilt]=6;out['observed'][rebuilt]=False;out['qualified_source'][rebuilt]=False
        if arm!='A':allowed|=(original_x>stable_edge) if right else (original_x<stable_edge)
        event=dict(direction=1 if right else -1,proof=proof,anchor_x=anchor['x'],anchor_y=anchor['y'],
            stable=anchor['stable'],scoring=anchor.get('scoring'),windows=[m['window'] for m in anchor['models']],
            degrees=[m['degree'] for m in anchor['models']],hold=[m['hold'] for m in anchor['models']],
            residual=[m['residual'] for m in anchor['models']],reconstructed=int(rebuilt.sum()),
            output_interval=[float(target[edit].min()),float(target[edit].max())])
        events.append(event)
    # Start from the old source, not R72's possibly edited tip; only new registered windows may move it.
    source_at=np.searchsorted(target,original_x);keep=~allowed
    out['y'][source_at[keep]]=original_y[keep]
    # Restoring source geometry must also restore its source labels.
    out['kind'][source_at[keep]]=np.asarray(path.get('kind',np.zeros(len(original_x),int)))[keep]
    out['observed'][source_at[keep]]=np.asarray(path.get('observed',np.ones(len(original_x),bool)))[keep]
    out['qualified_source'][source_at[keep]]=np.asarray(path.get('source_qualified',np.ones(len(original_x),bool)))[keep]
    changed=abs(out['y'][source_at]-original_y)>1e-8
    assert not np.any(changed&~allowed),'core changed outside proposed reconstruction window'
    return out,dict(reason='completed',endpoint=events,changed_original=int(changed.sum()),stable_core_preserved=True)
