"""GT-free stable-window completion. Speculation is never promoted to observation."""
from copy import deepcopy
import numpy as np
from scripts.round66_geometry import fit
from scripts.round63_paths import hermite


def models(path,right,width,cfg):
    x,y=np.asarray(path['x']),np.asarray(path['y']);direction=1 if right else -1;at=-1 if right else 0
    edge=x[at];origin=y[at];t=direction*(x-edge)/width
    eligible=np.asarray(path.get('anchor_eligible',path.get('observed',np.ones(len(x),bool))),bool)
    conf=np.asarray(path.get('confidence',np.ones(len(x))*.9));bank=[]
    for fraction in (.06,.10,.18,.28,.40,.60):
        use=(t>=-fraction-1e-10)&eligible
        if use.sum()<12 or np.ptp(t[use])<.65*fraction:continue
        tt,yy,ww=t[use],y[use],conf[use];train=tt<=-.2*fraction;held=~train
        if train.sum()<8 or held.sum()<3:continue
        for degree in (1,2):
            c=fit(tt,yy,ww,degree);v=fit(tt[train],yy[train],ww[train],degree)
            hold=float(np.quantile(abs(np.polynomial.polynomial.polyval(tt[held],v)-yy[held]),.9))
            res=float(np.quantile(abs(np.polynomial.polynomial.polyval(tt,c)-yy),.9))
            score=hold+res+abs(c[0]-origin)
            bank.append(dict(coef=c,degree=degree,window=fraction,hold=hold,residual=res,score=score))
    # Poor anchors still produce explicitly uncertain shape estimates, not a false absence of prediction.
    if not bank and eligible.sum()>=8:
        indices=np.flatnonzero(eligible);take=np.zeros(len(x),bool)
        take[indices[-max(8,len(indices)//2):] if right else indices[:max(8,len(indices)//2)]]=True
        c=fit(t[take],y[take],np.maximum(conf[take],.05),1)
        residual=float(np.quantile(abs(np.polynomial.polynomial.polyval(t[take],c)-y[take]),.9))
        bank=[dict(coef=c,degree=1,window=float(np.ptp(t[take])),hold=residual,residual=residual,score=residual+abs(c[0]-origin),fallback=True)]
    if not bank:return None
    # Compare linear/quadratic at the same window, penalizing curvature unless it improves held-out prediction.
    chosen=[]
    for window in sorted({m['window'] for m in bank}):
        group=[m for m in bank if m['window']==window];linear=next((m for m in group if m['degree']==1),None)
        quadratic=next((m for m in group if m['degree']==2),None)
        winner=linear or quadratic
        if linear and quadratic and quadratic['hold']<.75*linear['hold'] and quadratic['score']<linear['score']:winner=quadratic
        chosen.append(winner)
    best=min(m['score'] for m in chosen);sigma=max(.35,2*width/1046.)
    selected=[m for m in chosen if m['score']<=best+sigma]
    selected=sorted(selected,key=lambda m:(m['score'],-m['window']))[:3]
    # Uncertainty records selection weakness rather than recursively using newly imputed points as anchors.
    return dict(x=float(edge),y=float(origin),direction=direction,width=float(width),models=selected,
                anchor_error=float(np.median([m['score'] for m in selected])),
                stable=bool(len(selected)>=2 and max(m['hold'] for m in selected)<=2*sigma),sigma=sigma)


def forecast(anchor,x,cfg):
    x=np.asarray(x,float);d=anchor['direction']*(x-anchor['x']);w=anchor['width'];t=d/w;L=cfg['decay_width']
    values=[]
    for model in anchor['models']:
        c=model['coef'];curvature=2*c[2]
        values.append(anchor['y']+c[1]*t+curvature*L*(t-L*(1-np.exp(-t/L))))
    values=np.asarray(values);y=np.median(values,axis=0)
    spread=np.ptp(values,axis=0)
    u=anchor['sigma']+anchor['anchor_error']*(1+d/max(.1*w,1))+spread+.008*d+.015*d*d/w
    return y,u


def complete(path,domain,width,cfg,gaps=True,ends=True):
    if path is None:return None,dict(reason='no_anchor_path')
    p=deepcopy(path);x,y=np.asarray(p['x'],float),np.asarray(p['y'],float)
    if len(x)<8:return None,dict(reason='insufficient_anchor_points')
    lo,hi=domain
    target=np.unique(np.r_[x,np.arange(np.ceil(lo),np.floor(hi)+1)])
    # Preserve all old coordinates even if the independent domain is smaller; never hide lost extent.
    result=np.interp(target,x,y);kind=np.full(len(target),-1,int);uncertainty=np.zeros(len(target));events=[]
    exact=np.searchsorted(target,x);kind[exact]=np.asarray(p.get('kind',np.zeros(len(x),int)))
    obs=np.zeros(len(target),bool);obs[exact]=p.get('observed',np.ones(len(x),bool))
    visible=np.zeros(len(target),bool);visible[exact]=True;qualified=visible.copy()
    count_gaps=0
    for i in np.flatnonzero(np.diff(x)>1.5):
        use=(target>x[i])&(target<x[i+1]);qualified[use]=False
        if not gaps:continue
        left={k:(np.asarray(v)[:i+1] if k in ('x','y','kind','observed','confidence','anchor_eligible') else v) for k,v in p.items()}
        right={k:(np.asarray(v)[i+1:] if k in ('x','y','kind','observed','confidence','anchor_eligible') else v) for k,v in p.items()}
        a,b=models(left,True,width,cfg),models(right,False,width,cfg)
        if a is None or b is None:continue
        ma=float(np.median([m['coef'][1] for m in a['models']]))/width
        mb=-float(np.median([m['coef'][1] for m in b['models']]))/width
        xx=target[use];yy,_,_=hermite(xx,x[i],y[i],ma,x[i+1],y[i+1],mb)
        pa,ua=forecast(a,xx,cfg);pb,ub=forecast(b,xx,cfg)
        conflict=max(abs(float(forecast(a,[x[i+1]],cfg)[0][0])-y[i+1]),abs(float(forecast(b,[x[i]],cfg)[0][0])-y[i]))
        phase=(xx-x[i])/(x[i+1]-x[i]);uu=np.minimum(ua,ub)+abs(pa-pb)*4*phase*(1-phase)
        if conflict>max(4*a['sigma'],.025*width):uu+=conflict
        result[use]=yy;kind[use]=4;uncertainty[use]=uu;count_gaps+=1
        events.append(dict(type='gap_completion',interval=[float(x[i]),float(x[i+1])],shape_conflict_px=float(conflict)))
    for right in (False,True):
        use=(target>x[-1]) if right else (target<x[0]);qualified[use]=False
        if not use.any() or not ends:continue
        anchor=models(p,right,width,cfg)
        if anchor is None:continue
        result[use],uncertainty[use]=forecast(anchor,target[use],cfg);kind[use]=5
        events.append(dict(type='endpoint_completion',direction=1 if right else -1,interval=[float(target[use].min()),float(target[use].max())],
            windows=[m['window'] for m in anchor['models']],degrees=[m['degree'] for m in anchor['models']],
            stable_anchor=anchor['stable'],anchor_error=anchor['anchor_error']))
    finite=kind>=0
    # Missing portions remain missing when that ablation's mechanism is disabled.
    out=dict(x=target[finite],y=result[finite],kind=kind[finite],observed=obs[finite],
             uncertainty=uncertainty[finite],existing=visible[finite],qualified_source=qualified[finite],
             ids=list(p.get('ids',[])),events=list(p.get('events',[]))+events)
    assert np.array_equal(np.interp(x,out['x'],out['y']),y),'existing geometry changed'
    return out,dict(reason='completed',new_columns=int((~visible&finite).sum()),gap_count=count_gaps,events=events)
