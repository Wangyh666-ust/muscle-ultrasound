"""Common long-core scoring and conditional endpoint repair. No GT accepted."""
from copy import deepcopy
import numpy as np
import hashlib
from scripts.round66_geometry import fit
from scripts.round71_geometry import complete as old_complete,models as old_models,forecast

FIELDS=('x','y','kind','observed','confidence','anchor_eligible','source_qualified')

def memoize(mode):
    cache={}
    def decorate(function):
        def wrapped(path,side_or_domain,width,cfg):
            h=hashlib.sha256()
            for name in FIELDS:
                a=np.asarray(path.get(name,[]));h.update(str((name,a.dtype.str,a.shape)).encode());h.update(a.tobytes())
            h.update(repr(path.get('ids',[])).encode());h.update(repr(path.get('events',[])).encode())
            key=(h.digest(),tuple(side_or_domain) if mode=='old' else bool(side_or_domain),float(width),cfg['decay_width'],cfg['endpoint_z'] if mode=='proof' else 0.)
            if key not in cache:
                if len(cache)>=256:cache.clear()
                cache[key]=function(path,side_or_domain,width,cfg)
            return deepcopy(cache[key])
        return wrapped
    return decorate

@memoize('old')
def frozen_completion(path,domain,width,cfg):return old_complete(path,domain,width,cfg,True,True)

def crop(path,mask):
    return {k:np.asarray(v)[mask] if k in FIELDS else deepcopy(v) for k,v in path.items()}

def canonical(path,right,width):
    x,y=np.asarray(path['x']),np.asarray(path['y']);direction=1 if right else -1;edge=x[-1 if right else 0]
    t=direction*(x-edge)/width;order=np.argsort(t)
    eligible=np.asarray(path.get('anchor_eligible',path.get('observed',np.ones(len(x),bool))),bool)
    confidence=np.asarray(path.get('confidence',np.full(len(x),.9)))
    return t[order],y[order],confidence[order],eligible[order],float(edge),float(y[-1 if right else 0])

@memoize('models')
def long_models(path,right,width,cfg):
    t,y,w,eligible,edge,origin=canonical(path,right,width);sigma=max(.35,2*width/1046.)
    common=eligible&(t<=-.08)&(t>=-.60)
    if common.sum()<24 or np.ptp(t[common])<.18:
        a=old_models(path,right,width,cfg)
        if a:a.update(scoring='short_fallback_no_long_core',long_core=False)
        return a
    bank=[]
    for window in (.18,.28,.40,.60):
        use=eligible&(t>=-window-1e-10)
        if use.sum()<12 or np.ptp(t[use])<.65*window:continue
        training=use&(t<=-.2*window);held=use&~training
        if training.sum()<8 or held.sum()<3:continue
        for degree in (1,2):
            c=fit(t[use],y[use],w[use],degree);v=fit(t[training],y[training],w[training],degree)
            err=abs(np.polynomial.polynomial.polyval(t[common],c)-y[common])
            common_error=float(np.mean(np.minimum(err,np.quantile(err,.90))))
            hold=float(np.quantile(abs(np.polynomial.polynomial.polyval(t[held],v)-y[held]),.9))
            residual=float(np.quantile(abs(np.polynomial.polynomial.polyval(t[use],c)-y[use]),.9))
            # Endpoint mismatch is diagnostic, not an unconditional penalty against the stable core.
            score=common_error+.25*hold+(degree-1)*.10*sigma
            bank.append(dict(coef=c,degree=degree,window=window,hold=hold,residual=residual,score=score,
                 common_error=common_error,endpoint_mismatch=float(c[0]-origin)))
    if not bank:return None
    # A common reference interval makes scores comparable. Deterministic quantized ties favor a longer span.
    selected=sorted(bank,key=lambda m:(round(m['score'],9),-m['window'],m['degree']))
    best=selected[0]['score'];selected=[m for m in selected if m['score']<=best+.5*sigma][:3]
    return dict(x=edge,y=origin,direction=1 if right else -1,width=float(width),models=selected,
        anchor_error=float(np.median([m['common_error']+.25*m['hold'] for m in selected])),
        stable=bool(len(selected)>=2 and max(m['hold'] for m in selected)<=2*sigma),sigma=sigma,long_core=True,
        scoring='common_core_error_plus_quarter_holdout',common_reference=[float(t[common].min()),float(t[common].max())])

@memoize('proof')
def head_proof(path,right,width,cfg):
    t,y,w,eligible,edge,origin=canonical(path,right,width);sigma=max(.35,2*width/1046.)
    endpoint=eligible&(t>=-.005)
    if endpoint.sum()<3:endpoint=eligible&(t>=-.015)
    endpoint_q=float(np.mean(w[endpoint])) if endpoint.any() else 1.
    if endpoint_q>=.5:return None,dict(reason='endpoint_confident_preserved',endpoint_q=endpoint_q)
    bank=[]
    for window in (.28,.40,.60):
        use=eligible&(t>=-window-1e-10)&(t<=-.08)
        if use.sum()<24 or np.ptp(t[use])<.16:continue
        split=-.08-.2*(window-.08);train=use&(t<=split);held=use&~train
        if train.sum()<12 or held.sum()<3:continue
        for degree in (1,2):
            c=fit(t[use],y[use],w[use],degree);v=fit(t[train],y[train],w[train],degree)
            error=float(np.quantile(abs(np.polynomial.polynomial.polyval(t[held],v)-y[held]),.9));res=float(np.quantile(abs(np.polynomial.polynomial.polyval(t[use],c)-y[use]),.9))
            if error<=2*sigma and res<=2*sigma:
                bank.append(dict(coef=c,window=window,degree=degree,error=error,residual=res))
    choices=[]
    from itertools import combinations
    for a,b in combinations(bank,2):
        if a['window']==b['window']:continue
        probe=np.asarray([-.08,-.04,0.]);pa=np.polynomial.polynomial.polyval(probe,a['coef']);pb=np.polynomial.polynomial.polyval(probe,b['coef'])
        spread=float(np.max(abs(pa-pb)))
        if spread<=2.5*sigma:choices.append((round(a['error']+b['error']+a['residual']+b['residual']+.25*spread,9),a,b))
    if not choices:return None,dict(reason='no_two_validated_long_windows',endpoint_q=endpoint_q)
    _,a,b=min(choices,key=lambda q:(q[0],-max(q[1]['window'],q[2]['window']),q[1]['degree']+q[2]['degree']))
    tip=eligible&(t>=-.12);tt=t[tip];expected=np.median([np.polynomial.polynomial.polyval(tt,m['coef']) for m in (a,b)],axis=0)
    error=abs(y[tip]-expected);threshold=cfg['endpoint_z']*sigma
    endpoint_error=float(abs(origin-np.median([m['coef'][0] for m in (a,b)])))
    if endpoint_error<=threshold or np.mean(error>threshold)<.25:return None,dict(reason='no_persistent_bad_head',endpoint_q=endpoint_q,endpoint_error=endpoint_error)
    bad=abs(tt[error>threshold]);retreat=min(.12,float(bad.max(initial=0))+.015)
    keep=t<=-retreat+1e-10
    if keep.sum()<16:return None,dict(reason='no_core_after_retreat')
    # Return an x boundary, not new imputed anchor points.
    direction=1 if right else -1;new_edge=float(edge-direction*retreat*width)
    return new_edge,dict(reason='weak_endpoint_conflicts_with_two_long_cores',endpoint_q=endpoint_q,
        endpoint_error=endpoint_error,retreat_fraction=retreat,core_windows=[a['window'],b['window']],core_degrees=[a['degree'],b['degree']],threshold=threshold)

def complete(path,domain,width,cfg,penalty=True,head=True):
    q,audit=frozen_completion(path,domain,width,cfg)
    if q is None:return q,audit
    p=deepcopy(path);x,y=np.asarray(p['x']),np.asarray(p['y']);target=q['x'];decisions=[]
    for right in (False,True):
        original_edge=float(x[-1 if right else 0]);edge=original_edge;proof=dict(reason='head_repair_disabled')
        if head:
            proposed,proof=head_proof(p,right,width,cfg)
            if proposed is not None:edge=proposed
        keep=x<=edge if right else x>=edge
        core=crop(p,keep);edge=float(core['x'][-1 if right else 0])
        rebuilt=abs(edge-original_edge)>1e-8
        anchor=(long_models if penalty else old_models)(core,right,width,cfg)
        if anchor is None:decisions.append(dict(direction=1 if right else -1,head=proof,reason='no_anchor'));continue
        edit=(target>edge) if right else (target<edge)
        if not edit.any():decisions.append(dict(direction=1 if right else -1,head=proof,reason='no_missing_end'));continue
        q['y'][edit],q['uncertainty'][edit]=forecast(anchor,target[edit],cfg)
        prior=(target>=x[0])&(target<=x[-1]);tip=edit&prior
        q['kind'][edit&~prior]=5
        if rebuilt:q['kind'][tip]=6;q['observed'][tip]=False;q['qualified_source'][tip]=False
        q['uncertainty'][edit]=np.maximum(q['uncertainty'][edit],abs(anchor.get('models',[{}])[0].get('endpoint_mismatch',0.)))
        event=dict(type='R72_endpoint',direction=1 if right else -1,interval=[float(target[edit].min()),float(target[edit].max())],
           edit_existing=rebuilt,original_edge=original_edge,anchor_x=edge,anchor_y=anchor['y'],head=proof,
           windows=[m['window'] for m in anchor['models']],degrees=[m['degree'] for m in anchor['models']],
           scoring=anchor.get('scoring','R71_local_score'),long_core=anchor.get('long_core',False),
           stable=anchor['stable'],endpoint_mismatches=[m.get('endpoint_mismatch') for m in anchor['models']])
        decisions.append(event);q['events'].append(event)
    # Core preservation is exact; only registered head windows may replace existing coordinates.
    changed=abs(np.interp(x,q['x'],q['y'])-y)>1e-8
    allowed=np.zeros(len(x),bool)
    for event in decisions:
        if event.get('edit_existing'):allowed|=(x>=event['anchor_x']) if event['direction']==1 else (x<=event['anchor_x'])
    assert not np.any(changed&~allowed),'stable core modified outside registered head window'
    return q,dict(reason='completed',endpoint=decisions,original_changed_columns=int(changed.sum()),stable_core_preserved=True)
