"""Explicit edit permission, common validation tasks and non-winning failed quadratics."""
from copy import deepcopy
from itertools import combinations
import hashlib
import numpy as np
from scripts.round66_geometry import fit
from scripts.round71_geometry import forecast
from scripts.round72_geometry import crop
from scripts.round73_geometry import complete as old_complete,baseline

_cache={}

def prepared(source,right,width,cfg):
    x=np.asarray(source['x']);y=np.asarray(source['y']);q=np.asarray(source.get('confidence',np.full(len(x),.9)))
    eligible=np.asarray(source.get('anchor_eligible',source.get('observed',np.ones(len(x),bool))),bool)
    kind=np.asarray(source.get('kind',np.zeros(len(x),int)))
    d=1 if right else -1;edge=float(x[-1 if right else 0]);dist=d*(edge-x)/width
    tip=eligible&(dist<=.015);state='unknown' if not tip.any() else 'weak' if q[tip].mean()<.5 else 'strong'
    quarantine=(dist<cfg['quarantine_fraction']-1e-10) if state=='weak' else np.zeros(len(x),bool)
    use=eligible&(q>=.5)&~quarantine
    locked=(~eligible)|(kind!=0)
    proof=dict(state=state,old_edge=edge,direction=d,quarantine_indices=np.flatnonzero(quarantine).tolist(),
        fit_indices=np.flatnonzero(use).tolist(),protected_indices=np.flatnonzero(locked).tolist(),
        fit_evidence_is_not_edit_permission=True)
    return crop(source,use) if use.sum()>=24 else None,proof

def predict(coef,edge,value,direction,width,x,cfg):
    anchor=dict(x=float(edge),y=float(value),direction=direction,width=float(width),models=[dict(coef=coef)],sigma=0.,anchor_error=0.)
    return forecast(anchor,np.asarray(x),cfg)[0]

def bank(clean,right,width,scale,lookahead,cfg):
    h=hashlib.sha256()
    for k in ('x','y','confidence'):h.update(np.asarray(clean[k]).tobytes())
    key=(h.digest(),right,width,scale,lookahead,cfg['decay_width'])
    if key in _cache:return deepcopy(_cache[key])
    x,y,q=[np.asarray(clean[k]) for k in ('x','y','confidence')];d=1 if right else -1
    edge=float(x[-1 if right else 0]);value=float(y[-1 if right else 0]);t=d*(x-edge)/width;o=np.argsort(t)
    t,y,q=t[o],y[o],q[o];tol=2*scale;models=[];seen=set();common=(t>=-.28)&(t<=-.08)
    for window in (.4,.6,.8,1.):
        use=t>=-window-1e-10
        if use.sum()<24 or np.ptp(t[use])<.65*window:continue
        signature=tuple(np.flatnonzero(use))
        if signature in seen:continue
        seen.add(signature);tt,yy,qq=t[use],y[use],q[use]
        for degree in (1,2):
            c=fit(tt,yy,qq,degree);res=float(np.quantile(abs(np.polynomial.polynomial.polyval(tt,c)-yy),.9))
            common_res=float(np.quantile(abs(np.polynomial.polynomial.polyval(t[common],c)-y[common]),.9)) if common.sum()>=8 else 1e6
            errors=[];own=[];target_counts=[];target_hashes=[]
            for gap in (.04,.08):
                train=use&(t<=-gap);held=(t>-gap)&(t<=0)
                target_counts.append(int(held.sum()));target_hashes.append(hashlib.sha256(np.stack([t[held],y[held]]).tobytes()).hexdigest())
                if train.sum()<12 or held.sum()<3:errors.append(1e6);continue
                at=t[train][-1];a=fit(t[train]-at,y[train],q[train],degree)
                yp=predict(a,0.,y[train][-1],1,width,(t[held]-at)*width,cfg)
                errors.append(float(np.quantile(abs(yp-y[held]),.9)))
            for fraction in (.1,.2):
                train=tt<=tt.max()-fraction*np.ptp(tt);held=~train
                at=tt[train][-1];a=fit(tt[train]-at,yy[train],qq[train],degree)
                yp=predict(a,0.,yy[train][-1],1,width,(tt[held]-at)*width,cfg)
                own.append(float(np.quantile(abs(yp-yy[held]),.9)))
            models.append(dict(coef=c,degree=degree,window=window,actual_span=float(np.ptp(tt)),n=int(use.sum()),
                cv=max(errors),cv_errors=errors,own_cv=max(own),common_res=common_res,residual=res,
                common_target_counts=target_counts,common_target_hashes=target_hashes,
                old_score=max(own)+.25*res+.05*tol*(degree-1),new_score=max(errors)+.25*common_res+.05*tol*(degree-1)))
    linear=[m for m in models if m['degree']==1]
    global_slope=float(np.median([m['coef'][1]/width for m in linear])) if linear else 0.
    near=t>=-.12;local_slopes=[]
    for win in (.08,.12):
        use=t>=-win
        if use.sum()>=8:local_slopes.append(float(fit(t[use],y[use],q[use],1)[1]/width))
    turn=bool(near.sum()>=8 and np.ptp(y[near])>=tol and len(local_slopes)==2 and all(s*global_slope<0 and abs(s)*.08*width>=tol for s in local_slopes))
    for m in models:
        c=m['coef'];probes=np.array([0.,min(.08,lookahead/width),lookahead/width])
        derivative=(c[1]+2*c[2]*cfg['decay_width']*(1-np.exp(-probes/cfg['decay_width'])))/width
        opposite=bool(abs(global_slope)*.4*width>tol and np.any(derivative*global_slope<0))
        line=next((z for z in linear if z['window']==m['window']),None)
        gain=m['degree']==1 or bool(line and m['cv']<=.75*line['cv'])
        m.update(curve_gain=bool(gain),turn_supported=turn,direction_ok=not opposite or turn,
            admitted=bool(m['cv']<=tol and m['common_res']<=tol and gain and (not opposite or turn)))
    result=dict(x=edge,y=value,direction=d,width=float(width),sigma=scale,tolerance=tol,models=models,
        global_slope=global_slope,turn_supported=turn,common_targets=[.04,.08])
    if len(_cache)>128:_cache.clear()
    _cache[key]=deepcopy(result);return result

def select(data,long_reference,new_score):
    models=[m for m in data['models'] if long_reference or m['window']<=.6+1e-9]
    valid=[m for m in models if m['admitted']]
    if valid:
        if new_score:chosen=min(valid,key=lambda m:(-round(m['actual_span'],10),m['degree'],m['new_score']))
        else:chosen=min(valid,key=lambda m:(m['old_score'],-m['actual_span'],m['degree']))
        return chosen,'common_validation_passed',True
    linear=sorted([m for m in models if m['degree']==1],key=lambda m:-m['actual_span'])
    if len(linear)>=2:
        probe=np.array([.04,.08])*data['width'];d=data['direction'];xx=data['x']+d*probe
        values=[predict(m['coef'],data['x'],data['y'],d,data['width'],xx,dict(decay_width=.1)) for m in linear[:2]]
        if np.max(abs(values[0]-values[1]))<=data['tolerance']:
            return linear[0],'uncertain_long_linear_consensus',False
    return None,'inherit_parent_no_qualified_model_or_linear_consensus',False

def permission(source,parent,clean,proof,data,width,scale,cfg):
    base=dict(allowed=False,reason='no_independent_anomaly',interval=None)
    if clean is None or data is None:return dict(base,reason='no_fit_evidence')
    if proof['state']!='weak':return dict(base,reason='no_raw_weak_tip_evidence')
    x,y=np.asarray(source['x']),np.asarray(source['y']);indices=np.array(proof['quarantine_indices'],int)
    locked=set(proof['protected_indices'])
    if not len(indices) or locked.intersection(indices):return dict(base,reason='protected_or_derived_geometry_in_candidate_window')
    d=proof['direction'];edge=proof['old_edge'];inner=float(clean['x'][-1 if d==1 else 0])
    if abs(inner-edge)>cfg['quarantine_fraction']*width+2*scale:return dict(base,reason='fit_anchor_cannot_expand_edit_window')
    linear=sorted([m for m in data['models'] if m['degree']==1],key=lambda m:-m['actual_span'])
    choices=[]
    for a,b in combinations(linear,2):
        vals=[predict(m['coef'],data['x'],data['y'],d,width,np.array([edge]),cfg)[0] for m in (a,b)]
        if abs(vals[0]-vals[1])<=2*scale:choices.append((-(a['actual_span']+b['actual_span']),a,b))
    if not choices:return dict(base,reason='no_two_long_position_checks')
    _,a,b=min(choices,key=lambda q:q[0]);px=x[indices]
    expected=np.mean([predict(m['coef'],data['x'],data['y'],d,width,px,cfg) for m in (a,b)],axis=0)
    current=np.interp(px,parent['x'],parent['y']);error=abs(current-expected);threshold=cfg['permission_z']*2*scale
    tip_value=np.interp(edge,parent['x'],parent['y']);at_edge=np.mean([predict(m['coef'],data['x'],data['y'],d,width,[edge],cfg)[0] for m in (a,b)])
    tip_t=d*(px-edge)/width;tip_coef=fit(tip_t,current,np.ones(len(px)),1)
    core_slope=float(np.mean([a['coef'][1],b['coef'][1]]));bend=abs(tip_coef[1]-core_slope)*np.ptp(tip_t)
    if abs(tip_value-at_edge)<=threshold or np.mean(error>threshold)<.25 or bend<=threshold:
        return dict(base,reason='weak_probability_without_persistent_geometric_anomaly',edge_discrepancy=float(abs(tip_value-at_edge)),bend=float(bend),threshold=threshold)
    return dict(allowed=True,reason='independent_position_and_direction_anomaly',interval=sorted([edge,inner]),
        threshold=threshold,edge_discrepancy=float(abs(tip_value-at_edge)),persistent_fraction=float(np.mean(error>threshold)),bend=float(bend),
        validated_core_windows=[a['window'],b['window']],replacement_indices=indices.tolist())

def apply(source,parent,proposal,right,anchor,license,width,mode):
    out=deepcopy(parent);x=np.asarray(source['x']);target=np.asarray(parent['x']);d=1 if right else -1;oldedge=float(x[-1 if right else 0])
    stable=float(anchor);outside=(target>oldedge) if right else (target<oldedge)
    edit=(target>stable) if right else (target<stable);existing=(target>=x[0])&(target<=x[-1])
    if np.any(edit&existing) and not license['allowed']:return out,dict(applied=False,reason=license['reason'])
    if license['allowed']:
        lo,hi=license['interval'];permitted=(target>=lo-1e-8)&(target<=hi+1e-8)
        if np.any(edit&existing&~permitted):return out,dict(applied=False,reason='proposal_exceeds_explicit_permission')
        source_edit=(x>stable) if right else (x<stable)
        if not set(np.flatnonzero(source_edit))<=set(license['replacement_indices']):
            return out,dict(applied=False,reason='proposal_covers_unlicensed_existing_indices')
    if abs(np.interp(stable,parent['x'],parent['y'])-np.interp(stable,proposal['x'],proposal['y']))>1e-7:
        return out,dict(applied=False,reason='geometry_join_mismatch')
    for field in ('y','kind','observed','uncertainty','qualified_source'):
        out[field][edit]=proposal[field][edit]
    at=np.searchsorted(target,x);changed=abs(np.asarray(out['y'])[at]-np.interp(x,parent['x'],parent['y']))>1e-8
    eligible=np.asarray(source.get('anchor_eligible',source.get('observed',np.ones(len(x),bool))),bool)
    locked=~eligible|(np.asarray(source.get('kind',np.zeros(len(x),int)))!=0)
    assert not np.any(changed&locked),'protected geometry changed'
    if changed.any():
        assert license['allowed'];assert set(np.flatnonzero(changed))<=set(license['replacement_indices'])
    return out,dict(applied=True,reason=mode,changed=int(changed.sum()),new_columns=int((edit&outside).sum()),
        interval=[float(target[edit].min()),float(target[edit].max())])

def evaluate(source,domain,width,scale,cfg):
    parent,_=baseline(source,domain,width);legacy,la=old_complete(source,domain,width,cfg,'SL')
    if parent is None:return {a:dict(path=None,audit=[]) for a in ('R','U','P','OS','OL','NS','NL')}
    result={a:dict(path=deepcopy(parent),audit=[]) for a in ('R','P','OS','OL','NS','NL')};result['U']=dict(path=legacy,audit=la.get('endpoint',[]))
    sx=np.asarray(source['x']);target=np.asarray(parent['x'])
    for right in (False,True):
        edge=sx[-1 if right else 0];outside=(target>edge) if right else (target<edge)
        if not outside.any():continue
        clean,proof=prepared(source,right,width,cfg)
        data=bank(clean,right,width,scale,abs((domain[1] if right else domain[0])-(clean['x'][-1 if right else 0])),cfg) if clean is not None else None
        permit=permission(source,parent,clean,proof,data,width,scale,cfg)
        event=next((e for e in la.get('endpoint',[]) if e.get('direction')==(1 if right else -1) and 'anchor_x' in e),None)
        if event:
            pp,au=apply(source,result['P']['path'],legacy,right,event['anchor_x'],permit,width,'legacy_scoring_permission_fixed')
            result['P']['path']=pp;result['P']['audit'].append(dict(direction=1 if right else -1,permission=permit,decision=au))
        for arm in ('OS','OL','NS','NL'):
            if data is None:result[arm]['audit'].append(dict(direction=1 if right else -1,decision=dict(applied=False,reason='no_fit_evidence'),permission=permit));continue
            m,reason,valid=select(data,arm.endswith('L'),arm.startswith('N'))
            if m is None:result[arm]['audit'].append(dict(direction=data['direction'],decision=dict(applied=False,reason=reason),permission=permit));continue
            proposal=deepcopy(parent);at=data['x'];edit=(target>at) if right else (target<at)
            proposal['y'][edit]=predict(m['coef'],at,data['y'],data['direction'],width,target[edit],cfg)
            proposal['kind'][edit]=5;inside=(target>=sx[0])&(target<=sx[-1]);proposal['kind'][edit&inside]=6
            proposal['observed'][edit]=False;proposal['qualified_source'][edit]=False
            dist=abs(target[edit]-at);proposal['uncertainty'][edit]=2*scale+max(m['cv'],m['common_res'])*(1+dist/max(.08*width,1))+.015*dist*dist/width
            if not valid:proposal['uncertainty'][edit]=np.maximum(proposal['uncertainty'][edit],100*scale)
            out,au=apply(source,result[arm]['path'],proposal,right,at,permit,width,reason);result[arm]['path']=out
            result[arm]['audit'].append(dict(direction=data['direction'],permission=permit,decision=au,model_admitted=valid,
                model={k:v for k,v in m.items() if k!='coef'},stable_anchor=at,fit_indices=proof['fit_indices'],
                quarantine_indices=proof['quarantine_indices'],fallback_reason=reason,scale=scale,common_targets=data['common_targets']))
    return result
