"""No reference labels enter these optional mechanisms."""
from copy import deepcopy
from collections import Counter
import math
import numpy as np
from scripts.round63_paths import pack_path, bridge, sample, path_quality, containment
from scripts.round64_seams import crop_path
from scripts.round77_layer import rank as prior_rank, sortkey, tail


def stable(path,right,width,window):
    x=np.asarray(path['x']);y=np.asarray(path['y']);at=-1 if right else 0
    use=(abs(x-x[at])<=window*width)&np.asarray(path['observed'],bool)
    xx,yy=x[use],y[use]
    if len(xx)<8 or np.ptp(xx)<.08*width:return None
    order=np.argsort(abs(xx-x[at]));hold=order[:max(3,len(xx)//5)];train=order[max(3,len(xx)//5):]
    if len(train)<5:return None
    center=x[at];A=np.c_[xx-center,np.ones(len(xx))]
    def fit(indices):
        a=A[indices];v=yy[indices];coef=np.linalg.lstsq(a,v,rcond=None)[0]
        for _ in range(3):
            e=v-a@coef;mad=max(width/1046.,1.4826*np.median(abs(e-np.median(e))))
            w=np.sqrt(np.minimum(1.,1.5*mad/np.maximum(abs(e),1e-8)));coef=np.linalg.lstsq(a*w[:,None],v*w,rcond=None)[0]
        return coef
    test=fit(train);error=float(np.quantile(abs(A[hold]@test-yy[hold]),.9));coef=fit(np.arange(len(xx)))
    if error>2*width/512:return None
    return dict(slope=float(coef[0]),holdout_px=error,span=float(np.ptp(xx)))


def connect(paths,field,gray,valid,width,base,cfg,fixed,existing=()):
    """Long stable context selects local retreats; original candidates remain untouched."""
    pool=list(paths)+list(existing);seen={tuple(p['ids']) for p in pool};new=[];rejected=Counter();attempts=0
    for a in pool:
        for b in pool:
            union=tuple(sorted(set(a['ids'])|set(b['ids'])))
            if set(a['ids'])&set(b['ids']) or len(union)>base['max_chain'] or union in seen:continue
            if b['x'][0]<=a['x'][0] or b['x'][-1]<=a['x'][-1]+.005*width:continue
            if b['x'][0]-a['x'][-1]>base['max_gap']*width:continue
            # Stable parallel ridges coexist over a long interval: do not average them.
            lo,hi=b['x'][0],a['x'][-1]
            if hi-lo>=.08*width:
                xx=np.linspace(lo,hi,31);delta=np.interp(xx,a['x'],a['y'])-np.interp(xx,b['x'],b['y'])
                if abs(np.median(delta))>2*width/512 and np.quantile(abs(delta-np.median(delta)),.9)<=width/512:
                    rejected['persistent_parallel_distinct_ridges']+=1;continue
            proposals=[]
            for lt in fixed['trim_fractions']:
                for rt in fixed['trim_fractions']:
                    ma=a['x']<=a['x'][-1]-lt*width;mb=b['x']>=b['x'][0]+rt*width
                    if ma.sum()<8 or mb.sum()<8:continue
                    aa,bb=crop_path(a,ma),crop_path(b,mb)
                    gap=bb['x'][0]-aa['x'][-1]
                    if gap<max(4.,.02*width) or gap>base['max_gap']*width:continue
                    sa,sb=stable(aa,True,width,cfg['endpoint_window']),stable(bb,False,width,cfg['endpoint_window'])
                    if sa is None or sb is None:rejected['no_stable_heldout_context']+=1;continue
                    if abs(math.atan(sa['slope'])-math.atan(sb['slope']))>math.radians(fixed['stable_trend_limit_deg']):rejected['long_context_different_trend']+=1;continue
                    settings={**base,'endpoint_window':cfg['endpoint_window']}
                    p,why=bridge(aa,bb,field,gray,valid,width,settings);attempts+=1
                    if p is None:rejected[why]+=1;continue
                    inside=(p['x']>aa['x'][-1])&(p['x']<bb['x'][0]);xx=p['x'][inside];yy=p['y'][inside]
                    hit=sample(field,xx,yy,max(1,round(width/1046.)))>=fixed['minimum_bridge_probability']
                    cuts=np.flatnonzero(np.diff(np.r_[False,~hit,False]));run=int(np.max(cuts[1::2]-cuts[::2],initial=0))
                    if not len(hit) or hit.mean()<cfg['required_probability_hit'] or run>fixed['maximum_bridge_blind_run']*width:
                        rejected['insufficient_probability_corridor']+=1;continue
                    # Use evidence and held-out prediction, never length alone, to choose a repair window.
                    evidence=float(sample(field,xx,yy).mean());penalty=(sa['holdout_px']+sb['holdout_px'])/max(width/512,1e-8)
                    merit=evidence-.02*penalty-.01*(lt+rt)
                    p['events'].append(dict(type='R80_C',interval=[float(aa['x'][-1]),float(bb['x'][0])],retreat=[lt*width,rt*width],stable_context=[sa,sb],probability_hit=float(hit.mean()),original_observations_promoted=False))
                    proposals.append((merit,p))
            if proposals:
                p=max(proposals,key=lambda z:(z[0],-len(z[1]['x'])))[1];new.append(p);seen.add(union)
    new.sort(key=lambda p:-path_quality(p,width))
    return new[:fixed['max_new_paths']],dict(added=min(len(new),fixed['max_new_paths']),attempts=attempts,rejections=dict(rejected))


def fair_rank(pairs,paths,prob,segments,width,bounds,use_prior=True):
    baseline,original=prior_rank(pairs,paths,prob,segments,width,bounds,None,use_prior)
    if len(pairs)<2:return baseline,dict(reason='single_candidate',original=original)
    # Preserve the original policy for geometric variants of one physical band.
    distinct=[];tol=2*max(prob.shape[1:])/512
    for p in sorted(pairs,key=sortkey):
        equivalent=False
        for q in distinct:
            lo=max(p['interval'][0],q['interval'][0]);hi=min(p['interval'][1],q['interval'][1])
            if hi-lo<.08*width:continue
            xx=np.linspace(lo,hi,31)
            if all(np.quantile(abs(np.interp(xx,p[s]['x'],p[s]['y'])-np.interp(xx,q[s]['x'],q[s]['y'])),.9)<=tol for s in ('upper','lower')):equivalent=True;break
        if not equivalent:distinct.append(p)
    if len(distinct)<2:return baseline,dict(reason='one_physical_band',original=original)
    lo=max(p['interval'][0] for p in distinct);hi=min(p['interval'][1] for p in distinct)
    known=hi-lo>=.12*width;common=[]
    if known:
        for p in pairs:
            u,d=deepcopy(p['upper']),deepcopy(p['lower'])
            for v in (u,d):
                keep=(v['x']>=lo)&(v['x']<=hi)
                for k in ('x','y','observed','confidence','kind'):v[k]=np.asarray(v[k])[keep]
            if len(u['x'])<2 or len(d['x'])<2:known=False;break
            C,active,count=containment(u,d,segments,[lo,hi]);common.append((C,active,count))
        known=known and len(common)==len(pairs) and all(v[1] for v in common)
    if not known:
        result=deepcopy(baseline)
        if result:result['ambiguous']=True
        return result,dict(reason='C1_unknown_keep_baseline',common_window=[lo,hi],original=original,unknown_not_evidence_of_absence=True)
    ranked=[];audit=[]
    for i,p in enumerate(pairs):
        old=.2*(p['c1']-.5) if p['c1_active'] else 0.
        bonus=.2*(common[i][0]-.5) if known else 0.
        q=deepcopy(p);q['score']=p['score']-old+bonus-(tail(p['MT_mm']) if use_prior else 0.)
        ranked.append(q);audit.append(dict(ids=[list(p[s]['ids']) for s in ('upper','lower')],old_bonus=old,new_bonus=bonus,common_C=common[i][0] if known else None,known=known,score=q['score']))
    ranked.sort(key=sortkey);result=ranked[0]
    if not known:result['ambiguous']=True
    return result,dict(reason='common_observed_window' if known else 'C1_unknown_no_differential_bonus',common_window=[lo,hi],candidates=audit,original=original)


def fidelity_curve_v0(pair,legacy,width,height):
    """Replace only qualified finite samples; original external ray equations stay fixed."""
    if pair is None or legacy is None:return legacy,dict(reason='no_legacy_or_pair',changed=False,sides={})
    result=deepcopy(legacy);info={};changed=False
    for role,side in [('sup','upper'),('deep','lower')]:
        old=legacy[role];p=pair[side];lo,hi=old['support'];px=np.asarray(p['x']);py=np.asarray(p['y']);pq=np.asarray(p['qualified'],bool)
        x=np.unique(np.r_[np.asarray(old['knots_x']),px[(px>=lo)&(px<=hi)]])
        y=np.interp(x,old['knots_x'],old['knots_y']);idx=np.searchsorted(px,x);idx=np.clip(idx,0,len(px)-1)
        exact=(abs(px[idx]-x)<1e-7)&pq[idx];y[exact]=py[idx[exact]]
        # No completion or unconstrained interpolation across unknown evidence is introduced.
        moved=abs(y-np.interp(x,old['knots_x'],old['knots_y']))>1e-8
        if not moved.any():info[side]=dict(changed=False,qualified_points=int(exact.sum()));continue
        result[role]={**deepcopy(old), 'knots_x':x.tolist(),'knots_y':y.tolist(),'knots_u':((x-lo)/max(hi-lo,1)).tolist(),
                      '_round75_exact':True,'_round75_width':width,'_round75_np_x':x,'_round75_np_y':y}
        assert result[role]['left_ray']==old['left_ray'] and result[role]['right_ray']==old['right_ray']
        info[side]=dict(changed=True,qualified_points=int(exact.sum()),moved_points=int(moved.sum()),max_change_px=float(abs(y-np.interp(x,old['knots_x'],old['knots_y'])).max()),external_rays_unchanged=True)
        changed=True
    xx=np.unique(np.r_[result['sup']['knots_x'],result['deep']['knots_x']]);xx=xx[(xx>=legacy['support'][0])&(xx<=legacy['support'][1])]
    if np.any(np.interp(xx,result['sup']['knots_x'],result['sup']['knots_y'])>=np.interp(xx,result['deep']['knots_x'],result['deep']['knots_y'])):
        return legacy,dict(reason='qualified_geometry_order_conflict_preserve_legacy',changed=False,sides=info)
    return result,dict(reason='qualified_finite_samples_only',changed=changed,sides=info,external_rays_unchanged=True,qualification_unchanged=True)


def fidelity_curve(pair,legacy,width,height):
    """Qualified brackets cover fractional knots too; no old free knot may survive inside them."""
    if pair is None or legacy is None:return legacy,dict(reason='no_legacy_or_pair',changed=False,sides={})
    from scripts.round78_mt_core import qualified_at
    result=deepcopy(legacy);info={};changed=False
    for role,side in [('sup','upper'),('deep','lower')]:
        old=legacy[role];p=pair[side];lo,hi=old['support'];px=np.asarray(p['x']);py=np.asarray(p['y'])
        step=float(np.median(np.diff(px)));x=np.unique(np.r_[np.asarray(old['knots_x']),px[(px>=lo)&(px<=hi)]])
        known=np.array([qualified_at(p,float(v),step)[0] for v in x])
        y=np.interp(x,old['knots_x'],old['knots_y']);target=np.interp(x,px,py);y[known]=target[known]
        moved=abs(y-np.interp(x,old['knots_x'],old['knots_y']))>1e-8
        if not moved.any():info[side]=dict(changed=False,qualified_points=int(known.sum()));continue
        result[role]={**deepcopy(old),'knots_x':x.tolist(),'knots_y':y.tolist(),'knots_u':((x-lo)/max(hi-lo,1)).tolist(),
                      '_round75_exact':True,'_round75_width':width,'_round75_np_x':x,'_round75_np_y':y}
        info[side]=dict(changed=True,qualified_points=int(known.sum()),qualified_fractional_knots=int(np.sum(known&(abs(x-np.rint(x))>1e-7))),
                        moved_points=int(moved.sum()),max_change_px=float(abs(y-np.interp(x,old['knots_x'],old['knots_y'])).max()),external_rays_unchanged=True)
        changed=True
    xx=np.unique(np.r_[result['sup']['knots_x'],result['deep']['knots_x']]);xx=xx[(xx>=legacy['support'][0])&(xx<=legacy['support'][1])]
    if np.any(np.interp(xx,result['sup']['knots_x'],result['sup']['knots_y'])>=np.interp(xx,result['deep']['knots_x'],result['deep']['knots_y'])):
        return legacy,dict(reason='qualified_geometry_order_conflict_preserve_legacy',changed=False,sides=info)
    return result,dict(reason='qualified_nodes_and_brackets',changed=changed,sides=info,external_rays_unchanged=True,qualification_unchanged=True)
