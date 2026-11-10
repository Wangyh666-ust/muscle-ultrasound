"""Evidence/provenance separation. This module never sees manual boundaries."""
from copy import deepcopy
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scripts.round63_paths import sample,pack_path
from scripts.round68_geometry import bands,sigma,runs

def roughness(x,y,width):
    if len(x)<7:return np.zeros(len(x))
    n=min(len(x)//2*2-1,max(7,int(.08*width)//2*2+1))
    return abs(y-savgol_filter(y,n,2,mode='interp'))

def classify(source,w,field,role,gray,valid,width,cfg):
    """Local W qualification, stricter independent anchor mask, protected alternative.

    Raw extraction flags are provenance, NOT reused as new eligibility flags.
    No interpolation or X point can gain support here (only kind=3 W points).
    """
    if w is None:return None,None,dict(reason='no_pair')
    x,y=w['x'],w['y'];n=len(x);s=sigma(width);raw=w['observed'].copy();derived=(w['kind']==3)&~raw
    zero=np.zeros(n,bool);safe=zero.copy();uncertainty=np.full(n,-1.);bandwidth=np.full(n,-1.)
    gs=sample(gray,x,y);qs=sample(field,x,y);rs=sample(role,x,y)
    reason=np.full(n,'not_W_relocalization',dtype=object);reason[raw]='raw_prediction'
    if derived.any():
        levels=[];bb=None
        for t in (.35,.5,.65):
            b,_=bands(field,valid,w,t)
            if b is None:levels=[];break
            levels.append(b['center']);bb=b if t==.5 else bb
        if levels:
            ix=np.clip(np.rint(x).astype(int),0,field.shape[1]-1)
            cs=np.asarray(levels)[:,ix];finite=np.isfinite(cs).all(0);v=np.nan_to_num(cs)
            uncertainty=np.where(finite,np.ptp(v,axis=0),-1.)
            bandwidth=np.nan_to_num(bb['width'][ix],nan=-1.)
            peak=np.zeros(n)
            # Same connected band only. Parallel components are never averaged.
            for k in np.flatnonzero(derived&finite):
                lo,hi=int(bb['top'][ix[k]]),int(bb['bottom'][ix[k]])
                peak[k]=gray[max(0,lo):min(gray.shape[0],hi+1),ix[k]].max(initial=0)
            checks=[('outside_evidence',sample(valid.astype(float),x,y)>.99),
                    ('weak_probability',(qs>=cfg['q_min'])&(rs>=cfg['q_min'])),
                    ('ambiguous_or_branched_band',finite&(uncertainty<=2*s)&(bandwidth<=.06*width)),
                    ('off_stable_band_center',abs(y-v[1])<=cfg['center_tolerance_sigma']*s),
                    ('no_local_bright_ridge',(gs>=cfg['gray_min'])&(gs>=cfg['gray_ridge_fraction']*peak)),
                    ('local_shape_inconsistent',roughness(x,y,width)<=cfg['local_residual_sigma']*s)]
            safe=derived.copy();reason[derived]='qualified_W_local_probability_and_ridge'
            for label,ok in checks:
                reject=safe&~ok;reason[reject]=label;safe &= ok
        else:reason[derived]='missing_cross_threshold_seeded_band'
    # Fit and local support can coincide numerically but remain separate fields.
    fit=raw|safe;support=raw|safe
    anchor=raw|(safe&(uncertainty<=s)&(roughness(x,y,width)<=s))
    eligibility=dict(fit_eligible=fit,support_eligible=support,anchor_eligible=anchor,
        source_type=np.where(raw,'raw_prediction',np.where(w['kind']==3,'W_probability_relocalization',
                           np.where(w['kind']==1,'J_reconstruction','X_extension'))).tolist(),
        reason=reason.tolist(),uncertainty_px=uncertainty,band_width_px=bandwidth,
        gray=gs,fused_probability=qs,role_probability=rs,geometry_version='R68_W_frozen')
    # P is a separate, label-free geometric intervention. One persistent event is
    # accepted only with improved image ridge OR less inconsistent source geometry.
    py=np.interp(x,source['x'],source['y']);source_exact=np.isin(x,source['x'])
    psobs=np.zeros(n,bool);pskind=np.zeros(n,int)
    where=np.searchsorted(x,source['x']);exact=(where<n)&(x[np.minimum(where,n-1)]==source['x'])
    psobs[where[exact]]=source['observed'][exact];pskind[where[exact]]=source['kind'][exact]
    oldgray=sample(gray,x,py);ro=roughness(x,py,width);rn=roughness(x,y,width)
    allow=zero.copy();events=[]
    for event in w['events']:
        if event.get('type')!='R68_W':continue
        a,b=event['interval'];window=(x>=a)&(x<=b)&derived;known=window&source_exact
        if not known.any():continue  # New endpoint alone cannot justify changing the original.
        gain=float(np.mean(gs[known]-oldgray[known]));rg=float(np.mean(ro[known]-rn[known]))
        precision=float(safe[known].mean());image_better=gain>=cfg['gray_gain']
        shape_better=rg>=cfg['roughness_gain_sigma']*s and gain>=-.02
        accepted=precision>=.85 and (image_better or shape_better)
        if accepted:allow|=window
        events.append(dict(interval=[a,b],accepted=bool(accepted),qualified_fraction=precision,
                           mean_gray_gain=gain,mean_roughness_gain_px=rg,reason='ridge_or_shape_evidence' if accepted else 'stable_original_or_insufficient_independent_evidence'))
    # No implicit interpolation outside the S extent when an extension is refused.
    keep=source_exact|allow;px=x[keep];py=np.where(allow,y,py)[keep]
    po=np.where(allow,raw,psobs)[keep];pk=np.where(allow,w['kind'],pskind)[keep]
    protected=pack_path(px,py,sample(field,px,py),po,source['ids'],kind=pk,
                        events=list(source['events'])+[dict(type='R69_P',**e) for e in events])
    pe={k:(v[keep] if isinstance(v,np.ndarray) else [v[i] for i in np.flatnonzero(keep)] if isinstance(v,list) and len(v)==n else v) for k,v in eligibility.items()}
    refused=~allow[keep]
    for k in ('fit_eligible','support_eligible','anchor_eligible'):pe[k][refused]=po[refused]
    pe['source_type']=['raw_prediction' if po[i] else pe['source_type'][i] for i in range(len(px))]
    pe['geometry_version']='R69_P_alternative'
    return eligibility,(protected,pe),dict(reason='classified',W_columns=int(derived.sum()),qualified_W_columns=int(safe.sum()),
           P_events=events,P_retained_columns=int(allow.sum()),raw_flags_unmodified=bool(np.array_equal(w['observed'],raw)))

def adapt(pair,eligibility,width,height,mode='both'):
    """Frozen estimator and MT positions, with independent fit/support masks."""
    from scripts.round56_boundary import fit_side
    if pair is None:return None,dict(reason='no_pair')
    sides={};support={};audit={}
    for side,new in [('sup','upper'),('deep','lower')]:
        p=pair[new];e=eligibility[new];fit=np.asarray(e['fit_eligible'],bool)
        mask=np.asarray(e['support_eligible'],bool) if mode=='both' else p['observed']
        x,y=p['x'],p['y'];audit[new]=dict(fit_x=x[fit],fit_y=y[fit],support_x=x[mask],
             fit_derived=int((fit&~p['observed']).sum()),support_derived=int((mask&~p['observed']).sum()),
             raw_observed=int(p['observed'].sum()),raw_flags_changed=False)
        if fit.sum()<8:return None,dict(reason='insufficient_qualified_fit_points',side=new,points=audit)
        fitted=fit_side(np.column_stack([x[fit],y[fit]]),width,height)
        if fitted is None:return None,dict(reason='frozen_four_piece_fit_failed',side=new,points=audit)
        sides[side]=fitted;support[side]=x[mask]
    lo=max(sides[s]['support'][0] for s in sides);hi=min(sides[s]['support'][1] for s in sides)
    xs=np.intersect1d(support['sup'],support['deep'])
    if hi<=lo or len(xs)<8:return None,dict(reason='insufficient_shared_qualified_support',points=audit)
    check=np.linspace(lo,hi,101)
    if np.any(np.interp(check,sides['sup']['knots_x'],sides['sup']['knots_y'])>=np.interp(check,sides['deep']['knots_x'],sides['deep']['knots_y'])):
        return None,dict(reason='fitted_order_failure',points=audit)
    curve=dict(**sides,support=[lo,hi],paired_x=xs.tolist(),max_sample_gap=max(2,.015*width))
    positions=lo+np.asarray([.25,.5,.75])*(hi-lo)
    gap=np.min(abs(positions[:,None]-xs),axis=1)
    return dict(pair_id=0,curve=curve),dict(reason='adapted',mode=mode,points=audit,MT_positions=positions,
          nearest_joint_support_distance=gap,max_sample_gap=curve['max_sample_gap'],
          measurement_extent='local_qualified_support_not_bounding_box',four_piece_estimator_changed=False)
