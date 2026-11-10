"""S/Q/F isolated mechanisms. No label access; existing score terms stay fixed."""
from copy import deepcopy
import math
import numpy as np
from scripts.round63_measure import local_slope
from scripts.round63_paths import choose_pair

def region_density(pair,segments,bounds,window):
    lo,hi=window;left,right=bounds
    if hi<=lo:return None
    samples=[];ids=set()
    for s in segments:
        xy=np.linspace(s['p0'],s['p1'],31);x,y=xy.T;m=(x>=lo)&(x<=hi)
        if m.any():ids.add(s['label_id']);samples.append((xy,m,np.linalg.norm(np.subtract(s['p1'],s['p0']))/31))
    if len(ids)<3:return None
    values=[];weights=[]
    for b,rw in enumerate((.25,.5,.25)):
        a=max(lo,left+(right-left)*b/3);z=min(hi,left+(right-left)*(b+1)/3)
        if z-a<2:continue
        mass=0.;observed=False
        for xy,m,step in samples:
            x,y=xy.T;use=m&(x>=a)&(x<=z)
            if use.any():observed=True
            within=use&(y>=np.interp(x,pair['upper']['x'],pair['upper']['y']))&(y<=np.interp(x,pair['lower']['x'],pair['lower']['y']))
            mass+=within.sum()*step
        if not observed:continue
        x=np.linspace(a,z,max(3,int(z-a)));thickness=np.interp(x,pair['lower']['x'],pair['lower']['y'])-np.interp(x,pair['upper']['x'],pair['upper']['y'])
        area=np.trapezoid(thickness,x)
        if area>0:values.append(mass/area);weights.append(rw)
    return float(np.average(values,weights=weights)) if values else None

def rank_density(pairs,segments,bounds,cfg):
    """Only replaces +.2*(C-.5) by +.2*(C*D-.5). No other score edits."""
    if not pairs:return None,[]
    representatives=[]
    width=bounds[1]-bounds[0]
    for p in pairs:
        same=False
        for q in representatives:
            lo=max(p['interval'][0],q['interval'][0]);hi=min(p['interval'][1],q['interval'][1])
            if hi-lo<.12*width:continue
            xx=np.linspace(lo,hi,11)
            error=max(float(np.median(abs(np.interp(xx,p[s]['x'],p[s]['y'])-np.interp(xx,q[s]['x'],q[s]['y'])))) for s in ('upper','lower'))
            if error<=.02*width:same=True;break
        if not same:representatives.append(p)
    audit=[]
    for p in pairs:
        comparisons=[]
        if p['c1_active']:
            for q in representatives:
                lo=max(p['interval'][0],q['interval'][0]);hi=min(p['interval'][1],q['interval'][1])
                if hi-lo<cfg['minimum_common']*width:continue
                dp=region_density(p,segments,bounds,(lo,hi));dq=region_density(q,segments,bounds,(lo,hi))
                if dp is None or dq is None or max(dp,dq)<=1e-12:continue
                if dq<=dp:factor=1.
                elif dq/max(dp,1e-12)<cfg['contrast_guard']:factor=1.
                elif cfg['normalizer']=='ratio':factor=dp/dq
                else:factor=max(0.,(dp/dq-.5)/.5)
                comparisons.append(factor)
        D=min(comparisons) if comparisons else 1.
        if cfg.get('identity_D',False):D=1.
        score=p['score']+.20*p['c1']*(D-1) if p['c1_active'] else p['score']
        audit.append(dict(original=p,new_score=float(score),D=float(D),unknown=not bool(comparisons),comparisons=len(comparisons)))
    audit.sort(key=lambda a:(-a['new_score'],-np.ptp(a['original']['upper']['x'])-np.ptp(a['original']['lower']['x']),a['original']['upper']['ids'],a['original']['lower']['ids']))
    return deepcopy(audit[0]['original']),audit

def exact_adapter(pair,legacy,width,height):
    if pair is None:return None,dict(reason='no_pair')
    sides={};support={};n={}
    for role,side in [('sup','upper'),('deep','lower')]:
        p=pair[side];x,y=np.asarray(p['x']),np.asarray(p['y']);qualified=np.asarray(p['qualified'],bool)
        if len(x)<8 or np.any(np.diff(x)<=0):return None,dict(reason='invalid_path')
        support[role]=x[qualified];n[role]=dict(raw=int(np.asarray(p['observed']).sum()),qualified=int(qualified.sum()),
                                               qualified_derived=int((qualified&~np.asarray(p['observed'])).sum()))
        # Keep the existing formal ray direction estimator; anchor to the same displayed path.
        rays={}
        for key,index in [('left_ray',0),('right_ray',-1)]:
            slope=legacy[role][key][0] if legacy else local_slope(p,x[index],width)
            rays[key]=[slope,float(y[index]-slope*x[index])]
        sides[role]=dict(knots_x=x.tolist(),knots_y=y.tolist(),knots_u=((x-x[0])/(x[-1]-x[0])).tolist(),
            support=[float(x[0]),float(x[-1])],paired_x=support[role].tolist(),max_sample_gap=max(2,.015*width),
            image_shape=[height,width],_round75_exact=True,_round75_width=width,_round75_np_x=x,_round75_np_y=y,**rays)
    joint=np.intersect1d(support['sup'],support['deep'])
    if len(joint)<8:return None,dict(reason='insufficient_qualified_joint_support',counts=n)
    interval=legacy['support'] if legacy else [max(s['support'][0] for s in sides.values()),min(s['support'][1] for s in sides.values())]
    xx=np.linspace(*interval,101)
    if np.any(np.interp(xx,sides['sup']['knots_x'],sides['sup']['knots_y'])>=np.interp(xx,sides['deep']['knots_x'],sides['deep']['knots_y'])):
        return None,dict(reason='order_failure')
    return dict(**sides,support=interval,paired_x=joint.tolist(),max_sample_gap=max(2,.015*width)),dict(reason='exact_displayed_geometry',counts=n,
          MT_positions=np.linspace(*interval,5)[1:-1],legacy_positions_retained=legacy is not None,source_flags_unchanged=True)

_INSTALLED=False
def install_exact_geometry():
    """Process-local opt-in adapter. Old four-knot fits take the unchanged functions."""
    global _INSTALLED
    if _INSTALLED:return
    import sys,types
    from scripts import round29_core as r29,round27_geometry as r27
    oldline=r29.line_at;oldvalue=r29.value_at;oldroots=r29.roots_on_curve
    def value_at(fit,x):
        if not fit.get('_round75_exact'):return oldvalue(fit,x)
        xs,ys=fit['_round75_np_x'],fit['_round75_np_y']
        if x<xs[0]:m,b=fit['left_ray'];return float(m*x+b)
        if x>xs[-1]:m,b=fit['right_ray'];return float(m*x+b)
        return float(np.interp(x,xs,ys))
    def line_at(fit,x):
        if not fit.get('_round75_exact'):return oldline(fit,x)
        xs,ys=fit['knots_x'],fit['knots_y']
        if x<xs[0]:return fit['left_ray']
        if x>xs[-1]:return fit['right_ray']
        p=dict(x=fit['_round75_np_x'],y=fit['_round75_np_y']);m=local_slope(p,x,fit['_round75_width']);y=float(np.interp(x,p['x'],p['y']))
        return [m,y-m*x]
    def roots(seg,fit):
        if not fit.get('_round75_exact'):return oldroots(seg,fit)
        x,y=np.asarray(fit['knots_x']),np.asarray(fit['knots_y']);m=np.diff(y)/np.diff(x);b=y[:-1]-m*x[:-1]
        th=math.radians(seg['direction_deg']);dx,dy=math.cos(th),math.sin(th);cx,cy=seg['centroid']
        den=dy-m*dx;rhs=m*cx+b-cy;parallel=abs(den)<1e-12;overlap=np.flatnonzero(parallel&(abs(rhs)<1e-8)).tolist()
        t=np.divide(rhs,den,out=np.full(len(m),np.nan),where=~parallel);rx=cx+t*dx
        indices=np.flatnonzero((rx>=x[:-1]-1e-8)&(rx<=x[1:]+1e-8)&~parallel)
        out=[]
        for i in indices:
            if any(abs(t[i]-q['t'])<=1e-8 for q in out):continue
            yy=cy+t[i]*dy;slope=line_at(fit,float(rx[i]))[0]
            out.append(dict(t=float(t[i]),point=[float(rx[i]),float(yy)],piece=int(i),line=[slope,float(yy-slope*rx[i])]))
        return dict(roots=sorted(out,key=lambda q:q['t']),coincident_pieces=overlap)
    for name,module in list(sys.modules.items()):
        if not name.startswith('scripts.') or module is None:continue
        for key,value in list(vars(module).items()):
            if value is oldline:setattr(module,key,line_at)
            elif value is oldvalue:setattr(module,key,value_at)
            elif value is oldroots:setattr(module,key,roots)
    _INSTALLED=True

def reweight_F(out,cfg,cal,scale,tau):
    from scripts.round39_core import fl_inputs,combine_fl
    q=deepcopy(out);ids,vals,quality,kernel,active,span=fl_inputs(q,cfg,True)
    risks=q.get('boundary_sensitivity',{}).get('tracks',{})
    if not ids or not risks:return q,dict(reason='no_FL_contributors')
    relative=[];old=[]
    for i,v in zip(ids,vals):
        d=risks[str(i)];a=d.get('min_probe_FL');b=d.get('max_probe_FL')
        u=max(abs((a if a is not None else v)-v),abs((b if b is not None else v)-v))/max(abs(v),1.)
        relative.append(u);old.append(d['risk_factor'])
    factor=1/(1+(np.asarray(relative)/tau)**2)
    weights,regions=combine_fl(vals,quality*np.asarray(old)*factor,kernel,active,span,cfg['aggregate'],cal,True,True)
    if weights.sum()<=0:return q,dict(reason='unchanged_zero_evidence')
    q['values']['FL']=float(weights@vals)/scale;q['FL_weights']={str(i):float(w) for i,w in zip(ids,weights)}
    q['selection']['value_px']=float(weights@vals)
    return q,dict(reason='relative_sensitivity_discount',tau=tau,
       tracks={str(i):dict(relative_change=float(u),factor=float(f),weight=float(w)) for i,u,f,w in zip(ids,relative,factor,weights)},
       no_observation_deleted=True,regions=regions)
