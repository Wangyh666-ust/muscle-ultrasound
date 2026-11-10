"""Counterfactual local-evidence gate; shape proposals never authorize their own edits."""
from copy import deepcopy
from collections import Counter
import numpy as np
from scripts.round63_paths import pack_path,sample,hermite,refine_bridge
from scripts.round64_seams import crop_path
from scripts.round66_geometry import anchors,extend_from_anchor,join_paths,noise,anchor_at


def runs(mask):
    cuts=np.flatnonzero(np.diff(np.r_[False,np.asarray(mask,bool),False]))
    return list(zip(cuts[::2],cuts[1::2]))


def restore_path(p,field):
    return pack_path(p['x'],p['y'],sample(field,np.asarray(p['x']),np.asarray(p['y'])),
        p['observed'],p['ids'],kind=p['kind'],events=p['events'])


def original_with_confidence(p,field):
    if 'confidence' in p:return p
    return restore_path(p,field)


def compare(source,candidate,field,gray,valid,width,cfg,gain,region=None):
    """Compare both alternatives at the same original coordinates. GT never enters."""
    if candidate is None:return False,dict(reason='no_proposal')
    x,y=source['x'],source['y'];nx,ny=candidate['x'],candidate['y']
    if x[0]<nx[0]-1e-8 or x[-1]>nx[-1]+1e-8:return False,dict(reason='original_extent_lost')
    predicted=np.interp(x,nx,ny);difference=abs(predicted-y)
    event=candidate['events'][-1];lo,hi=event['interval']
    if region is None:
        direction=event.get('direction')
        if direction:
            edge=x[-1] if direction==1 else x[0]
            region=(edge-cfg['edit_fraction']*width,edge) if direction==1 else (edge,edge+cfg['edit_fraction']*width)
        else:region=(lo,hi)
    area=(x>=region[0]-1e-8)&(x<=region[1]+1e-8)
    outside=~area
    if np.max(difference[outside],initial=0)>1e-8:return False,dict(reason='outside_fixed_region_changed')
    sigma=max(.35,2*width/1046.,cfg['minimum_edit_network_pixels']*max(field.shape)/512.)
    edited=area&(difference>sigma)
    if edited.sum()<max(3,cfg['minimum_changed_fraction']*width):return False,dict(reason='no_meaningful_edit')
    q0=sample(field,x,y);q1=sample(field,x,predicted);g0=sample(gray,x,y);g1=sample(gray,x,predicted)
    mid=sample(field,x,(y+predicted)/2)
    same=mid>=cfg['same_band_ratio']*np.minimum(q0,q1)
    details=dict(region=[float(region[0]),float(region[1])],edited_columns=int(edited.sum()),
        q_old=float(q0[edited].mean()),q_new=float(q1[edited].mean()),
        g_old=float(g0[edited].mean()),g_new=float(g1[edited].mean()),same_band=float(same[edited].mean()),
        q_gain=float((q1-q0)[edited].mean()),g_gain=float((g1-g0)[edited].mean()),
        original_region_probability=float(q0[area].mean()),original_region_gray=float(g0[area].mean()))
    # Brightness alone cannot resolve the center of one connected, confident probability band.
    # Assess original support over the fixed region, not a proposal-dependent subset.
    if details['same_band']>=.90 and details['original_region_probability']>=cfg['strong_probability']:
        return False,dict(**details,reason='old_curve_already_supported_in_same_band')
    if details['q_gain'] < -cfg['noninferior_channel'] or details['g_gain'] < -cfg['noninferior_channel']:
        return False,dict(**details,reason='contradictory_local_channel')
    improved=edited&(((q1-q0>=cfg['clear_channel_gain'])&(g1-g0>=-cfg['noninferior_channel']))|
                     ((g1-g0>=cfg['clear_channel_gain'])&(q1-q0>=-cfg['noninferior_channel'])))
    longest=max((b-a for a,b in runs(improved)),default=0)
    details['persistent_gain_columns']=int(longest)
    if longest<max(3,cfg['minimum_run_fraction']*width) or improved.sum()<.60*edited.sum():
        return False,dict(**details,reason='no_persistent_independent_gain')
    supported=((q1>=.1)|(g1>=.6))& (sample(valid.astype(float),x,predicted)>.99)
    if supported[edited].mean()<cfg['required_support_fraction']:
        return False,dict(**details,reason='new_corridor_not_supported')
    score=.5*(q1-q0)+.5*(g1-g0)
    details['common_region_gain']=float(score[area].mean());details['edited_gain']=float(score[edited].mean())
    if details['edited_gain']<gain:return False,dict(**details,reason='insufficient_gain')
    return True,dict(**details,reason='accepted')


def clipped_repair(source,candidate):
    """Do not smuggle extrapolation into the local repair comparison."""
    x,y=source['x'],source['y'];lo,hi=candidate['events'][-1]['interval']
    use=(x>=lo)&(x<=hi);new=np.interp(x,candidate['x'],candidate['y'])
    yy=np.where(use,new,y);obs=source['observed'].copy();kind=source['kind'].copy()
    changed=abs(yy-y)>1e-8;obs[changed]=False;kind[changed]=1
    event={**candidate['events'][-1], 'type':'R67_repair','interval':[float(max(lo,x[0])),float(min(hi,x[-1]))],
        'observations_promoted':False}
    return pack_path(x,yy,source['confidence'],obs,source['ids'],kind=kind,events=source['events']+[event])


def hypotheses(source,field,gray,valid,width,shape_cfg):
    proposals=[];audit=Counter();variant=shape_cfg['variants']['CR']
    for right in (False,True):
        aa=anchors(source,right,width,shape_cfg,variant)
        audit['stable_right_anchors' if right else 'stable_left_anchors']=len(aa)
        for anchor in aa:
            p=extend_from_anchor(source,anchor,field,gray,valid,width,shape_cfg,True)
            if p is not None:proposals.append(p)
    return proposals,dict(audit)


def gap_proposals(source,field,gray,valid,width,shape_cfg,base):
    out=[]
    for a,b in runs(~source['observed']):
        if a==0 or b==len(source['x']):continue
        lo,hi=source['x'][a-1],source['x'][b]
        if hi-lo<max(4,.01*width) or hi-lo>.30*width:continue
        left=crop_path(source,np.arange(len(source['x']))<a);right=crop_path(source,np.arange(len(source['x']))>=b)
        aa=anchor_at(left,True,width,shape_cfg,0,True);bb=anchor_at(right,False,width,shape_cfg,0,True)
        if aa is None or bb is None:continue
        x=source['x'][a:b];y,dy,ddy=hermite(x,lo,aa['y'],aa['slope'],hi,bb['y'],bb['slope'])
        y,_,_,_=refine_bridge(x,y,dy,ddy,field,gray,width,base)
        yy=source['y'].copy();yy[a:b]=y
        event=dict(type='R67_gap',interval=[float(lo),float(hi)],direction=0,
            slopes=[aa['slope'],bb['slope']],shape_windows=[aa['windows'],bb['windows']],observations_promoted=False)
        out.append(pack_path(source['x'],yy,source['confidence'],source['observed'],source['ids'],kind=source['kind'],events=source['events']+[event]))
    return out


def attach_extension(source,proposal,field,gray,valid,width,cfg):
    """Append only. Rejected old-curve edits cannot be adopted through an extension."""
    from scipy.interpolate import PchipInterpolator
    from scripts.round63_measure import local_slope
    event=proposal['events'][-1];direction=event['direction'];edge=source['x'][-1 if direction==1 else 0]
    distance=direction*(proposal['x']-edge);use=(distance>1e-8)&(distance<=cfg['maximum_extension']*width)
    if use.sum()<2:return None,dict(reason='no_finite_extension')
    x=proposal['x'][use];y=proposal['y'][use];d=direction*(x-edge);order=np.argsort(d);x,y,d=x[order],y[order],d[order]
    y0=source['y'][-1 if direction==1 else 0];old_slope=direction*local_slope(source,edge,width)
    oriented=direction*(proposal['x']-edge);order_all=np.argsort(oriented)
    interpolator=PchipInterpolator(oriented[order_all],proposal['y'][order_all])
    raw0=float(interpolator(0));raw_slope=float(interpolator.derivative()(0))
    length=max(2,min(.05*width,d[-1]));t=np.clip(d/length,0,1)
    correction=(y0-raw0)*(2*t**3-3*t**2+1)+(old_slope-raw_slope)*length*(t**3-2*t**2+t)
    if abs(correction).max()>.02*width:return None,dict(reason='reattachment_conflicts_with_source')
    y=y+correction;q=sample(field,x,y,max(1,round(width/1046.)));g=sample(gray,x,y,max(1,round(width/1046.)))
    valid_points=sample(valid.astype(float),x,y)>.99;hit=(q>=.1)|(g>=.6)
    stop=len(x);blind=0
    for i in range(len(x)):
        blind=0 if hit[i] else blind+1
        if not valid_points[i] or blind>max(1,.015*width):stop=max(0,i-blind+1);break
    x,y,q,hit=x[:stop],y[:stop],q[:stop],hit[:stop]
    if len(x)<max(2,.01*width) or hit.mean()<cfg['required_support_fraction']:return None,dict(reason='extension_has_insufficient_evidence')
    newevent=dict(type='R67_extension',interval=[float(x.min()),float(x.max())],direction=direction,
        anchor=[float(edge),float(y0)],observations_promoted=False,local_evidence_only=True,
        initial_position_error=float(abs(raw0+(y0-raw0)-y0)),
        initial_slope_error=float(abs(raw_slope+(old_slope-raw_slope)-old_slope)))
    return pack_path(np.r_[source['x'],x],np.r_[source['y'],y],np.r_[source['confidence'],q],
        np.r_[source['observed'],np.zeros(len(x),bool)],source['ids'],kind=np.r_[source['kind'],np.full(len(x),2)],
        events=source['events']+[newevent]),dict(reason='accepted',support=float(hit.mean()))


def process_path(source,field,gray,valid,width,cfg,shape_cfg,base,gain):
    if source is None:return dict(repaired=None,extended=None,estimates=[],audit=dict(reason='no_original_path'))
    source=original_with_confidence(source,field);proposals,ga=hypotheses(source,field,gray,valid,width,shape_cfg)
    gaps=gap_proposals(source,field,gray,valid,width,shape_cfg,base);audit=Counter();details=[]
    # Each local repair is compared with the unchanged original at fixed coordinates.
    current=source;accepted=[]
    for direction in (-1,1,0):
        pool=[p for p in proposals if p['events'][-1]['direction']==direction] if direction else gaps
        evaluated=[]
        for p in pool:
            r=clipped_repair(source,p);ok,d=compare(source,r,field,gray,valid,width,cfg,gain)
            audit[d['reason']]+=1;details.append(dict(direction=direction,**d))
            if ok:evaluated.append((d['common_region_gain'],r,d))
        if not evaluated:continue
        _,winner,d=max(evaluated,key=lambda z:z[0]);lo,hi=winner['events'][-1]['interval']
        use=(current['x']>=lo)&(current['x']<=hi);yy=current['y'].copy();yy[use]=np.interp(current['x'][use],winner['x'],winner['y'])
        changed=abs(yy-current['y'])>1e-8;obs=current['observed'].copy();kind=current['kind'].copy();obs[changed]=False;kind[changed]=1
        current=pack_path(current['x'],yy,current['confidence'],obs,current['ids'],kind=kind,events=current['events']+[winner['events'][-1]])
        accepted.append(d)
    selected_extensions=[]
    for direction in (-1,1):
        choices=[]
        for p in proposals:
            if p['events'][-1]['direction']!=direction:continue
            q,d=attach_extension(current,p,field,gray,valid,width,cfg);audit['extension_'+d['reason']]+=1
            if q is not None:
                edge=current['x'][-1 if direction==1 else 0]
                xx=edge+direction*np.arange(1,max(2,int(cfg['maximum_extension']*width))+1)
                supported=(xx>=q['x'][0])&(xx<=q['x'][-1]);yy=np.interp(xx,q['x'],q['y'])
                signal=.5*sample(field,xx,yy)+.5*sample(gray,xx,yy)
                score=float(np.mean(np.where(supported,signal,0.)))
                ev=p['events'][-1];tie=(-round(ev['heldout_error'],6),-round(ev['forecast_disagreement'],6),-round(ev['retreat'],6))
                choices.append(((round(score,6),*tie),q))
        if choices:selected_extensions.append(max(choices,key=lambda q:q[0])[1])
    # Both ends are chosen against the same original; processing order has no geometric effect.
    chunks=[(current['x'],current['y'],current['confidence'],current['observed'],current['kind'])];events=list(current['events'])
    for q in selected_extensions:
        event=q['events'][-1];direction=event['direction']
        use=q['x']<current['x'][0]-1e-8 if direction==-1 else q['x']>current['x'][-1]+1e-8
        chunks.append(tuple(q[k][use] for k in ('x','y','confidence','observed','kind')));events.append(event)
    extended=pack_path(*[np.concatenate([c[i] for c in chunks]) for i in range(4)],current['ids'],
        kind=np.concatenate([c[4] for c in chunks]),events=events)
    # Estimates are displayed separately, never sent to the formal measurement adapter.
    estimates=[]
    for direction in (-1,1):
        pool=[p for p in proposals if p['events'][-1]['direction']==direction]
        if pool:
            def merit(p):
                e=p['events'][-1];lo,hi=e['interval'];x=np.linspace(lo,hi,65);y=np.interp(x,p['x'],p['y'])
                return float(np.mean(.5*sample(field,x,y)+.5*sample(gray,x,y))-.03*e['heldout_error']/max(.35,2*width/1046.))
            estimates.append(max(pool,key=merit))
    return dict(repaired=current,extended=extended,estimates=estimates,
        audit=dict(generator=ga,decisions=dict(audit),comparisons=details,accepted=accepted))


def controlled_join(paths,field,gray,valid,width,cfg,shape_cfg,gain):
    ordered=sorted(paths,key=lambda p:p['x'][0])
    source=pack_path(np.concatenate([p['x'] for p in ordered]),np.concatenate([p['y'] for p in ordered]),
        np.concatenate([p['confidence'] for p in ordered]),np.concatenate([p['observed'] for p in ordered]),
        sum([list(p['ids']) for p in ordered],[]))
    pool,_=join_paths(paths,field,gray,valid,width,shape_cfg,shape_cfg['variants']['CR'],inspect_all=True)
    middle=(ordered[0]['x'][-1]+ordered[-1]['x'][0])/2;region=(middle-.20*width,middle+.20*width)
    accepted=[];rows=[]
    for p in pool:
        r=clipped_repair(source,p);ok,d=compare(source,r,field,gray,valid,width,cfg,gain,region)
        rows.append(d)
        if ok:accepted.append((d['common_region_gain'],r))
    return (max(accepted,key=lambda q:q[0])[1] if accepted else source),dict(accepted=bool(accepted),comparisons=rows)
