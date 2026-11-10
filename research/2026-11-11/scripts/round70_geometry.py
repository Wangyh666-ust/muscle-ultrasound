"""W2 follows a seeded band only after a source jump, never just brighter/central."""
from copy import deepcopy
import numpy as np
from scipy.ndimage import median_filter,maximum_filter1d
from scripts.round63_paths import sample,pack_path
from scripts.round68_geometry import sigma,localize,bands,runs

def localize2(source,field,role,gray,valid,width,cfg):
    if source is None:return None,dict(reason='no_source')
    candidate,old=localize(source,field,role,gray,valid,width,dict(min_width_sigma=6.))
    if old.get('reason')!='accepted':candidate=source
    x,y=source['x'],source['y'];s=sigma(width);cx=np.interp(x,candidate['x'],candidate['y']);dx=np.diff(x)
    dy=np.diff(y);dc=np.diff(cx);n=max(3,int(.03*width)//2*2+1)
    expected=median_filter(dy,size=n,mode='reflect');cexpected=median_filter(dc,size=n,mode='reflect')
    abrupt=(abs(dy-expected)>cfg['jump_sigma']*s)&(abs(dc-cexpected)<=1.5*s)&(dx<=max(2,.005*width))
    # A jump needs a real adjacent sample. Long missing columns are not fake jumps.
    flags=np.r_[abrupt,False]|np.r_[False,abrupt];near=maximum_filter1d(flags.astype(int),size=max(3,int(.10*width)//2*2+1),mode='constant')>0
    changed=(candidate['kind']==3)&~candidate['observed'];allow=np.zeros(len(candidate['x']),bool);events=[]
    for event in candidate['events']:
        if event.get('type')!='R68_W':continue
        lo,hi=event['interval'];area=(candidate['x']>=lo)&(candidate['x']<=hi)&changed
        inside=(x>=lo)&(x<=hi);signal=bool(np.any(flags&inside))
        if not signal:
            # The jump can fall at a proposal boundary after threshold blending.
            signal=bool(np.any(flags&(x>=lo-.02*width)&(x<=hi+.02*width)))
        if signal:allow|=area
        events.append(dict(type='R70_W',interval=[lo,hi],accepted=signal,source_jump_count=int((flags&inside).sum()),
            reason='source_jump_with_continuous_seeded_band' if signal else 'smooth_original_preserved'))
    grid=candidate['x'];orig=np.interp(grid,x,y);exact=np.isin(grid,x);keep=exact|allow
    obs=np.zeros(len(grid),bool);kind=np.ones(len(grid),int);ix=np.searchsorted(grid,x);obs[ix]=source['observed'];kind[ix]=source['kind']
    yy=np.where(allow,candidate['y'],orig);obs[allow]=False;kind[allow]=3
    out=pack_path(grid[keep],yy[keep],sample(field,grid[keep],yy[keep]),obs[keep],source['ids'],kind=kind[keep],events=source['events']+events)
    # Trace probability that is already present in this very same seeded band.
    # This is not X: no blind polynomial extension and no brightness-only support.
    maps=[]
    for threshold in (.35,.5,.65):
        b,_=bands(field,valid,source,threshold)
        if b is None:maps=[];break
        maps.append(b)
    additions=[];span_events=[]
    if maps:
        centers=np.asarray([b['center'] for b in maps]);finite=np.isfinite(centers).all(0);cc=np.nan_to_num(centers)
        stable=finite&(np.ptp(cc,axis=0)<=2*s)&(maps[1]['width']<=.06*width)
        # A finite fragment has no old neighbour beyond its endpoint. Do not fade
        # a justified W correction back toward the rejected old peak at that edge.
        oy=out['y'].copy()
        for at in (0,-1):
            if out['kind'][at]!=3:continue
            edge=out['x'][at];end=abs(out['x']-edge)<=max(2,.005*width);ix=np.rint(out['x']).astype(int)
            end&=(out['kind']==3)&stable[ix]
            oy[end]=cc[1,ix[end]]
        out={**out,'y':oy,'confidence':sample(field,out['x'],oy)}
        for direction in (-1,1):
            at=-1 if direction==1 else 0;previous_x=out['x'][at];previous_y=out['y'][at];step=int(round(previous_x))+direction;xs=[];ys=[]
            while 0<=step<field.shape[1] and stable[step]:
                yy=float(cc[1,step]);xx=float(step)
                if not bool(sample(valid.astype(float),[xx],[yy])[0]>.99):break
                if min(float(sample(field,[xx],[yy])[0]),float(sample(role,[xx],[yy])[0]))<.65:break
                if abs(yy-previous_y)>max(2*s,1.5*abs(xx-previous_x)):break
                xs.append(xx);ys.append(yy);previous_x,previous_y=xx,yy;step+=direction
            if len(xs)>=max(3,.01*width):
                additions.append((np.asarray(xs),np.asarray(ys)));span_events.append(dict(type='R70_Wspan',interval=[float(min(xs)),float(max(xs))],direction=direction,columns=len(xs),reason='same_seeded_probability_band_observed',brightness_alone=False))
    if additions:
        x,y,obs,kind=out['x'],out['y'],out['observed'],out['kind']
        for xx,yy in additions:x=np.r_[x,xx];y=np.r_[y,yy];obs=np.r_[obs,np.zeros(len(xx),bool)];kind=np.r_[kind,np.full(len(xx),3)]
        out=pack_path(x,y,sample(field,x,y),obs,source['ids'],kind=kind,events=out['events']+span_events)
    return out,dict(reason='accepted' if allow.any() or additions else 'normal_original_preserved',source_jump_count=int(abrupt.sum()),W_columns=int(allow.sum()),same_band_observed_added=sum(len(x) for x,y in additions),events=events+span_events)

def eligibility(path,field,role,gray,valid,width,allow_J=False):
    x,y=path['x'],path['y'];raw=path['observed'];kind=path['kind'];s=sigma(width)
    q=sample(field,x,y);r=sample(role,x,y);inside=sample(valid.astype(float),x,y)>.99
    wp=(kind==3)&~raw;w=wp&inside&(q>=.5)&(r>=.5)
    # Stable cross-threshold centre checks remain part of W2 generation.
    # J is not promoted merely by its kind: only this accepted edit window.
    Jwindow=np.zeros(len(x),bool)
    if allow_J:
        for e in path['events']:
            if e.get('type')=='R68_J':Jwindow|=(x>e['interval'][0])&(x<e['interval'][1])
    jj=(kind==1)&~raw&Jwindow;jsupport=jj&inside&(q>=.1)&(r>=.18)
    anchor=raw|(w&(q>=.65)&(r>=.65))
    return dict(fit_eligible=raw|w|jj,support_eligible=raw|w|jsupport,anchor_eligible=anchor,
         source_type=np.where(raw,'raw_prediction',np.where(kind==3,'W_relocalization',np.where(kind==1,'J_reconstruction','X_extension'))).tolist(),
         fused_probability=q,role_probability=r,geometry_version='R70_current',
         raw_flags_unchanged=True,J_shape_only_points=int((jj&~jsupport).sum()))

def synchronize(source,current,pool,field,role,gray,valid,width):
    """Update associated old positions; preserve raw alternatives, forbid stale anchors."""
    s=sigma(width);moved=(current['kind']==3)&~current['observed'];out=[];audit=[]
    for p in pool:
        q=deepcopy(p);x=q['x'];inside=(x>=source['x'][0])&(x<=source['x'][-1]);oldy=np.interp(x,source['x'],source['y']);newy=np.interp(x,current['x'],current['y'])
        touched=np.interp(x,current['x'],moved.astype(float))>.5
        associated=inside&touched&(abs(q['y']-oldy)<=2*s)
        shared=bool(set(p['ids'])&set(source['ids']))
        # Association is local AND geometric, not source ID alone.
        if shared or (associated.sum()>=12 and associated.sum()>=.70*max(1,(inside&touched).sum())):
            q['y'][associated]=newy[associated];q['observed'][associated]=False;q['kind'][associated]=3
        e=eligibility(q,field,role,gray,valid,width);q['anchor_eligible']=e['anchor_eligible']
        conflict=inside&touched&(abs(q['y']-newy)>2*s);q['anchor_eligible'][conflict]=False
        q['geometry_version']='R70_synced';q['confidence']=sample(field,q['x'],q['y']);out.append(q)
        audit.append(dict(ids=list(p['ids']),updated_columns=int(associated.sum()) if shared or associated.sum()>=12 else 0,
                         conflicting_anchor_columns=int(conflict.sum()),raw_alternative_preserved=True))
    return out,audit

def guard_edit(source,candidate,event,width):
    """Protect qualified W positions inside a J proposal; all else stays append/local."""
    x,y=source['x'],source['y'];stable=source.get('protected_W',np.zeros(len(x),bool));lo,hi=event['interval']
    inside=(x>lo)&(x<hi);change=abs(np.interp(x,candidate['x'],candidate['y'])-y);s=sigma(width)
    if np.any((~inside)&(change>1e-8)):return False,'outside_registered_edit_window'
    affected=inside&stable
    if affected.any() and np.any(change[affected]>s):return False,'would_overwrite_current_qualified_W'
    # A compatible, held-out-checked current flank can protect an already normal
    # raw curve too. A wrong old pool must not manufacture its own bad-window proof.
    from scripts.round70_j import _stable_anchor
    from scripts.round64_seams import crop_path
    forecasts=[]
    for mask,right,edge in ((x<=lo,True,lo),(x>=hi,False,hi)):
        if mask.sum()<12:continue
        flank=crop_path(source,mask);flank['anchor_eligible']=source.get('anchor_eligible',source['observed'])[mask]
        anchors=_stable_anchor(flank,right,width,0.)
        if not anchors:continue
        a=anchors[0];distance=x[inside]-a['x']
        prediction=a['y']+a['slope']*distance+.5*a['curvature']*distance**2
        olderror=abs(y[inside]-prediction);newerror=abs(np.interp(x[inside],candidate['x'],candidate['y'])-prediction)
        if np.quantile(olderror,.9)<=2*s and np.mean(newerror)>np.mean(olderror)+s:
            return False,'would_degrade_current_flank_supported_shape'
    return True,'local_edit_respects_current_W'
