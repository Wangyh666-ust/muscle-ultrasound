"""Probability-band localization, two-fragment repair and strict extension. No GT inputs."""
from copy import deepcopy
from collections import Counter
import numpy as np
from scipy.ndimage import label,gaussian_filter1d
from scripts.round63_paths import pack_path,sample,hermite
from scripts.round67_geometry import runs,hypotheses,attach_extension
from scripts.round64_seams import crop_path
from scripts.round66_geometry import fit,noise


def sigma(width):return max(.35,2*width/1046.)


def bands(field,valid,source,threshold=.5):
    """One seeded component, reject branched columns rather than average layers."""
    labels,n=label((field>=threshold)&valid);h,w=field.shape
    ix=np.clip(np.rint(source['x']).astype(int),0,w-1);iy=np.clip(np.rint(source['y']).astype(int),0,h-1)
    seeds=labels[iy,ix];seeds=seeds[seeds>0]
    if not len(seeds):return None,dict(reason='no_seeded_probability_band')
    counts=np.bincount(seeds);most=np.flatnonzero(counts==counts.max())
    if len(most)!=1:return None,dict(reason='ambiguous_seeded_component')
    component=labels==most[0];xs=np.flatnonzero(component.any(0));center=np.full(w,np.nan);widths=np.full(w,np.nan);top=center.copy();bottom=center.copy();branches=0
    for x in xs:
        yy=np.flatnonzero(component[:,x]);parts=np.split(yy,np.flatnonzero(np.diff(yy)>1)+1)
        if len(parts)!=1:branches+=1;continue
        top[x],bottom[x]=yy[0],yy[-1];center[x]=(yy[0]+yy[-1])/2;widths[x]=len(yy)
    return dict(center=center,width=widths,top=top,bottom=bottom),dict(components=n,branched_columns=branches)


def localize(source,field,role,gray,valid,width,cfg):
    if source is None:return None,dict(reason='no_source')
    b,audit=bands(field,valid,source,.5)
    if b is None:return source,audit
    centers=[b['center']];other=[]
    for t in (.35,.65):
        z,_=bands(field,valid,source,t)
        if z is None:return source,dict(reason='cross_threshold_band_missing')
        centers.append(z['center']);other.append(z)
    grid=source['x'];ix=np.rint(grid).astype(int);s=sigma(width)
    stack=np.asarray(centers);finite=np.isfinite(stack).all(0)
    safe=np.nan_to_num(stack);stable=finite&(safe.max(0)-safe.min(0)<=2*s)
    wide=(b['width']>=cfg['min_width_sigma']*s)&(b['width']<=.06*width)
    center=b['center'];offset=abs(center[ix]-source['y'])
    q0=sample(field,grid,source['y']);qc=sample(field,grid,np.nan_to_num(center[ix]))
    qm=sample(field,grid,np.nan_to_num((center[ix]+source['y'])/2))
    eligible=stable[ix]&wide[ix]&(offset>2*s)&(q0>=.5)&(qc>=.5)&(qm>=.8*np.minimum(q0,qc))
    intervals=[(a,b) for a,b in runs(eligible) if b-a>=max(3,.015*width)]
    if not intervals:return source,dict(**audit,reason='normal_or_no_persistent_ambiguous_offset',eligible_columns=int(eligible.sum()))
    # Only authorized ambiguous runs move; untriggered source coordinates stay exact.
    x=grid.copy();y=source['y'].copy();obs=source['observed'].copy();kind=source['kind'].copy();events=list(source['events'])
    changed=np.zeros(len(x),bool)
    for a,b in intervals:
        target=center[ix[a:b]];phase=np.minimum(np.arange(b-a)+1,np.arange(b-a,0,-1))/max(2,round(.005*width))
        blend=np.minimum(phase,1.);y[a:b]+=(target-y[a:b])*blend;changed[a:b]=True
        events.append(dict(type='R68_W',interval=[float(x[a]),float(x[b-1])],reason='stable_wide_band_center',observations_promoted=False))
    obs[changed]=False;kind[changed]=3
    # A band already connected beyond the selected fragment is observed probability,
    # but newly localized points are still derived, not promoted to original observations.
    extra=[]
    for direction in (-1,1):
        edge=int(round(grid[-1 if direction==1 else 0]));step=edge+direction
        if not (0<=step<len(center)) or not eligible[-1 if direction==1 else 0]:continue
        xx=[]
        while 0<=step<len(center) and stable[step] and wide[step]:xx.append(step);step+=direction
        if len(xx)>=2:
            xx=np.asarray(xx,float);extra.append((xx,center[xx.astype(int)]))
            events.append(dict(type='R68_W',interval=[float(xx.min()),float(xx.max())],reason='same_connected_band_continues',observations_promoted=False))
    if extra:
        for xx,yy in extra:
            x=np.r_[x,xx];y=np.r_[y,yy];obs=np.r_[obs,np.zeros(len(xx),bool)];kind=np.r_[kind,np.full(len(xx),3)]
    result=pack_path(x,y,sample(field,x,y),obs,source['ids'],kind=kind,events=events)
    return result,dict(**audit,reason='accepted',moved_original_columns=int(changed.sum()),new_band_columns=len(x)-len(grid),windows=[e['interval'] for e in events if e['type']=='R68_W'])


def stable_anchor(p,right,width,max_retreat):
    """Short and long windows with actual held-out geometric prediction checks."""
    out=[];s=sigma(width);direction=1 if right else -1
    for retreat in np.arange(0,max_retreat+.001,.02):
        edge=p['x'][-1 if right else 0]-direction*retreat*width
        use=p['x']<=edge if right else p['x']>=edge
        if use.sum()<12:continue
        q=crop_path(p,use);x0=q['x'][-1 if right else 0];y0=q['y'][-1 if right else 0];t=direction*(q['x']-x0)/width
        models=[]
        for window in (.06,.10,.18,.28,.40):
            m=(t>=-window)&q['observed']&(q['confidence']>=.18)
            if m.sum()<12 or np.ptp(t[m])<.70*window:continue
            tt,yy,ww=t[m],q['y'][m],q['confidence'][m];train=tt<=-.2*window;held=~train
            if train.sum()<8 or held.sum()<3:continue
            for degree in (1,2):
                c=fit(tt,yy,ww,degree);v=fit(tt[train],yy[train],ww[train],degree)
                residual=float(np.quantile(abs(np.polynomial.polynomial.polyval(tt,c)-yy),.9));error=float(np.quantile(abs(np.polynomial.polynomial.polyval(tt[held],v)-yy[held]),.9))
                if residual>2*s or error>2.5*s or abs(c[0]-y0)>2*s:continue
                models.append(dict(coef=c,window=window,error=error,residual=residual,degree=degree))
        # Two compatible windows are required; a short fragment may use .06 and .10.
        for degree in (1,2):
            mm=[m for m in models if m['degree']==degree]
            if len(mm)<2:continue
            future=np.asarray([np.polynomial.polynomial.polyval([0,.04,.08],m['coef']) for m in mm])
            if np.ptp(future,axis=0).max()>2.5*s:continue
            c=np.median([m['coef'] for m in mm],axis=0)
            out.append(dict(path=q,x=float(x0),y=float(y0),slope=float(direction*c[1]/width),curvature=float(2*c[2]/width**2),
                retreat=float(abs(p['x'][-1 if right else 0]-x0)/width),windows=[m['window'] for m in mm],
                heldout=float(np.mean([m['error'] for m in mm])),residual=float(np.mean([m['residual'] for m in mm])),degree=degree))
            break
    return out


def reconstruct(source,pool,field,role,gray,valid,width,cfg):
    if source is None:return None,dict(reason='no_source')
    s=sigma(width);paths=[]
    for p in [source]+list(pool):
        # Keep original geometry, but do not let duplicate tracks act as two anchors.
        duplicate=False
        for q in paths:
            lo=max(p['x'][0],q['x'][0]);hi=min(p['x'][-1],q['x'][-1]);span=min(np.ptp(p['x']),np.ptp(q['x']))
            if hi>lo and (hi-lo)>=.8*span and max(np.ptp(p['x']),np.ptp(q['x']))<=1.25*max(span,1):
                xx=np.linspace(lo,hi,31)
                if np.max(abs(np.interp(xx,p['x'],p['y'])-np.interp(xx,q['x'],q['y'])))<=2*s:duplicate=True;break
        if not duplicate:paths.append(p)
    audit=Counter();candidates=[];anchors_cache={}
    def aa(i,right):
        key=(i,right)
        if key not in anchors_cache:anchors_cache[key]=stable_anchor(paths[i],right,width,cfg['max_retreat'])
        return anchors_cache[key]
    for i,a in enumerate(paths):
        for k,b in enumerate(paths):
            if i==k or set(a['ids'])&set(b['ids']):continue
            if b['x'][0]<=a['x'][0] or b['x'][-1]<=a['x'][-1]+.005*width:continue
            if b['x'][0]-a['x'][-1]>.20*width or b['x'][0]-a['x'][-1]<-.20*width:continue
            if not set(source['ids'])&set(a['ids']+b['ids']):continue
            for left in aa(i,True):
                for right in aa(k,False):
                    lo,hi=left['x'],right['x'];length=hi-lo
                    if length<=1 or length>.30*width:continue
                    # Compatibility is checked after retreat, never on the corrupted raw endpoints.
                    expected_left=left['y']+left['slope']*length+.5*left['curvature']*length**2
                    expected_right=right['y']-right['slope']*length+.5*right['curvature']*length**2
                    if max(abs(expected_left-right['y']),abs(expected_right-left['y']))>max(4*s,.025*width):audit['stable_fragments_different_layer']+=1;continue
                    xx=np.arange(np.floor(lo)+1,np.ceil(hi));yy,dy,ddy=hermite(xx,lo,left['y'],left['slope'],hi,right['y'],right['slope'])
                    if len(xx)<3 or max(abs(dy),default=0)>1.5 or np.quantile(width*abs(ddy)/(1+dy*dy)**1.5,.95)>12:continue
                    q=sample(field,xx,yy);r=sample(role,xx,yy);g=sample(gray,xx,yy)
                    if not (sample(valid.astype(float),xx,yy)>.99).all():audit['outside_evidence']+=1;continue
                    # Two separated anchors may justify a low-probability gap only when image
                    # evidence also supports the corridor. Shape on its own is insufficient.
                    hit=((q>=.10)&(r>=.18))|(g>=.60)
                    if hit.mean()<.85:audit['rebuild_corridor_missing']+=1;continue
                    inside=(source['x']>lo)&(source['x']<hi)
                    if inside.sum()<3:audit['no_old_bad_region']+=1;continue
                    oldx=source['x'][inside];oldy=source['y'][inside];newy=np.interp(oldx,xx,yy)
                    old_left=left['y']+left['slope']*(oldx-lo)+.5*left['curvature']*(oldx-lo)**2
                    old_right=right['y']+right['slope']*(oldx-hi)+.5*right['curvature']*(oldx-hi)**2
                    # Independent endpoint forecasts agree on where the old line is inconsistent.
                    old_error=float(np.mean(np.minimum(abs(oldy-old_left),abs(oldy-old_right))))
                    new_error=float(np.mean(np.minimum(abs(newy-old_left),abs(newy-old_right))))
                    if old_error<=2*s or old_error-new_error<cfg['improvement']*old_error:audit['no_persistent_shape_correction']+=1;continue
                    if np.mean(abs(oldy-newy)>2*s)<.25:audit['edit_not_substantial']+=1;continue
                    # Large changes need a competing real fragment, not just an attractive smooth fit.
                    if left['retreat']+right['retreat']<=0:audit['no_retreat_evidence']+=1;continue
                    ax,bx=left['path'],right['path']
                    if ax['x'][0]>source['x'][0] or bx['x'][-1]<source['x'][-1]:continue
                    grid=np.unique(np.r_[source['x'],ax['x'],xx,bx['x']]);y=np.interp(grid,source['x'],source['y'])
                    outside_old=(grid<source['x'][0])|(grid>source['x'][-1]);left_add=grid<source['x'][0];right_add=grid>source['x'][-1]
                    y[left_add]=np.interp(grid[left_add],ax['x'],ax['y']);y[right_add]=np.interp(grid[right_add],bx['x'],bx['y'])
                    area=(grid>lo)&(grid<hi);y[area]=np.interp(grid[area],xx,yy)
                    obs=np.zeros(len(grid),bool);kind=np.ones(len(grid),int)
                    idx=np.searchsorted(grid,source['x']);obs[idx]=source['observed'];kind[idx]=source['kind']
                    changed=area&(abs(y-np.interp(grid,source['x'],source['y']))>1e-8);obs[changed]=False;kind[changed]=1
                    for fragment,mask in ((ax,left_add),(bx,right_add)):
                        pos=np.searchsorted(fragment['x'],grid[mask]);inrange=pos<len(fragment['x']);exact=np.zeros(len(pos),bool);exact[inrange]=abs(fragment['x'][pos[inrange]]-grid[mask][inrange])<1e-8
                        oo=np.zeros(len(pos),bool);oo[exact]=fragment['observed'][pos[exact]];obs[mask]=oo;kind[mask]=np.where(oo,0,1)
                    event=dict(type='R68_J',interval=[float(lo),float(hi)],source_ids=[list(a['ids']),list(b['ids'])],
                        anchors=[[lo,left['y']],[hi,right['y']]],retreats=[left['retreat'],right['retreat']],windows=[left['windows'],right['windows']],
                        old_forecast_error=old_error,new_forecast_error=new_error,probability=float(q.mean()),gray=float(g.mean()),observations_promoted=False)
                    p=pack_path(grid,y,sample(field,grid,y),obs,tuple(sorted(set(source['ids']+a['ids']+b['ids']))),kind=kind,events=source['events']+[event])
                    merit=(old_error-new_error)/s-.5*(left['retreat']+right['retreat'])-left['heldout']/s-right['heldout']/s
                    candidates.append((round(merit,6),p))
    if not candidates:return source,dict(reason='no_qualified_two_fragment_repair',pool=len(paths),rejected=dict(audit))
    best=max(v[0] for v in candidates);ties=[p for merit,p in candidates if merit==best]
    if len(ties)>1:
        # Identical geometry is harmless; conflicting equal evidence preserves the original.
        xx=source['x'];reference=np.interp(xx,ties[0]['x'],ties[0]['y'])
        if any(np.max(abs(np.interp(xx,p['x'],p['y'])-reference))>s for p in ties[1:]):return source,dict(reason='ambiguous_equal_rebuilds',candidates=len(candidates))
    return ties[0],dict(reason='accepted',pool=len(paths),candidates=len(candidates),event=ties[0]['events'][-1],rejected=dict(audit))


def extend(source,field,role,gray,valid,width,cfg,shape_cfg):
    if source is None:return None,[],dict(reason='no_source')
    base_cfg=dict(maximum_extension=.12,required_support_fraction=.85)
    proposals,generation=hypotheses(source,field,gray,valid,width,shape_cfg);accepted=[];audit=Counter();s=sigma(width)
    for direction in (-1,1):
        candidates=[]
        for p in proposals:
            if p['events'][-1]['direction']!=direction:continue
            joined,decision=attach_extension(source,p,field,gray,valid,width,base_cfg)
            if joined is None:audit[decision['reason']]+=1;continue
            xx=joined['x'];yy=joined['y'];new=(xx>source['x'][-1]) if direction==1 else (xx<source['x'][0])
            x,y=xx[new],yy[new];order=np.argsort(direction*(x-source['x'][-1 if direction==1 else 0]));x,y=x[order],y[order]
            q=sample(field,x,y);r=sample(role,x,y);hit=(q>=.1)&(r>=.18);center_hit=np.zeros(len(x),bool)
            for n,(xc,yc) in enumerate(zip(x,y)):
                ix=int(round(xc));iy=int(round(yc))
                if not (0<=ix<field.shape[1] and 0<=iy<field.shape[0]) or field[iy,ix]<.1:continue
                col=(field[:,ix]>=.1)&(role[:,ix]>=.18)&valid[:,ix];lo=hi=iy
                if not col[iy]:continue
                while lo>0 and col[lo-1]:lo-=1
                while hi<len(col)-1 and col[hi+1]:hi+=1
                center_hit[n]=abs(yc-(lo+hi)/2)<=2*s
            hit&=center_hit;inside=sample(valid.astype(float),x,y)>.99
            stop=len(x);blind=0
            for n in range(len(x)):
                blind=0 if hit[n] else blind+1
                if not inside[n] or blind>max(1,.015*width):stop=max(0,n-blind+1);break
            while stop>0 and not hit[stop-1]:stop-=1
            x,y,hit=x[:stop],y[:stop],hit[:stop]
            if len(x)<max(2,.01*width) or hit.mean()<cfg['support_fraction']:audit['strict_probability_or_band_position_failed']+=1;continue
            event=dict(type='R68_X',interval=[float(x.min()),float(x.max())],direction=direction,
                support_fraction=float(hit.mean()),probability=float(sample(field,x,y).mean()),role_probability=float(sample(role,x,y).mean()),
                brightness_alone_qualifies=False,observations_promoted=False)
            candidates.append((round(float(hit.sum()),6),x,y,event))
        if candidates:accepted.append(max(candidates,key=lambda c:c[0]))
    if not accepted:return source,proposals,dict(reason='no_probability_supported_extension',generation=generation,rejected=dict(audit))
    x=source['x'];y=source['y'];obs=source['observed'];kind=source['kind'];events=list(source['events'])
    for _,xx,yy,event in accepted:
        x=np.r_[x,xx];y=np.r_[y,yy];obs=np.r_[obs,np.zeros(len(xx),bool)];kind=np.r_[kind,np.full(len(xx),2)];events.append(event)
    return pack_path(x,y,sample(field,x,y),obs,source['ids'],kind=kind,events=events),proposals,dict(reason='accepted',generation=generation,rejected=dict(audit),events=[p[3] for p in accepted])
