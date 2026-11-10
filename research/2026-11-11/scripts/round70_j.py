"""Isolated R68 J, with explicit anchor eligibility and per-proposal edit guard.
Generated from the protected R68 source before protocol registration. Raw point
provenance is never promoted. Old algorithm/files remain untouched.
"""
import numpy as np
from collections import Counter
from scripts.round63_paths import pack_path,sample,hermite
from scripts.round64_seams import crop_path
from scripts.round66_geometry import fit
from scripts.round68_geometry import sigma,stable_anchor as legacy_anchor
from copy import deepcopy
from scripts.round63_common import array_sha

def variants(source,pool,synced,field,role,gray,valid,width,cfg,guard_function,verify=False):
    """Reuse only identical effective states or a surviving highest-score edit."""
    cache={};results={};states={}
    def key(paths,mode):
        return tuple((tuple(p['ids']),array_sha(np.stack([p['x'],p['y'],p['confidence'],p['observed'],p['kind'],p.get('anchor_eligible',p['observed']) if mode else p['observed']]))) for p in [source]+paths)
    for arm,mode,useguard in [('J00',False,False),('J10',True,False),('J01',False,True),('WJ',True,True)]:
        paths=synced if mode else pool;k=key(paths,mode);states[arm]=(paths,mode,useguard)
        if k not in cache:cache[k]=reconstruct(source,paths,field,role,gray,valid,width,cfg,anchor_mode=mode)
        out,audit=deepcopy(cache[k]);reuse='identical_effective_anchor_state'
        if useguard:
            if audit.get('reason')=='accepted':
                ok,why=guard_function(source,out,audit['event'],width)
                if not ok:out,audit=reconstruct(source,paths,field,role,gray,valid,width,cfg,anchor_mode=mode,guard=guard_function);reuse='full_guard_recompute'
                else:reuse='highest_score_edit_passed_guard'
            elif audit.get('reason')!='no_qualified_two_fragment_repair':
                out,audit=reconstruct(source,paths,field,role,gray,valid,width,cfg,anchor_mode=mode,guard=guard_function);reuse='full_ambiguity_recompute'
        if audit.get('event'):
            audit['event']['anchor_mode']='independent_eligibility' if mode else 'legacy_observed'
            audit['event']['effective_state_equivalence_reused']=reuse
        audit['computation_reuse']=reuse
        if verify:
            direct,da=reconstruct(source,paths,field,role,gray,valid,width,cfg,anchor_mode=mode,guard=guard_function if useguard else None)
            assert np.array_equal(out['x'],direct['x']) and np.max(abs(out['y']-direct['y']),initial=0)<1e-8,(arm,'J reuse changed geometry')
            assert np.array_equal(out['observed'],direct['observed']) and np.array_equal(out['kind'],direct['kind'])
            audit['direct_four_arm_equivalence_verified']=True
        results[arm]=(out,audit)
    return results

def _stable_anchor(p,right,width,max_retreat):
    """Short and long windows with actual held-out geometric prediction checks."""
    out=[];s=sigma(width);direction=1 if right else -1
    for retreat in np.arange(0,max_retreat+.001,.02):
        edge=p['x'][-1 if right else 0]-direction*retreat*width
        use=p['x']<=edge if right else p['x']>=edge
        if use.sum()<12:continue
        q=crop_path(p,use);q['anchor_eligible']=p.get('anchor_eligible',p['observed'])[use];x0=q['x'][-1 if right else 0];y0=q['y'][-1 if right else 0];t=direction*(q['x']-x0)/width
        models=[]
        for window in (.06,.10,.18,.28,.40):
            m=(t>=-window)&q.get('anchor_eligible',q['observed'])&(q['confidence']>=.18)
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


def reconstruct(source,pool,field,role,gray,valid,width,cfg,anchor_mode=False,guard=None):
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
        if key not in anchors_cache:anchors_cache[key]=(_stable_anchor if anchor_mode else legacy_anchor)(paths[i],right,width,cfg['max_retreat'])
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
                    event['anchor_geometry_versions']=[paths[i].get('geometry_version','legacy_raw'),paths[k].get('geometry_version','legacy_raw')]
                    event['anchor_mode']='independent_eligibility' if anchor_mode else 'legacy_observed'
                    if guard is not None:
                        allowed,why=guard(source,p,event,width)
                        if not allowed:audit[why]+=1;continue
                    candidates.append((round(merit,6),p))
    if not candidates:return source,dict(reason='no_qualified_two_fragment_repair',pool=len(paths),rejected=dict(audit))
    best=max(v[0] for v in candidates);ties=[p for merit,p in candidates if merit==best]
    if len(ties)>1:
        # Identical geometry is harmless; conflicting equal evidence preserves the original.
        xx=source['x'];reference=np.interp(xx,ties[0]['x'],ties[0]['y'])
        if any(np.max(abs(np.interp(xx,p['x'],p['y'])-reference))>s for p in ties[1:]):return source,dict(reason='ambiguous_equal_rebuilds',candidates=len(candidates))
    return ties[0],dict(reason='accepted',pool=len(paths),candidates=len(candidates),event=ties[0]['events'][-1],rejected=dict(audit))

