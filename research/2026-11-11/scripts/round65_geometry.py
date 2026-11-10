"""Direction-neutral local boundary repair. This module never accepts GT."""
from copy import deepcopy
from collections import Counter
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scripts.round63_paths import pack_path,bridge,path_quality,sample,hermite
from scripts.round64_seams import crop_path


def scale_noise(width,cfg):return max(.35,cfg['noise_pixels']*width/cfg['nominal_width'])


def anchor_fit(path,right,width,cfg,retreat=0.):
    """Fit a stable observed window pointing inward from either endpoint."""
    x,y=path['x'],path['y'];at=x[-1]-retreat*width if right else x[0]+retreat*width
    keep=x<=at if right else x>=at
    if keep.sum()<8:return None
    p=crop_path(path,keep);x0=p['x'][-1 if right else 0];y0=p['y'][-1 if right else 0]
    near=abs(p['x']-x0)<=cfg['anchor_fraction']*width
    good=near&p['observed']&(p['confidence']>=cfg['low'])
    if good.sum()<8 or np.ptp(p['x'][good])<max(4,.015*width):return None
    u=(p['x'][good]-x0)/width;v=p['y'][good]/width
    A=np.column_stack((np.ones(len(u)),u,u*u));w=np.sqrt(np.maximum(p['confidence'][good],.05))
    coef=np.linalg.lstsq(A*w[:,None],v*w,rcond=None)[0]
    for _ in range(4):
        e=v-A@coef;s=max(scale_noise(width,cfg)/width,1.4826*np.median(abs(e-np.median(e))))
        weights=w*np.sqrt(np.minimum(1.,1.5*s/np.maximum(abs(e),1e-10)))
        coef=np.linalg.lstsq(A*weights[:,None],v*weights,rcond=None)[0]
    residual=np.quantile(abs(v-A@coef),.9)*width
    if residual>2*scale_noise(width,cfg):return None
    if abs(coef[1])>cfg['max_track_turn']:return None
    if abs(coef[0]*width-y0)>2*scale_noise(width,cfg):return None
    return dict(path=p,x=float(x0),y=float(y0),slope=float(coef[1]),
        curvature=float(np.clip(2*coef[2]/width,-12/width,12/width)),
        residual=float(residual),retreat=retreat,right=right,observations=int(good.sum()))


def corridor(x,y,field,gray,valid,width):
    radius=max(1,round(width/1046.))
    q=sample(field,x,y,radius);g=sample(gray,x,y,radius)
    v=sample(valid.astype(float),x,y)>.99;hit=(q>=.10)|(g>=.60)
    cuts=np.flatnonzero(np.diff(np.r_[False,~hit,False]));run=int(np.max(cuts[1::2]-cuts[::2],initial=0))
    return dict(probability=q,gray=g,valid=bool(v.all()),hit_fraction=float(hit.mean()),
        longest_blind=run,passed=bool(v.all() and hit.mean()>=.80 and run<=max(1,.02*width)))


def preserve_sources(result,sources,window):
    """Numerical guard, not a metadata promise, on unchanged source observations."""
    lo,hi=window
    for p in sources:
        use=((p['x']<lo)|(p['x']>hi))&(p['x']>=result['x'][0])&(p['x']<=result['x'][-1])
        if np.any(abs(np.interp(p['x'][use],result['x'],result['y'])-p['y'][use])>1e-8):return False
    return True


def narrow_reconstruct(paths,field,gray,valid,width,cfg,base):
    """N: gap-only version of S; overlapping inputs do not call overlap fusion."""
    candidates=[];audit=Counter();seen=set();fractions=[0.,.02,.04]
    for a in paths:
        for b in paths:
            if set(a['ids'])&set(b['ids']):continue
            if b['x'][0]<=a['x'][0] or b['x'][-1]<=a['x'][-1]+.005*width:continue
            if b['x'][0]-a['x'][-1]>.20*width:continue
            proposals=[]
            for left in fractions:
                for right in fractions:
                    if not left and not right:continue
                    ma=a['x']<=a['x'][-1]-left*width;mb=b['x']>=b['x'][0]+right*width
                    if ma.sum()<8 or mb.sum()<8:continue
                    aa,bb=crop_path(a,ma),crop_path(b,mb)
                    if bb['x'][0]<=aa['x'][-1]:audit['overlap_not_gap']+=1;continue
                    if aa['confidence'][-1]<.5 or bb['confidence'][0]<.5:continue
                    p,reason=bridge(aa,bb,field,gray,valid,width,base)
                    if p is None:audit[reason]+=1;continue
                    lo,hi=float(aa['x'][-1]),float(bb['x'][0]);xx=np.linspace(lo,hi,max(9,int(hi-lo)+1))[1:-1]
                    if not len(xx):audit['empty_gap_check']+=1;continue
                    quality=corridor(xx,np.interp(xx,p['x'],p['y']),field,gray,valid,width)
                    if not quality['passed']:audit['unreliable_gap_corridor']+=1;continue
                    if not preserve_sources(p,(aa,bb),(lo,hi)):raise AssertionError('N moved an outside source point')
                    p['events'].append(dict(type='N_gap',interval=[lo,hi],outside_window_preserved=True,
                        trusted=False,probability_hits=quality['hit_fraction']))
                    proposals.append(p)
            if proposals:
                p=max(proposals,key=lambda q:path_quality(q,width))
                signature=(p['ids'],p['x'][0],p['x'][-1])
                if signature not in seen:candidates.append(p);seen.add(signature)
    return sorted(candidates,key=lambda p:-path_quality(p,width))[:10],dict(audit)


def bridge_from_anchors(a,b,field,gray,valid,width,cfg):
    x0,x1=a['x'],b['x'];length=x1-x0
    if length<1 or length>cfg['local_window']*width:return None,'window_out_of_range'
    normal=max(abs(b['y']-a['y']-a['slope']*length)/np.sqrt(1+a['slope']**2),
               abs(a['y']-b['y']+b['slope']*length)/np.sqrt(1+b['slope']**2))
    if normal>.04*width:return None,'different_layer_trend_distance'
    x=np.arange(np.floor(x0)+1,np.ceil(x1),dtype=float)
    test=np.linspace(x0,x1,max(17,round(length)+1))
    y,dy,ddy=hermite(test,x0,a['y'],a['slope'],x1,b['y'],b['slope'])
    bend=width*abs(ddy)/(1+dy*dy)**1.5
    if np.max(abs(dy))>cfg['max_track_turn'] or np.quantile(bend,.95)>12:return None,'anchor_trend_conflict'
    # A narrow evidence correction cannot jump to a separate strong tissue layer.
    offsets=np.linspace(-scale_noise(width,cfg),scale_noise(width,cfg),9)
    xx=np.repeat(test[:,None],len(offsets),1);yy=y[:,None]+offsets[None]
    signal=.65*sample(field,xx.ravel(),yy.ravel()).reshape(yy.shape)+.35*sample(gray,xx.ravel(),yy.ravel()).reshape(yy.shape)
    signal-=.08*(offsets[None]/max(scale_noise(width,cfg),.1))**2
    correction=offsets[signal.argmax(1)];correction=gaussian_filter1d(correction,max(1.,len(test)/24.))
    t=(test-x0)/length;correction*=16*t*t*(1-t)*(1-t)
    revised=y+correction;check=corridor(test[1:-1],revised[1:-1],field,gray,valid,width)
    if not check['passed']:return None,'no_reliable_local_corridor'
    ay=a['path'];by=b['path'];new_y=np.interp(x,test,revised)
    p=pack_path(np.r_[ay['x'],x,by['x']],np.r_[ay['y'],new_y,by['y']],
        np.r_[ay['confidence'],sample(field,x,new_y),by['confidence']],
        np.r_[ay['observed'],np.zeros(len(x),bool),by['observed']],ay['ids']+by['ids'],
        kind=np.r_[ay['kind'],np.ones(len(x),int),by['kind']],events=ay['events']+by['events']+[
            dict(type='T_gap',interval=[x0,x1],anchors=[[x0,a['y']],[x1,b['y']]],
                slopes=[a['slope'],b['slope']],retreats=[a['retreat'],b['retreat']],
                outside_window_preserved=True,observations_promoted=False)])
    if not preserve_sources(p,(ay,by),(x0,x1)):raise AssertionError('T moved an outside source point')
    p['repair_quality']=float(.65*check['probability'].mean()+.35*check['gray'].mean()-
        cfg['curvature_cost']*np.quantile(bend,.95)-.10*(a['retreat']+b['retreat']))
    return p,None


def retreat_reconstruct(paths,field,gray,valid,width,cfg,base):
    """T: symmetric trimming search; stable anchors determine the direction."""
    audit=Counter();out=[];fractions=np.unique(np.r_[0.,np.arange(.04,cfg['trim_fraction']+.001,.04)])
    anchors={}
    for i,p in enumerate(paths):
        for right in (False,True):
            anchors[i,right]=[v for f in fractions if (v:=anchor_fit(p,right,width,cfg,float(f))) is not None]
    for i,a in enumerate(paths):
        for j,b in enumerate(paths):
            if set(a['ids'])&set(b['ids']):continue
            original_gap=b['x'][0]-a['x'][-1]
            if original_gap<=0 or original_gap>cfg['local_window']*width:continue
            if len(set(a['ids']+b['ids']))>base['max_chain']:continue
            original,reason=bridge(a,b,field,gray,valid,width,base)
            if original is not None:continue # preserve an already admissible connection
            candidates=[]
            for aa in anchors[i,True]:
                for bb in anchors[j,False]:
                    if not aa['retreat'] and not bb['retreat']:continue
                    p,why=bridge_from_anchors(aa,bb,field,gray,valid,width,cfg)
                    if p is None:audit[why]+=1
                    else:candidates.append(p)
            if candidates:out.append(max(candidates,key=lambda p:p['repair_quality']))
    return sorted(out,key=lambda p:-p['repair_quality'])[:10],dict(audit)


def protected_candidate(candidate,reference,width,cfg):
    """P: keep reference outside explicit, nonempty repair windows exactly."""
    if reference is None:return None,'no_reference_to_protect'
    p=deepcopy(candidate);windows=[e['interval'] for e in p['events'] if e['type'] in ('N_gap','T_gap') and e['interval'][1]>e['interval'][0]]
    common=(p['x']>=reference['x'][0])&(p['x']<=reference['x'][-1]);allowed=~common
    for lo,hi in windows:allowed|=(p['x']>=lo)&(p['x']<=hi)
    old=np.interp(p['x'],reference['x'],reference['y']);different=abs(p['y']-old)>1e-7
    if np.any(different&common&~allowed):return None,'nonlocal_output_replacement'
    # Keep all old coordinates outside the union of explicit windows.
    rx=reference['x'];retain=np.ones(len(rx),bool)
    for lo,hi in windows:retain&=~((rx>=lo)&(rx<=hi))
    inside=(rx>=p['x'][0])&(rx<=p['x'][-1]);retain&=inside
    if np.any(abs(np.interp(rx[retain],p['x'],p['y'])-reference['y'][retain])>1e-7):
        return None,'outside_window_coordinates_changed'
    return p,None


def roughness(path,width):
    """Physical-window roughness, not adjacent-pixel derivative noise."""
    if len(path['x'])<5:return 0.
    x=np.linspace(path['x'][0],path['x'][-1],max(5,round(np.ptp(path['x'])/max(1,.01*width))))
    y=np.interp(x,path['x'],path['y']);d=np.gradient(y,x);k=width*abs(np.gradient(d,x))/(1+d*d)**1.5
    return float(np.quantile(k,.9))


def protected_score(pair,width,cfg):
    score=0.
    for side in ('upper','lower'):
        p=pair[side];observed=p['observed'].sum()/width;span=np.ptp(p['x'])/width
        score+=.5*(.70*min(float(np.mean(p['confidence'])),.9)+.10*min(observed,.75)+
            .05*min(span,.75)-cfg['curvature_cost']*min(roughness(p,width),20))
    score+=.20*(pair['c1']-.5) if pair['c1_active'] else 0.
    return float(score-pair['prior_penalty'])


def extension_path(path,right,field,gray,valid,width,cfg,exploratory=False,anchor=None):
    """E: evidence-guided dynamic path from either stable side; no GT or side preference."""
    anchor=anchor if anchor is not None else anchor_fit(path,right,width,cfg)
    if anchor is None:return None
    source=path;path=anchor['path']
    direction=1 if right else -1;x0,y0=anchor['x'],anchor['y'];step=max(1.,width/96.)
    maximum=cfg['extension_fraction']*width
    if exploratory:
        domain=np.flatnonzero(valid.any(0))
        maximum=(domain[-1]-x0) if right else (x0-domain[0])
        maximum=min(maximum,.40*width)
    maximum=max(0.,float(np.ceil(maximum)))
    if maximum<2:return None
    distances=np.unique(np.r_[0.,np.arange(step,maximum,step),maximum])
    xx=x0+direction*distances;m0=anchor['slope'];k0=anchor['curvature']
    prior=y0+m0*(xx-x0)+.5*k0*(xx-x0)**2/(1+distances/max(.08*width,1))
    sigma=scale_noise(width,cfg)+.02*distances+.04*distances*distances/width
    nodes=[prior[i]+np.linspace(-1,1,17)*min(.012*width,2*sigma[i]) for i in range(len(xx))]
    nodes[0]=np.full(17,y0);emission=[]
    for i,(x,yy) in enumerate(zip(xx,nodes)):
        q=sample(field,np.full(len(yy),x),yy);g=sample(gray,np.full(len(yy),x),yy)
        v=sample(valid.astype(float),np.full(len(yy),x),yy)
        c=-.65*q-.35*g+.08*((yy-prior[i])/max(sigma[i],.3))**2
        c[v<.99]=1e7;emission.append(c)
    slopes=(nodes[1][None]-nodes[0][:,None])/(xx[1]-xx[0]);cost=emission[0][:,None]+emission[1][None]+.05*(slopes-m0)**2
    pointers={}
    for i in range(2,len(xx)):
        previous=(nodes[i-1][None]-nodes[i-2][:,None])/(xx[i-1]-xx[i-2])
        following=(nodes[i][None]-nodes[i-1][:,None])/(xx[i]-xx[i-1])
        bend=width*(following[None]-previous[:,:,None])/max(abs(xx[i]-xx[i-2])/2,1)
        trial=cost[:,:,None]+cfg['curvature_cost']*bend*bend
        best=trial.argmin(0);cost=np.take_along_axis(trial,best[None],axis=0)[0]+emission[i][None]
        cost[abs(following)>cfg['max_track_turn']]=1e7;pointers[i]=best
    if cost.min()>1e6:return None
    indices=np.empty(len(xx),int);indices[-2:]=np.unravel_index(np.argmin(cost),cost.shape)
    for i in range(len(xx)-1,1,-1):indices[i-2]=pointers[i][indices[i-1],indices[i]]
    chosen=np.asarray([v[i] for v,i in zip(nodes,indices)]);order=np.argsort(xx)
    grid=x0+direction*np.arange(1,int(maximum)+1,dtype=float)
    y=PchipInterpolator(xx[order],chosen[order],extrapolate=False)(grid)
    q=sample(field,grid,y,max(1,round(width/1046.)));g=sample(gray,grid,y,max(1,round(width/1046.)))
    good=(q>=.1)|(g>=.6);ok=sample(valid.astype(float),grid,y)>.99
    stop=len(grid);run=0
    for i in range(len(grid)):
        run=0 if good[i] else run+1
        uncertainty=scale_noise(width,cfg)+.02*(i+1)+.04*(i+1)**2/width
        if not ok[i] or uncertainty>.02*width or (not exploratory and run>max(1,.015*width)):
            stop=max(0,i-run+1) if not exploratory and run else i;break
    if stop<2:return None
    grid,y,q,g=grid[:stop],y[:stop],q[:stop],g[:stop]
    credible=bool(not exploratory and ((q>=.1)|(g>=.6)).mean()>=.8)
    new_kind=np.where((grid>=source['x'][0])&(grid<=source['x'][-1]),1,2)
    p=pack_path(np.r_[path['x'],grid],np.r_[path['y'],y],np.r_[path['confidence'],q],
        np.r_[path['observed'],np.zeros(len(grid),bool)],path['ids'],
        kind=np.r_[path['kind'],new_kind],events=path['events']+[
            dict(type='E_continuation',interval=[float(grid.min()),float(grid.max())],
                anchor=[x0,y0],direction=direction,exploratory=exploratory,reliable_candidate=credible,
                retreat_fraction=anchor['retreat'],
                observations_promoted=False,uncertainty_end=float(scale_noise(width,cfg)+.02*len(grid)+.04*len(grid)**2/width))])
    return p


def extensions(paths,field,gray,valid,width,cfg,exploratory=False):
    out=[]
    for path in paths:
        for right in (False,True):
            anchors=[]
            initial=anchor_fit(path,right,width,cfg)
            if initial is not None:anchors.append(initial)
            for fraction in np.arange(.04,cfg['trim_fraction']+.001,.04):
                anchor=anchor_fit(path,right,width,cfg,float(fraction))
                if anchor is None:continue
                x0=anchor['x'];discarded=path['x']>x0 if right else path['x']<x0
                if discarded.sum()<4:continue
                dx=path['x'][discarded]-x0
                expected=anchor['y']+anchor['slope']*dx+.5*anchor['curvature']*dx*dx/(1+abs(dx)/max(.08*width,1))
                # Retirement requires actual disagreement, not merely a longer available fit.
                if np.quantile(abs(path['y'][discarded]-expected),.75)>2*scale_noise(width,cfg):anchors.append(anchor)
            for anchor in anchors:
                p=extension_path(path,right,field,gray,valid,width,cfg,exploratory,anchor)
                if p is not None:out.append(p)
    return out
