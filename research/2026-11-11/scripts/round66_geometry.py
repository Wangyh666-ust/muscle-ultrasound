"""Consensus anchors with held-out continuation checks. No GT is accepted here."""
from copy import deepcopy
from itertools import combinations
import numpy as np
from scipy.interpolate import CubicHermiteSpline,PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scripts.round63_paths import pack_path,sample,hermite
from scripts.round64_seams import crop_path


def noise(width,cfg):return max(.35,cfg['noise_nominal_px']*width/cfg['nominal_width'])


def fit(t,y,weights,degree):
    A=np.column_stack([t**i for i in range(degree+1)]);w=np.sqrt(np.maximum(weights,.05))
    coef=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0]
    for _ in range(4):
        residual=y-A@coef;scale=max(.001,1.4826*np.median(abs(residual-np.median(residual))))
        rw=w*np.sqrt(np.minimum(1,1.5*scale/np.maximum(abs(residual),1e-12)))
        coef=np.linalg.lstsq(A*rw[:,None],y*rw,rcond=None)[0]
    return np.pad(coef,(0,3-len(coef)))


def anchor_at(path,right,width,cfg,retreat=0.,allow_curve=True):
    direction=1 if right else -1;edge=path['x'][-1 if right else 0]
    target=edge-direction*retreat*width
    keep=path['x']<=target+1e-8 if right else path['x']>=target-1e-8
    if keep.sum()<12:return None
    p=crop_path(path,keep);x0=p['x'][-1 if right else 0];y0=p['y'][-1 if right else 0]
    t=direction*(p['x']-x0)/width;sigma=noise(width,cfg);models={1:[],2:[]}
    for fraction in cfg['windows']:
        use=(t>=-fraction-1e-8)&p['observed']&(p['confidence']>=cfg['confidence_floor'])
        if use.sum()<12 or np.ptp(t[use])<.70*fraction:continue
        tt=t[use];yy=p['y'][use];ww=p['confidence'][use]
        train=tt<=-.20*fraction;held=~train
        if train.sum()<8 or held.sum()<3:continue
        for degree in (1,2):
            if degree==2 and not allow_curve:continue
            coef=fit(tt,yy,ww,degree);pred=np.polynomial.polynomial.polyval(tt,coef)
            validation=fit(tt[train],yy[train],ww[train],degree)
            errors=abs(np.polynomial.polynomial.polyval(tt[held],validation)-yy[held])
            hold=float(np.quantile(errors,.9));res=float(np.quantile(abs(pred-yy),.9))
            if hold>cfg['holdout_error_noise']*sigma or res>cfg['fit_error_noise']*sigma:continue
            if abs(coef[0]-y0)>cfg['fit_error_noise']*sigma:continue
            slope=coef[1]/width;curvature=2*coef[2]/width**2
            if abs(slope)>1.5 or abs(curvature)*width>12:continue
            future=np.asarray([.04,.08]);forecast=y0+coef[1]*future+coef[2]*future**2
            models[degree].append(dict(coef=coef,window=fraction,held=hold,residual=res,future=forecast))
    choices={}
    for degree,items in models.items():
        compatible=[]
        for size in range(len(items),cfg['minimum_windows']-1,-1):
            for group in combinations(items,size):
                if max(m['window'] for m in group)<cfg['minimum_long_window']:continue
                forecasts=np.asarray([m['future'] for m in group])
                spread=float(np.ptp(forecasts,axis=0).max())
                if spread<=cfg['agreement_noise']*sigma:
                    quality=np.mean([m['held']+m['residual'] for m in group])/sigma+.15*spread/sigma
                    compatible.append((quality,group,spread))
            if compatible:break
        if compatible:
            quality,group,spread=min(compatible,key=lambda q:(q[0],-max(m['window'] for m in q[1])))
            coef=np.median([m['coef'] for m in group],axis=0)
            choices[degree]=dict(coef=coef,quality=float(quality),spread=spread,
                windows=[m['window'] for m in group],held=float(np.mean([m['held'] for m in group])),
                residual=float(np.mean([m['residual'] for m in group])))
    if not choices:return None
    degree=1 if 1 in choices else 2
    if 2 in choices and 1 in choices:
        improvement=choices[2]['held']<=(1-cfg['curved_holdout_improvement'])*choices[1]['held']
        if improvement and choices[1]['held']>.25*sigma:degree=2
    a=choices[degree];coef=a.pop('coef')
    return dict(path=p,x=float(x0),y=float(y0),direction=direction,right=right,
        slope=float(direction*coef[1]/width),curvature=float(2*coef[2]/width**2 if degree==2 else 0),
        degree=degree,retreat=float(abs(x0-edge)/width),**a)


def anchors(path,right,width,cfg,variant):
    fs=cfg['retreats'] if variant['retreat'] else [0.]
    candidates=[]
    for f in fs:
        a=anchor_at(path,right,width,cfg,f,variant['curved'])
        if a is not None:
            if a['retreat']>0:
                discarded=(path['x']>a['x']) if right else (path['x']<a['x'])
                d=a['direction']*(path['x'][discarded]-a['x'])
                expected,_=prior(a,d,width,cfg)
                disagreement=float(np.quantile(abs(path['y'][discarded]-expected),.75)) if discarded.any() else 0.
                if disagreement<=2*noise(width,cfg):
                    # An interior stable window may estimate direction without deleting valid observations.
                    edge=path['x'][-1 if right else 0];distance=abs(edge-a['x'])
                    _,slope=prior(a,distance,width,cfg)
                    a={**a,'path':path,'x':float(edge),'y':float(path['y'][-1 if right else 0]),
                       'slope':float(a['direction']*slope),'curvature':float(a['curvature']*np.exp(-distance/(cfg['curvature_decay_width']*width))),
                       'retreat':0.,'direction_estimated_inside':float(distance/width)}
            candidates.append(a)
        if len(candidates)>=3:break
    # Fixed trust checks precede the minimum-removal preference; never rank by new span.
    return sorted(candidates,key=lambda a:(a['retreat'],a['quality']))


def prior(a,distance,width,cfg):
    d=np.asarray(distance);L=cfg['curvature_decay_width']*width;k=a['curvature']
    yy=a['y']+a['direction']*a['slope']*d+k*L*(d-L*(1-np.exp(-d/L)))
    slope=a['direction']*a['slope']+k*L*(1-np.exp(-d/L))
    return yy,slope


def extend_from_anchor(source,a,field,gray,valid,width,cfg,exploratory=False):
    direction=a['direction'];x0=a['x'];sigma=noise(width,cfg)
    old_edge=source['x'][-1 if direction==1 else 0]
    rebuild=abs(old_edge-x0);outside=cfg['maximum_exploratory_extension' if exploratory else 'maximum_reliable_extension']*width
    domain=np.flatnonzero(valid.any(0));limit=(domain[-1]-x0) if direction==1 else (x0-domain[0])
    maximum=int(np.floor(min(rebuild+outside,limit)))
    if maximum<2:return None
    step=max(1.,width/96.);d=np.unique(np.r_[0.,np.arange(step,maximum,step),float(maximum)])
    predicted,derivative=prior(a,d,width,cfg)
    uncertainty=sigma+a['spread']*d/max(.08*width,1)+.008*d+.015*d*d/width
    radius=np.minimum(.012*width,2*uncertainty)
    nodes=[predicted[i]+np.linspace(-1,1,17)*radius[i] for i in range(len(d))]
    nodes[0]=np.full(17,a['y'])
    # Initial direction is part of the feasible path, rather than only a small soft penalty.
    nodes[1]=np.full(17,predicted[1])
    emissions=[]
    for i,(distance,yy) in enumerate(zip(d,nodes)):
        xx=np.full(len(yy),x0+direction*distance)
        q=sample(field,xx,yy);g=sample(gray,xx,yy);v=sample(valid.astype(float),xx,yy)
        cost=-.65*q-.35*g+.20*((yy-predicted[i])/max(uncertainty[i],.35))**2
        cost[v<.99]=1e7;emissions.append(cost)
    cost=emissions[0][:,None]+emissions[1][None];pointers={}
    for i in range(2,len(d)):
        previous=(nodes[i-1][None]-nodes[i-2][:,None])/(d[i-1]-d[i-2])
        following=(nodes[i][None]-nodes[i-1][:,None])/(d[i]-d[i-1])
        bend=width*(following[None]-previous[:,:,None])/((d[i]-d[i-2])/2)
        trial=cost[:,:,None]+.10*bend*bend
        best=trial.argmin(0);cost=np.take_along_axis(trial,best[None],axis=0)[0]+emissions[i][None]
        cost[abs(following)>1.5]=1e7;pointers[i]=best
    if cost.min()>1e6:return None
    index=np.empty(len(d),int);index[-2:]=np.unravel_index(np.argmin(cost),cost.shape)
    for i in range(len(d)-1,1,-1):index[i-2]=pointers[i][index[i-1],index[i]]
    chosen=np.asarray([node[ix] for node,ix in zip(nodes,index)])
    slopes=PchipInterpolator(d,chosen).derivative()(d);slopes[0]=direction*a['slope']
    interpolator=CubicHermiteSpline(d,chosen,slopes)
    distance=np.arange(1,maximum+1,dtype=float);xx=x0+direction*distance;yy=interpolator(distance)
    q=sample(field,xx,yy,max(1,round(width/1046.)));g=sample(gray,xx,yy,max(1,round(width/1046.)))
    hit=(q>=.1)|(g>=.6);inside=sample(valid.astype(float),xx,yy)>.99
    stop=len(xx);run=0
    for i in range(len(xx)):
        run=0 if hit[i] else run+1
        uncertain=sigma+a['spread']*distance[i]/max(.08*width,1)+.008*distance[i]+.015*distance[i]**2/width
        if not inside[i] or uncertain>.025*width or (not exploratory and run>max(1,cfg['maximum_blind_width']*width)):
            stop=max(0,i-run+1) if not exploratory and run else i;break
    if stop<2:return None
    xx,yy,q,hit=xx[:stop],yy[:stop],q[:stop],hit[:stop]
    # A retreat cannot silently delete an unreconstructed part of the original support.
    if a['retreat']>0 and max(abs(xx-x0))+1<rebuild:return None
    # Evidence inside the rebuilt old support cannot certify a new, blind extension.
    beyond=(xx>source['x'][-1]+1e-8) if direction==1 else (xx<source['x'][0]-1e-8)
    reliable=bool(not exploratory and beyond.sum()>=2 and
                  hit.mean()>=cfg['corridor_hit_fraction'] and
                  hit[beyond].mean()>=cfg['corridor_hit_fraction'])
    if not exploratory and not reliable:return None
    retained=a['path'];kind=np.where((xx>=source['x'][0])&(xx<=source['x'][-1]),1,2)
    event=dict(type='Q_continuation',interval=[float(xx.min()),float(xx.max())],anchor=[a['x'],a['y']],
        direction=direction,retreat=a['retreat'],degree=a['degree'],windows=a['windows'],
        heldout_error=a['held'],forecast_disagreement=a['spread'],slope=a['slope'],curvature=a['curvature'],
        initial_slope_error=float(abs(interpolator.derivative()(0)-direction*a['slope'])),
        reliable_candidate=reliable,exploratory=exploratory,observations_promoted=False)
    return pack_path(np.r_[retained['x'],xx],np.r_[retained['y'],yy],np.r_[retained['confidence'],q],
        np.r_[retained['observed'],np.zeros(len(xx),bool)],source['ids'],
        kind=np.r_[retained['kind'],kind],events=retained['events']+[event])


def continue_path(path,field,gray,valid,width,cfg,variant,exploratory=False):
    out=[];audit={}
    for right in (False,True):
        aa=anchors(path,right,width,cfg,variant);audit['right' if right else 'left']=len(aa)
        for a in aa:
            p=extend_from_anchor(path,a,field,gray,valid,width,cfg,exploratory)
            if p is not None:out.append(p);break
    return out,audit


def join_paths(paths,field,gray,valid,width,cfg,variant,inspect_all=False):
    out=[];audit={};sigma=noise(width,cfg)
    for first in paths:
        for second in paths:
            if set(first['ids'])&set(second['ids']):continue
            gap=second['x'][0]-first['x'][-1]
            if gap<=0 or gap>cfg['maximum_join_width']*width:continue
            left=anchors(first,True,width,cfg,variant);right=anchors(second,False,width,cfg,variant)
            candidates=[]
            for a in left:
                for b in right:
                    lo,hi=a['x'],b['x'];length=hi-lo
                    if length<=0 or length>cfg['maximum_join_width']*width:continue
                    # Each stable side must predict the other's level within a bounded corridor.
                    pa,_=prior(a,length,width,cfg);pb,_=prior(b,length,width,cfg)
                    if max(abs(float(pa)-b['y']),abs(float(pb)-a['y']))>max(4*sigma,.025*width):continue
                    x=np.arange(np.floor(lo)+1,np.ceil(hi),dtype=float)
                    if len(x)<2:continue
                    y,dy,ddy=hermite(x,lo,a['y'],a['slope'],hi,b['y'],b['slope'])
                    bend=width*abs(ddy)/(1+dy*dy)**1.5
                    if abs(dy).max()>1.5 or np.quantile(bend,.95)>12:continue
                    # A bounded image-evidence correction cannot alter anchor position/tangent.
                    offsets=np.linspace(-sigma,sigma,9);xx=np.repeat(x[:,None],len(offsets),axis=1)
                    yy=y[:,None]+offsets[None]
                    signal=.65*sample(field,xx.ravel(),yy.ravel()).reshape(yy.shape)+.35*sample(gray,xx.ravel(),yy.ravel()).reshape(yy.shape)
                    signal-=.08*(offsets[None]/sigma)**2
                    correction=gaussian_filter1d(offsets[signal.argmax(1)],max(1.,len(x)/24))
                    phase=(x-lo)/length;y+=correction*16*phase**2*(1-phase)**2
                    q=sample(field,x,y,max(1,round(width/1046.)));g=sample(gray,x,y,max(1,round(width/1046.)))
                    hit=(q>=.1)|(g>=.6);inside=sample(valid.astype(float),x,y)>.99
                    cuts=np.flatnonzero(np.diff(np.r_[False,~hit,False]));run=int(np.max(cuts[1::2]-cuts[::2],initial=0))
                    if not inside.all() or hit.mean()<cfg['corridor_hit_fraction'] or run>max(1,.02*width):continue
                    aa,bb=a['path'],b['path']
                    event=dict(type='Q_join',interval=[float(lo),float(hi)],retreats=[a['retreat'],b['retreat']],
                        anchors=[[a['x'],a['y']],[b['x'],b['y']]],windows=[a['windows'],b['windows']],
                        slopes=[a['slope'],b['slope']],degrees=[a['degree'],b['degree']],
                        reliable_candidate=True,observations_promoted=False)
                    p=pack_path(np.r_[aa['x'],x,bb['x']],np.r_[aa['y'],y,bb['y']],
                        np.r_[aa['confidence'],q,bb['confidence']],np.r_[aa['observed'],np.zeros(len(x),bool),bb['observed']],
                        aa['ids']+bb['ids'],kind=np.r_[aa['kind'],np.ones(len(x),int),bb['kind']],
                        events=aa['events']+bb['events']+[event])
                    score=a['quality']+b['quality']+5*(a['retreat']+b['retreat'])-.5*float(q.mean())
                    candidates.append((score,p))
            if candidates:
                ordered=sorted(candidates,key=lambda q:q[0])
                if inspect_all:out.extend(p for _,p in ordered)
                else:out.append(ordered[0][1])
    return out,audit
