"""Frozen role-aware decoder experiments. Accepts images/probabilities, never GT."""
import copy
import cv2,numpy as np
from scripts.pair_selection import run_centres_per_column,build_tracks
from scripts.round29_core import merge_tracks,add_extensions,endpoint_line,value_at
from scripts.round25_boundary_fit import fit_curve,fit_piecewise
from scripts.round31_core import score_pairs,original_choice
from scripts.round32_common import map_curve as base_map_curve

def map_curve(curve,t):
    if curve is None:return None
    out=base_map_curve(curve,t)
    for side in ('sup','deep'):
        for field in ('observed_x','reconstructed_x'):
            if field in curve[side]:out[side][field]=[x*t['sx']+t['tx'] for x in curve[side][field]]
    if 'observed_paired_x' in curve:out['observed_paired_x']=[x*t['sx']+t['tx'] for x in curve['observed_paired_x']]
    return out

def identity_pool(masks,cfg):
    h,w=masks.shape[1:];tracks=[];rawcount=0;merge_events=[]
    for role in (1,2):
        raw=build_tracks(run_centres_per_column(masks[role]),cfg['track_tol_px'],cfg['track_gap_columns'])
        ts,merges=merge_tracks(raw,w,h,cfg);rawcount+=len(raw);merge_events.append(merges)
        tracks.append([t for t in ts if len(t['points'])>=max(5,cfg['min_track_fraction']*w)])
    maps=[[dict(t['points']) for t in ts] for ts in tracks];pairs=[]
    for i,a in enumerate(maps[0]):
        for k,b in enumerate(maps[1]):
            xs=sorted(a.keys()&b.keys())
            if len(xs)<max(8,cfg['min_common_fraction']*w) or xs[-1]-xs[0]<cfg['min_common_fraction']*w:continue
            # Roles are never swapped to make a reversed prediction legal.
            if any(b[x]-a[x]<=max(2,.005*h) for x in xs):continue
            pts=[(x,a[x],b[x]) for x in xs];support=[xs[0],xs[-1]]
            cv,reason=fit_curve(pts,support,h,max(2,.015*w))
            inside=sum(any(x in m and a[x]+cfg['interior_track_margin_px']<m[x]<b[x]-cfg['interior_track_margin_px'] for m in maps[0]+maps[1]) for x in xs)
            pairs.append(dict(pair_id=len(pairs),source={'sup':i,'deep':k},points=pts,curve=add_extensions(cv,pts,cfg) if cv else None,
                failure=reason or None,support=support,common_columns=len(xs),interior_columns=inside,
                quality=(len(xs)-inside)/w,spacing_px=float(np.median([b[x]-a[x] for x in xs]))))
    valid=[p for p in pairs if p['curve']]
    return dict(pairs=pairs,selected_pair_id=original_choice(valid,cfg,h) if valid else None,original_track_count=rawcount,track_count=sum(map(len,tracks)),merges=merge_events)

def image_roi(gray,cfg):
    """Largest textured field, image-only; fallback explicitly covers full image."""
    a=gray.astype(np.float32);k=cfg['roi_window'];mean=cv2.blur(a,(k,k));std=np.sqrt(np.maximum(0,cv2.blur(a*a,(k,k))-mean*mean))
    mask=((mean>cfg['roi_intensity'])&(std>cfg['roi_texture_std'])).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n<2:return [0,0,gray.shape[1],gray.shape[0]],True
    q=stats[1+np.argmax(stats[1:,cv2.CC_STAT_AREA])];x,y,w,h,area=map(int,q)
    if area<.05*gray.size:return [0,0,gray.shape[1],gray.shape[0]],True
    return [x,y,x+w,y+h],False

def best_path(prob,union,cfg):
    """Second-order local Viterbi: step and change-of-slope penalties."""
    h,w=prob.shape;steps=np.arange(-cfg['max_step'],cfg['max_step']+1);d=len(steps)
    reward=np.log(np.clip(prob,1e-5,1-1e-5)/np.clip(1-prob,1e-5,1))+cfg['union_bonus']*union
    previous=np.zeros((h,d),np.float32);parent=np.full((w,h,d),d,np.uint8);best=(0.,0,0,0)
    for x in range(w):
        current=np.zeros_like(previous)
        for si,step in enumerate(steps):
            ys=np.arange(max(0,step),min(h,h+step));yp=ys-step
            choices=previous[yp]-cfg['turn_penalty']*np.abs(steps-step)[None,:]-cfg['step_penalty']*abs(step)
            pi=choices.argmax(1);v=choices[np.arange(len(ys)),pi]
            current[ys,si]=np.maximum(0,reward[ys,x]+np.maximum(v,0))
            parent[x,ys,si]=np.where(v>0,pi,d)
        yi,si=np.unravel_index(current.argmax(),current.shape);score=float(current[yi,si])
        if score>best[0]:best=(score,x,yi,si)
        previous=current
    score,x,y,s=best;points=[]
    if score<=0:return [],0.
    while x>=0:
        points.append((x,int(y)));p=int(parent[x,y,s])
        if p==d:break
        y-=steps[s];s=p;x-=1
    return points[::-1],score

def runs(mask):
    a=np.asarray(mask,bool);starts=np.flatnonzero(a & ~np.r_[False,a[:-1]]);ends=np.flatnonzero(a & ~np.r_[a[1:],False])
    return list(zip(starts.tolist(),ends.tolist()))

def paths(prob,union,cfg):
    work=prob.copy();out=[];h,w=prob.shape
    for _ in range(cfg['max_paths_per_side']):
        pts,score=best_path(work,union,cfg)
        if not pts:break
        x,y=np.asarray(pts).T
        for xx,yy in pts:work[max(0,yy-cfg['suppression_radius']):min(h,yy+cfg['suppression_radius']+1),xx]=1e-5
        obs=prob[y,x]>=cfg['observation_probability']
        # Long unsupported spans split paths rather than creating observations.
        cut=runs(~obs);cut=[(a,b) for a,b in cut if b-a+1>cfg['max_gap_fraction']*w]
        pieces=[];lo=0
        for a,b in cut:pieces.append((lo,a));lo=b+1
        pieces.append((lo,len(x)))
        for lo,hi in pieces:
            xx=x[lo:hi];yy=y[lo:hi];observed=obs[lo:hi]
            if len(xx)<cfg['min_path_span_fraction']*w or observed.sum()<cfg['min_observed_fraction']*w:continue
            a,b=np.flatnonzero(observed)[[0,-1]];xx,yy,observed=xx[a:b+1],yy[a:b+1],observed[a:b+1]
            out.append({'points':np.stack([xx,yy],1).tolist(),'observed':observed.tolist(),
                        'mean_probability':float(prob[yy,xx].mean()),'score':score})
    return out

def fit_path(p,height,cfg):
    xy=np.asarray(p['points']);obs=np.asarray(p['observed']);points=[(int(x),int(y),int(y)) for x,y in xy[obs]]
    support=xy[[0,-1],0].tolist();f,reason=fit_piecewise(points,support,height,'sup')
    if f is None:return None,reason
    f['max_sample_gap']=max(2,.015*512);f['observed_x']=xy[obs,0].tolist();f['paired_x']=xy[:,0].tolist()
    f['reconstructed_x']=xy[~obs,0].tolist()
    for right,name in ((False,'left_ray'),(True,'right_ray')):
        slope=float(endpoint_line(xy[obs],right,cfg['extension_fit_fraction'])[0]);at=-1 if right else 0
        f[name]=[slope,float(f['knots_y'][at]-slope*f['knots_x'][at])]
    return f,None

def probability_pool(probs,gray,evidence,dcfg,bcfg):
    roi,fallback=image_roi(gray,dcfg);x0,y0,x1,y1=roi;rw,rh=x1-x0,y1-y0
    factor=dcfg['canonical_width']/max(rw,rh);cw,ch=max(1,round(rw*factor)),max(1,round(rh*factor))
    ps=np.stack([cv2.resize(p[y0:y1,x0:x1],(cw,ch),interpolation=cv2.INTER_LINEAR) for p in probs])
    ev=cv2.resize(evidence[y0:y1,x0:x1].astype(np.uint8),(cw,ch),interpolation=cv2.INTER_NEAREST)>0
    rolepaths=[paths(ps[i],ps[0],dcfg) for i in (1,2)];pairs=[]
    fits=[[fit_path(p,ch,bcfg) for p in side] for side in rolepaths]
    for i,(su,sreason) in enumerate(fits[0]):
        for k,(de,dreason) in enumerate(fits[1]):
            if su is None or de is None:continue
            common=sorted(set(su['paired_x'])&set(de['paired_x']))
            if len(common)<dcfg['min_path_span_fraction']*cw:continue
            probes=sorted(set(common+su['knots_x']+de['knots_x']))
            if any(value_at(su,x)+max(2,dcfg['min_spacing_fraction']*ch)>=value_at(de,x) for x in probes):continue
            shared_observed=sorted(set(su['observed_x'])&set(de['observed_x']))
            cv={'sup':copy.deepcopy(su),'deep':copy.deepcopy(de),'support':[common[0],common[-1]],'paired_x':common,
                'max_sample_gap':max(2,.015*cw),'observed_paired_x':shared_observed,
                'support_semantics':'paired_x includes explicitly reconstructed path; observed_paired_x does not'}
            lo,hi=common[0],common[-1];fractions=[]
            for aa,bb in zip(np.linspace(lo,hi+1,4)[:-1],np.linspace(lo,hi+1,4)[1:]):
                yy,xx=np.where(ev);keep=(xx>=aa)&(xx<bb);xx,yy=xx[keep],yy[keep]
                if len(xx):fractions.append(float(np.mean([(value_at(su,x)<=y<=value_at(de,x)) for x,y in zip(xx,yy)])))
            contain=float(np.mean(fractions)) if fractions else 0.
            confidence=(rolepaths[0][i]['mean_probability']+rolepaths[1][k]['mean_probability'])/2
            obsfrac=(len(su['observed_x'])+len(de['observed_x']))/(len(su['paired_x'])+len(de['paired_x']))
            weight=dcfg['containment_weight'];score=((1-weight)*confidence+weight*contain)*obsfrac
            transform={'sx':rw/cw,'sy':rh/ch,'tx':x0+(rw/cw-1)/2,'ty':y0+(rh/ch-1)/2}
            original=map_curve(cv,transform)
            pairs.append(dict(pair_id=len(pairs),curve=original,source={'sup':i,'deep':k},failure=None,
                              score=score,containment=contain,confidence=confidence,observed_fraction=obsfrac))
    chosen=max(pairs,key=lambda p:(p['score'],-p['pair_id'])) if pairs else None
    return {'pairs':pairs,'selected_pair_id':chosen['pair_id'] if chosen else None,'roi':roi,'roi_fallback':fallback,
            'canonical_wh':[cw,ch],'track_count':sum(map(len,rolepaths)),'role_paths':rolepaths,
            'fit_failures':[[reason for _,reason in side] for side in fits]}
