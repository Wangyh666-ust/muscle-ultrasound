"""Round39 isolated inference-only geometry/aggregation. No reference-data reads."""
import copy, math
import numpy as np
from scripts.round38_core import (decode, base_measure, qualified, aggregate, to_original,
    weighted_median, scores, choose, legacy_measure)
from scripts.round38_common import ROOT, j, CFG
from scripts.round29_core import endpoint_line, add_extensions, value_at
from scripts.round35_decoder import identity_pool, map_curve
from scripts.round36_core import merge_pools, prepare_evidence, support, interpolated_y
from scripts.round25_boundary_fit import _design
from scripts.round38_speed import run_centres_per_column
from scripts.round27_experiment import boundary
from scripts.round29_core import boundary_pool
from src.umud_baseline.apo_baseline import load_frozen_rule

ARMS=('B0','V','C','VC','Q','E','W','S','WS','J')
BCFG=j(ROOT/'manifest/round29_boundary_spatial_v1.json')['boundary']
MCONF=j(ROOT/'configs/round36_candidate_consensus.json')

def domain(roi,w,h):
    x0,y0,x1,y1=roi['box'];mx=round(.03*(x1-x0));my=round(.03*(y1-y0))
    return [max(0,x0-mx),max(0,y0-my),min(w,x1+mx),min(h,y1+my)]

def masked(masks,roi):
    h,w=masks.shape[1:];x0,y0,x1,y1=domain(roi,w,h);out=np.zeros_like(masks)
    out[:,y0:y1,x0:x1]=masks[:,y0:y1,x0:x1]
    return out

def old_pool(masks):
    rule,_=load_frozen_rule();_,b=boundary(masks[0],rule,masks.shape[1])
    pools={'D0':boundary_pool(masks[0],b,BCFG),'D1':identity_pool(masks,BCFG)}
    row=dict(height=masks.shape[1],pools={k:v['pairs'] for k,v in pools.items()},
        scores={k:{'scores':[dict(pair_id=p['pair_id'],quality=p['quality']) for p in v['pairs']]} for k,v in pools.items()})
    out=merge_pools(row,MCONF['merge'])
    for p in out:p['observations']=pools[p['source_decoder']]['pairs'][p['source_pair_id']]['points']
    return out

def coverage(xs,pitch):
    """Union length of native observed column cells in canonical coordinates."""
    x=np.unique(xs)
    return float(pitch+np.minimum(np.diff(x),pitch).sum()) if len(x) else 0.

def tracks_from_observations(columns,pitch,tolerance=8.,maxgap=3.):
    ts=[]
    for x,ys in columns:
        choices=[]
        for i,t in enumerate(ts):
            lx,ly=t['points'][-1];gap=x-lx
            # Adjacent observed pixel cells are never a missing-support gap,
            # even when one native pixel spans >3 canonical units.
            if gap<=0 or gap-pitch>maxgap-1+1e-8:continue
            slope=0.
            if len(t['points'])>1:
                px,py=t['points'][-2];slope=(ly-py)/(lx-px)
            for k,y in enumerate(ys):
                err=abs(y-ly-slope*gap)
                if err<=tolerance*max(1,gap)+1e-8:choices.append((err,i,k))
        used=set();taken=set()
        for _,i,k in sorted(choices):
            if i in used or k in taken:continue
            ts[i]['points'].append((x,ys[k]));used.add(i);taken.add(k)
        for k,y in enumerate(ys):
            if k not in taken:ts.append({'points':[(x,y)]})
    return ts

def merge_normalized(raw,w,h,pitch):
    ts=[dict(points=list(t['points']),source_ids=[i]) for i,t in enumerate(raw)]
    while True:
        options=[]
        for i,a in enumerate(ts):
            if len(a['points'])<2 or coverage([p[0] for p in a['points']],pitch)<5:continue
            la=endpoint_line(a['points'],True)
            for k,b in enumerate(ts):
                if i==k or len(b['points'])<2 or coverage([p[0] for p in b['points']],pitch)<5:continue
                gap=b['points'][0][0]-a['points'][-1][0]
                if not pitch+1e-8<gap<=max(3,.02*w)+1e-8:continue
                lb=endpoint_line(b['points'],False)
                angle=abs(math.degrees(math.atan(la[0])-math.atan(lb[0])))
                err=max(abs(np.polyval(la,b['points'][0][0])-b['points'][0][1]),abs(np.polyval(lb,a['points'][-1][0])-a['points'][-1][1]))
                if angle<=10 and err<=max(3,.01*h):options.append((err,gap,i,k))
        if not options:break
        _,_,i,k=min(options);ts[i]=dict(points=ts[i]['points']+ts[k]['points'],source_ids=ts[i]['source_ids']+ts[k]['source_ids']);ts.pop(k)
    return ts

def fit_float(points,h,w):
    """Same four-piece OLS estimator, without integer-coordinate truncation."""
    a=np.asarray(points,float);lo,hi=a[0,0],a[-1,0]
    if hi<=lo:return None
    A,u=_design(a[:,0],lo,hi)
    if A is None or np.linalg.matrix_rank(A)<5:return None
    cv=dict(support=[lo,hi],paired_x=a[:,0].tolist(),max_sample_gap=max(2,.015*w),knots_u=[0,.25,.5,.75,1])
    for col,s in [(1,'sup'),(2,'deep')]:
        knots=np.linalg.lstsq(A,a[:,col],rcond=None)[0];res=abs(A@knots-a[:,col]);p90=float(np.quantile(res,.9))
        if p90>max(3,.02*h) or not np.isfinite(knots).all():return None
        cv[s]=dict(knots_u=cv['knots_u'],knots_x=np.linspace(lo,hi,5).tolist(),knots_y=knots.tolist(),support=[lo,hi],paired_x=cv['paired_x'],max_sample_gap=cv['max_sample_gap'],image_shape=[h,w],residual_p90_px=p90,rank=5,n_observations=len(a))
    if np.any(np.array(cv['sup']['knots_y'])>=cv['deep']['knots_y']):return None
    return add_extensions(cv,points,BCFG)

def baseline_normalized(columns,raw,w,h,pitch):
    pts=[];ambiguous=[];occupied=[]
    for x,ys in columns:
        if ys:occupied.append(x)
        if len(ys)==2 and ys[1]-ys[0]>max(2,.005*h):pts.append((x,*ys))
        if len(ys)>2:ambiguous.append(x)
    if coverage(ambiguous,pitch)/max(coverage(occupied,pitch),1e-12)>.15:
        usable=[dict(t['points']) for t in raw if coverage([p[0] for p in t['points']],pitch)>=max(5,.05*w)]
        pairs=[]
        for i,a in enumerate(usable):
            for b in usable[i+1:]:
                xs=sorted(a.keys()&b.keys());cov=coverage(xs,pitch)
                if cov>=max(20,.25*w):pairs.append((cov,abs(float(np.median([abs(a[x]-b[x]) for x in xs]))-.358125*h),a,b,xs))
        if not pairs:return []
        near=[p for p in pairs if p[0]>=.9*max(q[0] for q in pairs)];_,_,a,b,xs=min(near,key=lambda p:p[1])
        if np.median(list(a.values()))>np.median(list(b.values())):a,b=b,a
        pts=[(x,a[x],b[x]) for x in xs if b[x]-a[x]>max(2,.005*h)]
    if not pts or coverage([p[0] for p in pts],pitch)<max(8,.3*w) or pts[-1][0]-pts[0][0]<.3*w:return []
    arr=np.array(pts);fits=[np.polyfit(arr[:,0],arr[:,c],1) for c in (1,2)]
    if any(np.quantile(abs(np.polyval(f,arr[:,0])-arr[:,c]),.9)>max(3,.02*h) for f,c in zip(fits,(1,2))):return []
    if np.any(np.polyval(fits[1]-fits[0],[pts[0][0],pts[-1][0]])<=0):return []
    sample=np.linspace(pts[0][0],pts[-1][0],5)[1:-1]
    if any(np.min(abs(arr[:,0]-x))>max(2,.015*w) for x in sample):return []
    if any(np.any((np.polyval(f,sample)<0)|(np.polyval(f,sample)>=h)) for f in fits):return []
    return pts

def normalized_pool_from_columns(columns,roi,pitch_native=1.):
    x0,y0,x1,y1=roi['box'];factor=1046/max(x1-x0,1);w=1046.;h=(y1-y0)*factor;pitch=pitch_native*factor
    cols=[[(float((x-x0)*factor),[(y-y0)*factor for y in ys]) for x,ys in ch] for ch in columns]
    raw=[tracks_from_observations(ch,pitch,5 if i==0 else 8) for i,ch in enumerate(cols)]
    # D0 pool tracking uses tol8; conservative baseline retains inherited tol5.
    union_raw=tracks_from_observations(cols[0],pitch,8)
    ts=[merge_normalized(rr,w,h,pitch) for rr in (union_raw,raw[1],raw[2])]
    maps=[[dict(t['points']) for t in tt if coverage([p[0] for p in t['points']],pitch)>=max(5,.05*w)] for tt in ts]
    pools={'D0':[],'D1':[]}
    def append(pts,d):
        if not pts:return
        if any(p['points']==pts for p in pools[d]):return
        cv=fit_float(pts,h,w)
        inside=[p[0] for p in pts if any(p[0] in m and p[1]+3<m[p[0]]<p[2]-3 for m in (maps[0] if d=='D0' else maps[1]+maps[2]))]
        qual=(coverage([p[0] for p in pts],pitch)-coverage(inside,pitch))/w
        pools[d].append(dict(pair_id=len(pools[d]),curve=cv,quality=qual,points=pts))
    append(baseline_normalized(cols[0],raw[0],w,h,pitch),'D0')
    for d,pairs in [('D0',((a,b) for i,a in enumerate(maps[0]) for b in maps[0][i+1:])),('D1',((a,b) for a in maps[1] for b in maps[2]))]:
        for a,b in pairs:
            xs=sorted(a.keys()&b.keys())
            if not xs or coverage(xs,pitch)<max(8,.3*w) or xs[-1]-xs[0]<.3*w:continue
            if d=='D0' and np.median([a[x]-b[x] for x in xs])>0:a,b=b,a
            pts=[(x,a[x],b[x]) for x in xs if b[x]-a[x]>max(2,.005*h)]
            if d=='D1' and len(pts)!=len(xs):continue
            if coverage([p[0] for p in pts],pitch)>=max(8,.3*w):append(pts,d)
    row=dict(height=h,pools=pools,scores={d:{'scores':[dict(pair_id=p['pair_id'],quality=p['quality']) for p in ps]} for d,ps in pools.items()})
    merged=merge_pools(row,MCONF['merge']);t=dict(sx=1/factor,sy=1/factor,tx=x0,ty=y0)
    for p in merged:
        pts=pools[p['source_decoder']][p['source_pair_id']]['points'];p['observations']=[(x/factor+x0,a/factor+y0,b/factor+y0) for x,a,b in pts]
        p['curve']=map_curve(p['curve'],t)
        p['ranking_height']=y1-y0
    return merged

def normalized_pool(masks,roi):
    # Same native run observations, no probability or binary-mask interpolation.
    cols=[list(enumerate(run_centres_per_column(m))) for m in masks]
    return normalized_pool_from_columns(cols,roi)

def fixed_extensions(pool,roi):
    out=copy.deepcopy(pool);span=roi['box'][2]-roi['box'][0]
    for p in out:
        a=np.asarray(p['observations']);cv=p['curve'];p['extension_diagnostics']=[]
        for col,s in [(1,'sup'),(2,'deep')]:
            for right,name in [(False,'left_ray'),(True,'right_ray')]:
                x=a[-1 if right else 0,0];keep=a[:,0]>=x-.25*span if right else a[:,0]<=x+.25*span
                pts=a[keep];short=len(pts)<2
                if short:pts=a[-2:] if right else a[:2]
                slope=float(np.polyfit(pts[:,0],pts[:,col],1)[0]);at=-1 if right else 0;y=cv[s]['knots_y'][at]
                cv[s][name]=[slope,float(y-slope*x)];p['extension_diagnostics'].append(dict(side=s,ray=name,observed_points=len(pts),insufficient_window=short,window_px=.25*span))
    return out

def eligibility(pool,st,row):
    # label_id belongs to the UNFILTERED original component labeling. Relabeling
    # a filtered mask changes IDs and silently attaches the wrong pixels.
    ev=prepare_evidence(st['raw_mask'],st['segments'],row['width'],row['height'],MCONF['c1']);result=[]
    yy,xx=np.nonzero(st['evidence'])
    assert np.array_equal(xx,ev['xs']) and np.array_equal(yy,ev['ys']),'C1 component IDs do not match evidence pixels'
    box=domain(st['roi'],row['width'],row['height']);lo=min((p['curve']['support'][0] for p in pool),default=0);hi=max((p['curve']['support'][1] for p in pool),default=1)
    from scripts.round28_core import region
    for p in pool:
        cv=p['curve'];xs=np.array(cv['paired_x']);reasons=[];su=interpolated_y(cv['sup'],xs);de=interpolated_y(cv['deep'],xs)
        if np.any(su>=de):reasons.append('unordered')
        if not np.all((xs>=box[0])&(xs<box[2])&(su>=box[1])&(de<box[3])):reasons.append('outside_image_valid_domain')
        span=st['roi']['box'][2]-st['roi']['box'][0]
        if coverage(xs,1.)<.3*span:reasons.append('insufficient_observed_support')
        fractions=[[],[],[]];records=[]
        for s,g,wt in ev['tracks']:
            b=region(s['centroid'][0],[lo,hi])
            if b is None:continue
            xx,yy=ev['pixels'][s['label_id']];inside=support(cv,xx)&(yy>=interpolated_y(cv['sup'],xx))&(yy<=interpolated_y(cv['deep'],xx))
            frac=float(inside.mean()) if len(inside) else 0.;fractions[b].append((frac,wt));records.append(dict(id=s['label_id'],inside=frac,inside80=frac>=.8,region=b))
        active=[b for b in range(3) if fractions[b]];rw=[.25,.5,.25];den=sum(rw[b] for b in active)
        c=sum(rw[b]*np.average([v for v,w in fractions[b]],weights=[w for v,w in fractions[b]]) for b in active)/den if den else None
        applied=len(records)>=3 and len(active)>=2
        if applied and c<.6:reasons.append('confident_C1_contradiction')
        result.append(dict(pair_id=p['pair_id'],allowed=not reasons,reasons=reasons,C1_gate=applied,C1_containment=c,tracks=records,
            basis='C1_and_boundary' if applied else 'boundary_only_C1_insufficient',independent_tracks=len(records)))
    return result

def measure_pool(pool,st,row,cfg,gates=None):
    sc=scores(pool,st['evidence']);allowed={q['pair_id'] for q in gates if q['allowed']} if gates is not None else None
    ranking_row={**row,'height':pool[0].get('ranking_height',row['height'])} if pool else row
    picked=choose(pool,[s for s in sc if allowed is None or s['pair_id'] in allowed],ranking_row)
    base,cs=base_measure([picked] if picked else [],st,row);base['candidate_scores']=sc
    out=aggregate(qualified(base,cs),cs,row,cfg);out['pair_eligibility']=gates
    return out,cs

def fl_inputs(out,cfg,continuous=False):
    ag=out['aggregation'];ids=ag['metrics']['FL']['contributing_ids'] if ag else []
    by={c['segment']['label_id']:c for c in out['candidates']};lo,_,hi,_=out['roi']['box'];span=max(hi-lo,1);ac=cfg['aggregate']
    xx=np.array([(by[i]['segment']['centroid'][0]-lo)/span for i in ids]);quality=np.array([ag['quality'][str(i)] for i in ids])
    vals=np.array([by[i]['raw_FL'] for i in ids]);bins=np.minimum((xx*3).astype(int),2);active=sorted(set(bins.tolist()))
    kernel=np.exp(-.5*((xx[:,None]-np.array(ac['anchors'])[None,:])/ac['spatial_sigma'])**2)
    if not continuous:kernel[:,[i for i in range(3) if i not in active]]=0
    if len(ids):kernel/=kernel.sum(1)[:,None]
    return ids,vals,quality,kernel,active,span

def robust(vals,w,floor,ac,fixed_scale=None):
    ix=w>0;out=np.zeros(len(w))
    if not ix.any():return out,None,None
    x=np.asarray(vals)[ix];v=np.asarray(w)[ix];v=v/v.sum();center=weighted_median(x,v)
    scale=max(floor,1.4826*weighted_median(abs(x-center),v)) if fixed_scale is None else max(floor,fixed_scale)
    final=v.copy()
    for _ in range(ac['iterations']):
        factor=np.minimum(1.,ac['huber_delta']*scale/np.maximum(abs(x-center),1e-12));final=v*factor;final/=final.sum();new=float(final@x)
        if abs(new-center)<1e-10:break
        center=new
    out[ix]=final;return out,float(final@x),scale

def combine_fl(vals,quality,kernel,active,span,ac,cal,use_w,use_s):
    mass=(quality[:,None]*kernel).sum(0);rw=np.array(ac['region_weights'],float)
    if use_w:rw*=mass/(mass+cal['lambda_evidence'])
    else:rw[[i for i in range(3) if i not in active]]=0
    if rw.sum()<=0:return np.zeros(len(vals)),[]
    rw/=rw.sum();final=np.zeros(len(vals));regions=[]
    for b in range(3):
        if rw[b]<=0:continue
        ww,v,sc=robust(vals,quality*kernel[:,b],ac['scale_floor_FL_roi']*span,ac,cal['FL_scale_roi']*span if use_s else None)
        final+=rw[b]*ww;regions.append(dict(region=['left','middle','right'][b],value=v,scale=sc,mass=float(mass[b]),region_weight=float(rw[b]),independent_evidence=b in active))
    return final,regions

def modify_fl(out,cfg,cal,use_w=False,use_s=False):
    q=copy.deepcopy(out)
    if q['curve'] is None or not q['aggregation']:return q
    ids,vals,qual,kernel,active,span=fl_inputs(q,cfg,use_w)
    if not ids:return q
    final,regions=combine_fl(vals,qual,kernel,active,span,cfg['aggregate'],cal,use_w,use_s)
    q['values']['FL']=float(final@vals);q['FL_weights']={str(i):float(w) for i,w in zip(ids,final)};q['selection']['value_px']=q['values']['FL']
    m=q['aggregation']['metrics']['FL'];m.update(regions=regions,dispersion=float(np.sqrt(final@((vals-final@vals)**2))),effective_tracks=float(1/np.sum(final**2)),continuous_evidence_weights=use_w,fixed_robust_scale=use_s)
    return q

def suppress(out,explore):
    q=copy.deepcopy(out);q['exploratory_values']=copy.deepcopy(explore['values']);q['exploratory_weights']={m:explore[m+'_weights'] for m in ('PA','FL')}
    # Keep candidates for exploratory inverse mapping; normal curve/weights are absent.
    q['candidates']=copy.deepcopy(explore['candidates']);q['rejected_curve']=explore['curve']
    return q
