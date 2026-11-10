"""Isolated Round37 prediction rules. No reference labels, training or default edits."""
import copy, itertools, math
import cv2
import numpy as np
from scipy import ndimage
from scripts.fascicle_geometry import extract_segments
from scripts.round28_core import select, accepted, region, REGIONS
from scripts.round29_core import extended_fl, value_at, line_at
from scripts.round36_core import measure as old_measure, axis_delta, support

_GEOM_CACHE={}
_J_CACHE={}
def begin_row():
    _GEOM_CACHE.clear();_J_CACHE.clear()
def segment_key(s):return (s['label_id'],s['direction_deg'],*s['centroid'],*s['p0'],*s['p1'])


def image_roi(gray):
    """Image-only texture field. Canonical search after removing outer empty canvas."""
    h,w=gray.shape; yy,xx=np.nonzero(gray>8)
    if len(xx)<.01*gray.size:return dict(box=[0,0,w,h],fallback=True,reason='insufficient_image_content')
    x0,x1,y0,y1=int(xx.min()),int(xx.max()+1),int(yy.min()),int(yy.max()+1)
    crop=gray[y0:y1,x0:x1];ch,cw=crop.shape;scale=512/max(ch,cw)
    a=cv2.resize(crop,(max(1,round(cw*scale)),max(1,round(ch*scale))),interpolation=cv2.INTER_AREA).astype(np.float32)
    mean=cv2.blur(a,(11,11));std=np.sqrt(np.maximum(0,cv2.blur(a*a,(11,11))-mean*mean))
    mask=cv2.morphologyEx(((mean>12)&(std>6)).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n<2:return dict(box=[0,0,w,h],fallback=True,reason='no_texture_component')
    q=stats[1+np.argmax(stats[1:,cv2.CC_STAT_AREA])];x,y,rw,rh,area=map(int,q)
    if area<.05*a.size or rw<.2*a.shape[1] or rh<.2*a.shape[0]:
        return dict(box=[0,0,w,h],fallback=True,reason='small_or_narrow_texture')
    # Pixel-edge mapping; conservative outward rounding. No model masks enter ROI.
    box=[max(0,x0+math.floor(x*cw/a.shape[1])),max(0,y0+math.floor(y*ch/a.shape[0])),
         min(w,x0+math.ceil((x+rw)*cw/a.shape[1])),min(h,y0+math.ceil((y+rh)*ch/a.shape[0]))]
    return dict(box=box,fallback=False,reason='largest_textured_field',content_box=[x0,y0,x1,y1])


def normalized_crop(image,roi,margin=.03):
    h,w=image.shape[:2]
    if roi['fallback']:return image.copy(),[0,0,w,h]
    x0,y0,x1,y1=roi['box'];mx=round((x1-x0)*margin);my=round((y1-y0)*margin)
    box=[x0-mx,y0-my,x1+mx,y1+my];a,b,c,d=box
    out=np.zeros((d-b,c-a,3),np.uint8);l,t,r,bt=max(0,a),max(0,b),min(w,c),min(h,d)
    out[t-b:bt-b,l-a:r-a]=image[t:bt,l:r]
    return out,box


def raw_components(mask):
    """Same frozen PCA observations; no skeleton edits or direction replacement."""
    ex=extract_segments(mask)
    return ex,sorted(ex['segments']+ex['rejected'],key=lambda s:s['label_id'])


def gate(s,roi,cfg,soft=False):
    ref=cfg['filter']['reference_roi_width'];width=roi['box'][2]-roi['box'][0];factor=ref/max(width,1)
    L=s['length_px']*factor;W=s['width_px']*factor;A=s['area_px']*factor*factor
    # All dimensional floors are now in the same reference coordinate system.
    elong=L/max(W,1.);flags=[]
    if A<20-1e-9:flags.append('area_lt_20_reference')
    if L<30-1e-9:flags.append('length_lt_30_reference')
    if s['linearity']<.9:flags.append('low_linearity')
    if elong<3:flags.append('low_elongation')
    quality=1.;rescued=False
    if W>8+1e-9:
        if soft and W<=16 and s['linearity']>=.985 and elong>=6:
            quality=math.exp(-((W/8-1)/.75)**2);rescued=True
        else:flags.append('wide_or_branch_suspected')
    return not flags,dict(reference_length=L,reference_width=W,reference_area=A,reference_elongation=elong,
                         flags=flags,quality=quality,soft_width_candidate=rescued,scale_factor=factor)


def filter_components(raw,roi,cfg,mode):
    selected=[];audit=[]
    for s in raw:
        if mode=='old':ok=s['category']=='evaluable';ev={'quality':1.,'flags':[] if ok else [s['category']]}
        else:ok,ev=gate(s,roi,cfg,mode=='soft')
        t=copy.deepcopy(s);t['filter_quality']=ev['quality'];t['filter_evidence']=ev
        audit.append({'id':s['label_id'],'accepted':ok,**ev})
        if ok:selected.append(t)
    return selected,audit


def duplicate(a,b,span,cfg):
    if abs(axis_delta(a['direction_deg'],b['direction_deg']))>cfg['spatial']['duplicate_angle_deg']:return False
    delta=np.subtract(a['centroid'],b['centroid']);perps=[];gaps=[]
    for s in (a,b):
        th=math.radians(s['direction_deg']);u=np.array([math.cos(th),math.sin(th)]);v=np.array([-u[1],u[0]])
        perps.append(abs(delta@v))
        aa=sorted(np.dot(p,u) for p in (a['p0'],a['p1']));bb=sorted(np.dot(p,u) for p in (b['p0'],b['p1']))
        gaps.append(max(0.,aa[0]-bb[1],bb[0]-aa[1]))
    # Must be both collinear and adjacent/overlapping; angle alone never merges.
    return max(perps)<=cfg['spatial']['duplicate_perp_fraction']*span and max(gaps)<=cfg['spatial']['duplicate_gap_fraction']*span


def sensitivity(seg,curve,row,roi,cfg):
    center=extended_fl(seg,curve,row['width'],row['height']);base=center['fl_px_raw'];vals=[];lost=0
    step=cfg['sensitivity']['endpoint_reference_pixels']*(roi['box'][2]-roi['box'][0])/cfg['filter']['reference_roi_width']
    probes=[]
    for da in cfg['sensitivity']['angle_offsets']:
        s=copy.deepcopy(seg);s['direction_deg']=(s['direction_deg']+da)%180;probes.append(s)
    # Normal translation of the observed segment: both endpoints and centroid move.
    th=math.radians(seg['direction_deg']);normal=np.array([-math.sin(th),math.cos(th)])
    for sign in (-1,1):
        s=copy.deepcopy(seg)
        for k in ('p0','p1','centroid'):s[k]=(np.array(s[k])+sign*step*normal).tolist()
        probes.append(s)
    for s in probes:
        f=extended_fl(s,curve,row['width'],row['height']);ok,_=accepted({'F0':f},'U')
        if ok:vals.append(f['fl_px_raw'])
        else:lost+=1
    rel=max([abs(v-base)/max(abs(base),1.) for v in vals],default=0.) if base is not None else 1.
    risk=min(1.,rel)+lost/len(probes)
    angles={}
    if curve:
        for side in ('sup','deep'):
            p=center.get('intersection_'+side);x=p[0] if p else seg['centroid'][0]
            m,_=line_at(curve[side],x);angles[side]=float(abs(axis_delta(seg['direction_deg'],math.degrees(math.atan(m)))))
    return dict(relative_max_change=rel,lost_probes=lost,n_probes=len(probes),risk=risk,
                flagged=bool(lost or rel>cfg['sensitivity']['flag_relative_change']),intersection_angles=angles)


def select_new(cs,curve,roi,cfg,spatial=False,soft=False,jitter=False,row=None):
    base=select(cs,curve['support'] if curve else None,'UW');jdiag={}
    if not(spatial or soft or jitter):return base,jdiag
    if curve is None:return base,jdiag
    lo,_,hi,_=roi['box'];span=max(hi-lo,1);bounds=[lo,hi] if spatial else curve['support'];targets=[lo+v*span for v in (1/6,.5,5/6)]
    candidates=[c for c in cs if accepted(c,'U')[0] and region(c['segment']['centroid'][0],bounds) is not None]
    if spatial:
        candidates=[c for c in candidates if not c['invalid'] and support(curve,np.array([c['segment']['centroid'][0]]))[0]]
    if jitter:
        for c in candidates:
            sk=(id(curve),segment_key(c['segment']),roi['box'][2]-roi['box'][0])
            if sk not in _J_CACHE:_J_CACHE[sk]=sensitivity(c['segment'],curve,row,roi,cfg)
            jdiag[str(c['label_id'])]=_J_CACHE[sk]
    def cost(c,b):
        q=c['F0']['extrapolation_ratio']
        if soft:q+=cfg['filter']['quality_penalty']*(1-c['segment'].get('filter_quality',1.))
        if jitter:q+=cfg['sensitivity']['score_weight']*jdiag[str(c['label_id'])]['risk']
        if spatial:q+=cfg['spatial']['target_distance_weight']*abs(c['segment']['centroid'][0]-targets[b])/span
        return q
    buckets=[[c for c in candidates if region(c['segment']['centroid'][0],bounds)==b] for b in range(3)]
    chosen=[]
    if spatial:
        # Exact joint choice over eligible fragments; no greedy bin-order preference.
        best=None
        for comb in itertools.product(*[[None]+b for b in buckets]):
            picked=[c for c in comb if c is not None]
            if any(abs(a['segment']['centroid'][0]-b['segment']['centroid'][0])<cfg['spatial']['min_separation_fraction']*span or duplicate(a['segment'],b['segment'],span,cfg) for a,b in itertools.combinations(picked,2)):continue
            rank=(-sum((.25,.5,.25)[b] for b,c in enumerate(comb) if c is not None),
                  sum(cost(c,b) for b,c in enumerate(comb) if c is not None),
                  -sum(c['F0']['visible_length_px'] for c in picked),tuple(c['label_id'] if c is not None else -1 for c in comb))
            if best is None or rank<best:best=rank;chosen=picked
    else:
        for b,pool in enumerate(buckets):
            if pool:chosen.append(min(pool,key=lambda c:(cost(c,b),-c['F0']['visible_length_px'],c['label_id'])))
    result=select(chosen,bounds,'UW')
    result.update(eligible_ids=[c['label_id'] for c in candidates],variant='Round37',
                  anchors_x=targets if spatial else None,selection_bounds=bounds,
                  original_weights=[(.25,.5,.25)[REGIONS.index(b)] for b in result['selected_regions']],
                  selection_costs={str(c['label_id']):cost(c,region(c['segment']['centroid'][0],bounds)) for c in candidates})
    xs=sorted(c['segment']['centroid'][0] for c in chosen)
    result['min_selected_separation_fraction']=min(np.diff(xs))/span if len(xs)>1 else None
    result['selected_sensitivity_flags']=[i for i in result['selected_ids'] if jdiag.get(str(i),{}).get('flagged')]
    return result,jdiag


def measure(curve,segments,row,pair_id,roi,cfg,arm):
    mk=(id(curve),tuple(segment_key(s) for s in segments))
    if mk not in _GEOM_CACHE:_GEOM_CACHE[mk]=old_measure(curve,segments,row,pair_id)
    out,cs=copy.deepcopy(_GEOM_CACHE[mk])
    by={s['label_id']:s for s in segments}
    for c in out['candidates']:c['segment']=by[c['segment']['label_id']]
    for c in cs:c['segment']=by[c['label_id']]
    spatial=arm in ('S','H','HI');soft=arm in ('F1','H','HI');jitter=arm in ('J','H','HI')
    selected,diag=select_new(cs,curve,roi,cfg,spatial,soft,jitter,row)
    out['selection']=selected;out['values']['FL']=selected['value_px']/(row.get('px_per_mm') or 1.) if selected['value_px'] is not None else None
    out['stage']='boundary_failure' if curve is None else 'output' if out['values']['FL'] is not None else 'no_FL'
    out['sensitivity']=diag
    out['stability_status']='missing' if out['values']['FL'] is None else 'flagged_exploratory' if selected.get('selected_sensitivity_flags') else 'not_flagged' if jitter else 'not_assessed'
    # Explicit per-line correspondence without changing frozen PA selection.
    for c,full in zip(out['candidates'],cs):
        c['PA']=full['PA'].get('pa_deg');c['intersection_sup']=full['F0'].get('intersection_sup');c['intersection_deep']=full['F0'].get('intersection_deep')
    return out
