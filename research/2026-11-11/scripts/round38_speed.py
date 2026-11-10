"""Process-local equivalent CPU operations; no model/rule/threshold changes."""
import numpy as np
from scripts.round29_core import value_at

def validity(seg,curve):
    if curve is None:return False,['no_apo_boundaries']
    xs=np.linspace(seg['p0'][0],seg['p1'][0],21);ys=np.linspace(seg['p0'][1],seg['p1'][1],21)
    inside=np.mean([value_at(curve['sup'],x)<=y<=value_at(curve['deep'],x) for x,y in zip(xs,ys)])
    # Same finite minimum, without Python iteration over every support pixel.
    px=np.asarray(curve['paired_x']);supported=np.mean(np.min(abs(xs[:,None]-px[None,:]),axis=1)<=curve['max_sample_gap'])
    reasons=[]
    if inside<.5:reasons.append('candidate_outside_band_region')
    if supported<.5:reasons.append('candidate_local_support_gap')
    return bool(reasons),reasons

def run_centres_per_column(mask):
    h,w=mask.shape;out=[];gap=max(2,round(h*.002))
    for x in range(w):
        ys=np.flatnonzero(mask[:,x])
        if not len(ys):out.append([]);continue
        splits=np.flatnonzero(np.diff(ys)>gap)+1;centres=[]
        for v in np.split(ys,splits):
            n=len(v);centres.append(float(v[n//2]) if n%2 else float((v[n//2-1]+v[n//2])/2))
        out.append(centres)
    return out

def install():
    import scripts.round29_core as r29
    import scripts.round35_decoder as r35
    import scripts.pair_selection as pair
    r29.validity=validity
    for module in (r29,r35,pair):module.run_centres_per_column=run_centres_per_column
