"""Round31 selection-only control. No reference or expert inputs."""
from fractions import Fraction
import hashlib,json
import numpy as np
from scipy import ndimage
from scripts.fascicle_geometry import extract_segments
from scripts.round29_core import value_at


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def actual_evidence(mask, expected_segments=None):
    """Use component pixels, never rasterized fitted segments or extrapolation."""
    extraction=extract_segments(mask)
    if expected_segments is not None:
        from scripts.round27_experiment import check_equal
        check_equal(extraction['segments'],expected_segments,'C1 segment replay')
    labels,_=ndimage.label(mask,structure=np.ones((3,3),bool))
    ids=[s['label_id'] for s in extraction['segments']]
    evidence=np.isin(labels,ids) if ids else np.zeros_like(mask,dtype=bool)
    return evidence,extraction


def original_choice(pairs,cfg,height):
    best=max(p['quality'] for p in pairs)
    near=[p for p in pairs if p['quality']>=best*(1-cfg['quality_near_fraction'])]
    return min(near,key=lambda p:(abs(p['spacing_px']-cfg['spacing_prior_height_fraction']*height),-p['quality'],p['pair_id']))['pair_id']


def score_pairs(pool,evidence,cfg):
    h,w=evidence.shape
    all_ids=[p['pair_id'] for p in pool['pairs']]
    if len(set(all_ids))!=len(all_ids):raise ValueError('duplicate pair identity')
    valid=[p for p in pool['pairs'] if p['curve'] is not None]
    original=pool['selected_pair_id']
    if valid and original!=original_choice(valid,cfg,h):raise ValueError('original pair ordering replay mismatch')
    if not valid and original is not None:raise ValueError('selected pair absent from valid pool')
    result={'R':original,'S':original,'pair_pool_sha256':canonical_hash(pool['pairs']),
            'evaluable_pixel_count':int(evidence.sum()),'regions':None,'scores':[],
            'fallback':None,'tie_ids':[],'changed':False}
    if not valid:
        result['fallback']='no_valid_pair';return result
    lo=min(p['support'][0] for p in valid);hi=max(p['support'][1] for p in valid)
    # Integer pixel centres over [lo,hi], partitioned into three equal bins.
    edges=np.linspace(lo,hi+1,4);ys,xs=np.nonzero(evidence)
    keep=(xs>=lo)&(xs<=hi);xs=xs[keep];ys=ys[keep]
    bins=np.minimum(np.searchsorted(edges[1:],xs,side='right'),2)
    den=[int(np.sum(bins==b)) for b in range(3)];active=[b for b in range(3) if den[b]]
    result.update(regions={'edges':edges.tolist(),'denominators':den,'active':active,
                          'weights':[1/len(active) if b in active else 0 for b in range(3)]},
                  outside_region_pixels=int(evidence.sum())-len(xs))
    ranked=[]
    for p in valid:
        cv=p['curve'];px=np.asarray(cv['paired_x'],float)
        ux,inv=np.unique(xs,return_inverse=True)
        up=np.array([value_at(cv['sup'],x) for x in ux]);dn=np.array([value_at(cv['deep'],x) for x in ux])
        # Bound memory while comparing actual observations to support columns.
        ix=np.searchsorted(px,ux);left=px[np.clip(ix-1,0,len(px)-1)];right=px[np.clip(ix,0,len(px)-1)]
        supported=(ux>=cv['support'][0])&(ux<=cv['support'][1])&(np.minimum(abs(ux-left),abs(ux-right))<=cv['max_sample_gap'])
        inside=supported[inv]&(ys>=up[inv])&(ys<=dn[inv])&(up[inv]<dn[inv])
        num=[int(np.sum(inside&(bins==b))) for b in range(3)]
        containment=sum((Fraction(num[b],den[b]) for b in active),Fraction())/len(active) if active else Fraction()
        quality=Fraction(p['common_columns']-p['interior_columns'],w)
        if abs(float(quality)-p['quality'])>1e-12:raise ValueError('quality definition changed')
        score=containment*quality
        rec={'pair_id':p['pair_id'],'numerators':num,'denominators':den,
             'containment':float(containment),'quality':float(quality),'score':float(score),
             'score_fraction':[score.numerator,score.denominator],'inside_pixels':int(inside.sum())}
        result['scores'].append(rec);ranked.append((score,p))
    if not active:result['fallback']='no_evaluable_pixels_in_fixed_regions'
    elif max(s for s,p in ranked)==0:result['fallback']='all_pair_scores_zero'
    else:
        top=max(s for s,p in ranked);tied=[p for s,p in ranked if s==top]
        result['tie_ids']=[p['pair_id'] for p in tied]
        result['S']=original if original in result['tie_ids'] else original_choice(tied,cfg,h)
        if len(tied)>1:result['tie_resolution']='baseline_if_tied_else_original_order_within_top'
    result['changed']=result['S']!=original
    return result
