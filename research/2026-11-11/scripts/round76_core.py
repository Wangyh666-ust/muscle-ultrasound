"""Opt-in additive density/prior and support-aware contributor selection. No GT input."""
from copy import deepcopy
import math
import numpy as np
from scripts.round75_core import region_density

def rank(pairs,segments,bounds,density=None,prior=None):
    rows=[];width=max(np.ptp(bounds),1.)
    for p in pairs:
        dp=region_density(p,segments,bounds,p['interval']) if density and p['c1_active'] else None
        # Dimensionless visible line mass per normalized band area. No normalization
        # against the densest candidate, and the original containment term is retained.
        normalized=dp*width if dp is not None else None
        D=1-math.exp(-normalized/density['saturation']) if normalized is not None else 0.
        reward=density['maximum_reward']*D if density else 0.
        penalty=p['prior_penalty']
        base=p['score']
        if prior and p['MT_mm'] is not None:
            from scripts.round63_paths import path_quality
            # Recompute the existing score components directly: subtracting and
            # adding back an old penalty can create 2e-16 advantages between ties.
            base=.5*(path_quality(p['upper'],width)+path_quality(p['lower'],width))+.10*np.ptp(p['interval'])/width
            if p['c1_active']:base+=.20*(p['c1']-.5)
            distance=max(10-p['MT_mm'],p['MT_mm']-31,0.)
            penalty=prior['maximum_penalty']*(-math.expm1(-.5*(distance/prior['shoulder_mm'])**2))
            score=base+reward-penalty
        else:score=base+reward
        rows.append(dict(pair=p,score=float(score),base_score=p['score'],density=normalized,
                         density_reward=reward,old_prior=p['prior_penalty'],new_prior=penalty,
                         MT_mm=p['MT_mm'],containment=p['c1'],C1_active=p['c1_active']))
    rows.sort(key=lambda q:(-q['score'],-np.ptp(q['pair']['upper']['x'])-np.ptp(q['pair']['lower']['x']),q['pair']['upper']['ids'],q['pair']['lower']['ids']))
    return (deepcopy(rows[0]['pair']) if rows else None),rows

def signature(p):
    if p is None:return None
    return tuple(tuple(p[s]['ids']) for s in ('upper','lower'))

def select_tracks(tracks,cfg,span):
    """Keep sparse sets; for rich sets keep >=10 and every occupied spatial bin."""
    n=len(tracks)
    if n<=10:return [t['id'] for t in tracks],dict(reason='sparse_keep_all',before=n,after=n)
    count=max(10,math.ceil(cfg['retain_fraction']*n));count=min(count,n)
    ordered=sorted(tracks,key=lambda t:(t['distance'],str(t['id'])))
    cutoff=ordered[count-1]['distance']+cfg['tie_margin_roi']*span
    chosen={t['id'] for t in ordered if t['distance']<=cutoff}
    for b in sorted({t['region'] for t in tracks}):
        chosen.add(next(t['id'] for t in ordered if t['region']==b))
    ids=[t['id'] for t in tracks if t['id'] in chosen]
    assert len(ids)>=10 and {t['region'] for t in tracks if t['id'] in chosen}=={t['region'] for t in tracks}
    return ids,dict(reason='distance_selection',before=n,after=len(ids),target=count,cutoff_px=cutoff,
                    original_regions=sorted({t['region'] for t in tracks}),kept_regions=sorted({t['region'] for t in tracks if t['id'] in chosen}))

def adaptive(out,pair,row,cfg,cal,policy):
    from scripts.round39_core import fl_inputs,combine_fl
    from scripts.round29_core import extended_fl
    q=deepcopy(out)
    if not q['curve'] or not q['aggregation'] or not q['FL_weights']:return q,dict(reason='no_FL')
    ids,vals,quality,kernel,active,span=fl_inputs(q,cfg,True)
    risk=q['boundary_sensitivity']['tracks'];quality*=np.array([risk[str(i)]['risk_factor'] for i in ids])
    oldw,_=combine_fl(vals,quality,kernel,active,span,cfg['aggregate'],cal,True,True)
    scale=row['px_per_mm'] or 1.
    assert abs(oldw@vals/scale-q['values']['FL'])<1e-7,('FL baseline recombination',oldw@vals/scale,q['values']['FL'])
    by={c['segment']['label_id']:c for c in q['candidates']};tracks=[];lo=q['roi']['box'][0]
    for i,v,w in zip(ids,vals,oldw):
        f=extended_fl(by[i]['segment'],q['curve'],row['width'],row['height']);d=[];points={}
        for role,side in [('sup','upper'),('deep','lower')]:
            point=f['roots'][role]['roots'][0]['point'];p=pair[side];use=np.asarray(p['qualified'],bool)
            # Unknown support is least preferred but never removes a sparse set.
            dist=float(np.min(np.hypot(np.asarray(p['x'])[use]-point[0],np.asarray(p['y'])[use]-point[1]))) if use.any() else 1e12*span
            d.append(dist);points[side]=point
        x=by[i]['segment']['centroid'][0];b=max(0,min(2,int(3*(x-lo)/span)))
        tracks.append(dict(id=i,distance=max(d),region=b,FL_px=float(v),weight_before=float(w),points=points))
    selected,audit=select_tracks(tracks,policy,span);keep=np.array([i in set(selected) for i in ids])
    if keep.all():audit.update(tracks=tracks,unchanged=True);return q,audit
    weights,regions=combine_fl(vals[keep],quality[keep],kernel[keep],active,span,cfg['aggregate'],cal,True,True)
    q['values']['FL']=float(weights@vals[keep]/scale);q['FL_weights']={str(i):float(w) for i,w in zip(np.asarray(ids)[keep],weights)}
    q['selection']['value_px']=float(weights@vals[keep]);q['aggregation']['metrics']['FL'].update(regions=regions,contributing_ids=np.asarray(ids)[keep].tolist(),effective_tracks=float(1/(weights@weights)))
    for t in tracks:t['weight_after']=q['FL_weights'].get(str(t['id']),0.)
    audit.update(tracks=tracks,unchanged=False)
    return q,audit
