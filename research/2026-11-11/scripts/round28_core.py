"""Round28 only: fixed 2x3 postprocessing comparison, independent of expert labels."""
import math
import numpy as np

VARIANTS={'R':('G1','global','equal'),'S':('G1','spatial','equal'),'W':('G1','spatial','center'),
          'U':('U','global','equal'),'US':('U','spatial','equal'),'UW':('U','spatial','center')}
REGIONS=('left','middle','right')


def accepted(candidate,policy):
    reasons=list(candidate['F0']['G1']['conditions'])
    if policy=='U': reasons=[r for r in reasons if r!='excessive_extrapolation']
    elif policy!='G1': raise ValueError('unknown policy')
    value=candidate['F0']['fl_px_raw']
    if value is None or not math.isfinite(value) or value<=0:
        if 'nonfinite_or_nonpositive_length' not in reasons: reasons.append('nonfinite_or_nonpositive_length')
    return not reasons,reasons


def region(x,support):
    if support is None: return None
    lo,hi=map(float,support)
    if hi<=lo or not math.isfinite(x) or x<lo or x>hi: return None
    return min(int((x-lo)/(hi-lo)*3),2)


def rank(c):
    f=c['F0']; r,v=f['extrapolation_ratio'],f['visible_length_px']
    if r is None or v is None or not math.isfinite(r) or not math.isfinite(v):
        raise ValueError('accepted candidate lacks ranking fields')
    return (r,-v)


def select(candidates,support,variant):
    if len({c['label_id'] for c in candidates})!=len(candidates): raise ValueError('duplicate candidate')
    policy,selection,weighting=VARIANTS[variant]
    eligible=[c for c in candidates if accepted(c,policy)[0]]
    outside=[c['label_id'] for c in eligible if region(c['segment']['centroid'][0],support) is None]
    if selection=='global': chosen=sorted(eligible,key=rank)[:3]
    else:
        chosen=[]
        for i in range(3):
            pool=[c for c in eligible if region(c['segment']['centroid'][0],support)==i]
            if pool: chosen.append(min(pool,key=rank))
    labels=[c['label_id'] for c in chosen]
    if len(labels)!=len(set(labels)): raise ValueError('duplicate selected candidate')
    bins=[region(c['segment']['centroid'][0],support) for c in chosen]
    weights=[(.25,.5,.25)[i] for i in bins] if weighting=='center' else [1.]*len(chosen)
    total=sum(weights); weights=[x/total for x in weights] if total else []
    value=sum(w*c['F0']['fl_px_raw'] for w,c in zip(weights,chosen)) if chosen else None
    return {'variant':variant,'policy':policy,'experimental':policy=='U',
        'eligible_ids':[c['label_id'] for c in eligible], 'outside_spatial_support_ids':outside,
        'selected_ids':labels,'selected_regions':[REGIONS[i] if i is not None else None for i in bins],
        'weights':weights,'value_px':float(value) if value is not None else None,
        'missing_regions':[REGIONS[i] for i in range(3) if i not in bins],
        'three_regions_complete':len(set(i for i in bins if i is not None))==3,
        'selected_invalid':[c['label_id'] for c in chosen if c['invalid']],
        'selected_over_80':[c['label_id'] for c in chosen if 'excessive_extrapolation' in c['F0']['G1']['conditions']],
        'selected_no_visible_support':[c['label_id'] for c in chosen if c['F0']['visible_length_px']<=0],
        'max_extrapolation':max((c['F0']['extrapolation_ratio'] for c in chosen),default=None)}


def rejection_details(component):
    flags=[]
    for key,limit,name in [('area_px',20,'area_lt_20'),('length_px',30,'length_lt_30'),
                           ('linearity',.9,'linearity_lt_0.90'),('elongation',3,'elongation_lt_3')]:
        if component[key]<limit: flags.append(name)
    if component['width_px']>8: flags.append('width_gt_8')
    return flags


def invariant_checks(arms,candidates,baseline):
    r=arms['R']
    if r['selected_ids']!=baseline['selected_ids']['G1']: raise ValueError('R selected identity replay')
    old=baseline['values']['G1']; scale=baseline['px_per_mm'] or 1.
    v=r['value_px']/scale if r['value_px'] is not None else None
    if (v is None)!=(old is None) or (v is not None and not np.isclose(v,old,atol=1e-8,rtol=1e-12)):
        raise ValueError('R value replay')
    if not set(r['eligible_ids'])<=set(arms['U']['eligible_ids']): raise ValueError('U removed old eligibility')
    for a,b in (('S','W'),('US','UW')):
        if arms[a]['selected_ids']!=arms[b]['selected_ids']: raise ValueError('weighting reselected')
    for a in arms.values():
        if len(a['selected_ids'])!=len(set(a['selected_ids'])): raise ValueError('selected duplicate')
        if a['weights'] and not np.isclose(sum(a['weights']),1.): raise ValueError('weight sum')
    return True
