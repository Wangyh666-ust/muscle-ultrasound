"""Experimental Round29 geometry. No model, expert or reference-data access."""
import copy
import itertools
import math
import numpy as np
from scripts.pair_selection import run_centres_per_column, build_tracks
from scripts.round25_boundary_fit import fit_curve
from scripts.round27_geometry import roots_on_curve, compact_fl
from scripts.fascicle_pa_fl import fl_for_segment, pa_for_segment, select_candidates
from scripts.round28_core import select, accepted, region, REGIONS

TOL=1e-8


def endpoint_line(points, right, fraction=.25):
    p=np.asarray(points,float); lo,hi=p[0,0],p[-1,0]
    q=p[p[:,0]>=hi-fraction*(hi-lo)] if right else p[p[:,0]<=lo+fraction*(hi-lo)]
    if len(q)<2: q=p[-2:] if right else p[:2]
    return np.polyfit(q[:,0],q[:,1],1)


def merge_tracks(tracks,width,height,cfg):
    """Only bridge nonoverlapping observed runs. Never invent support columns."""
    ts=[{'points':list(t['points']),'source_ids':[i]} for i,t in enumerate(tracks)]
    events=[]
    while True:
        choices=[]
        for i,a in enumerate(ts):
            if len(a['points'])<5: continue
            la=endpoint_line(a['points'],True)
            for k,b in enumerate(ts):
                if i==k or len(b['points'])<5: continue
                gap=b['points'][0][0]-a['points'][-1][0]
                if not 1<gap<=max(3,cfg['merge_gap_width_fraction']*width): continue
                lb=endpoint_line(b['points'],False)
                angle=abs(math.degrees(math.atan(la[0])-math.atan(lb[0])))
                err=max(abs(np.polyval(la,b['points'][0][0])-b['points'][0][1]),
                        abs(np.polyval(lb,a['points'][-1][0])-a['points'][-1][1]))
                if angle<=cfg['merge_angle_deg'] and err<=max(cfg['merge_min_error_px'],cfg['merge_error_height_fraction']*height):
                    choices.append((err,gap,i,k))
        if not choices: break
        _,gap,i,k=min(choices)
        events.append({'left':ts[i]['source_ids'],'right':ts[k]['source_ids'],'gap':gap})
        ts[i]={'points':ts[i]['points']+ts[k]['points'],'source_ids':ts[i]['source_ids']+ts[k]['source_ids']}
        ts.pop(k)
    return ts,events


def add_extensions(curve,points,cfg):
    c=copy.deepcopy(curve)
    for j,side in enumerate(('sup','deep'),1):
        for right,name in ((False,'left_ray'),(True,'right_ray')):
            slope=float(endpoint_line([(p[0],p[j]) for p in points],right,cfg['extension_fit_fraction'])[0])
            at=-1 if right else 0
            x=c[side]['knots_x'][at]; y=c[side]['knots_y'][at]
            c[side][name]=[slope,float(y-slope*x)]
    return c


def boundary_pool(mask,baseline,cfg):
    """Mask-only common rule, also applicable to all 153 apo reference images.

    Rank observed common support penalized by intervening long mask tracks;
    within 10% of best quality use the existing 0.358*height spacing prior.
    All proposals are fitted with F1 before any straight-line residual rejection.
    """
    h,w=mask.shape
    runs=run_centres_per_column(mask)
    raw=build_tracks(runs,cfg['track_tol_px'],cfg['track_gap_columns'])
    tracks,merges=merge_tracks(raw,w,h,cfg)
    tracks=[t for t in tracks if len(t['points'])>=max(5,cfg['min_track_fraction']*w)]
    maps=[dict(t['points']) for t in tracks]
    candidates=[]; keys=set()
    def append(points,source):
        if not points: return
        key=tuple(tuple(p) for p in points)
        if key in keys: return
        keys.add(key)
        x=[p[0] for p in points]; support=[min(x),max(x)]
        cv,reason=fit_curve(points,support,h,max(2,.015*w))
        spacing=float(np.median([p[2]-p[1] for p in points]))
        inside=sum(any(p[0] in m and p[1]+cfg['interior_track_margin_px']<m[p[0]]<p[2]-cfg['interior_track_margin_px'] for m in maps) for p in points)
        q=len(points)/w*(1-inside/len(points))
        candidates.append({'pair_id':len(candidates),'source':source,'points':points,'curve':add_extensions(cv,points,cfg) if cv else None,
            'failure':reason or None,'support':support,'common_columns':len(points),'interior_columns':inside,
            'quality':q,'spacing_px':spacing})
    # Include baseline even if its tracks fragment differently in the new pool.
    if baseline.get('points'): append(baseline['points'],'baseline_pair')
    for i,a in enumerate(maps):
        for k in range(i+1,len(maps)):
            b=maps[k]; xs=sorted(a.keys()&b.keys())
            if len(xs)<max(8,cfg['min_common_fraction']*w) or xs[-1]-xs[0]<cfg['min_common_fraction']*w: continue
            if np.median([a[x]-b[x] for x in xs])>0: su,de=b,a
            else: su,de=a,b
            pts=[(x,su[x],de[x]) for x in xs if de[x]-su[x]>max(2,.005*h)]
            if len(pts)>=max(8,cfg['min_common_fraction']*w): append(pts,[tracks[i]['source_ids'],tracks[k]['source_ids']])
    valid=[p for p in candidates if p['curve'] is not None]
    if valid:
        best=max(p['quality'] for p in valid)
        near=[p for p in valid if p['quality']>=best*(1-cfg['quality_near_fraction'])]
        chosen=min(near,key=lambda p:(abs(p['spacing_px']-cfg['spacing_prior_height_fraction']*h),-p['quality'],p['pair_id']))
    else: chosen=None
    return {'curve':chosen['curve'] if chosen else None,'selected_pair_id':chosen['pair_id'] if chosen else None,
        'selected_points':chosen['points'] if chosen else [],'pairs':candidates,'track_count':len(tracks),
        'original_track_count':len(raw),'merges':merges,'failure':None if chosen else 'no_valid_boundary_pair',
        'selected_pair_changed': bool(chosen and chosen['source']!='baseline_pair')}


def line_at(fit,x):
    xs,ys=fit['knots_x'],fit['knots_y']
    if x<xs[0]: return fit['left_ray']
    if x>xs[-1]: return fit['right_ray']
    i=min(max(int(np.searchsorted(xs,x,side='right'))-1,0),len(xs)-2)
    m=(ys[i+1]-ys[i])/(xs[i+1]-xs[i]); return [m,ys[i]-m*xs[i]]


def value_at(fit,x): return float(np.polyval(line_at(fit,x),x))


def extended_roots(seg,fit):
    out=roots_on_curve(seg,fit)
    cx,cy=seg['centroid']; th=math.radians(seg['direction_deg']); dx,dy=math.cos(th),math.sin(th)
    for side,name in ((0,'left_ray'),(-1,'right_ray')):
        m,b=fit[name]; den=dy-m*dx; rhs=m*cx+b-cy
        if abs(den)<1e-12:
            if abs(rhs)<TOL: out['coincident_pieces'].append(name)
            continue
        t=rhs/den; x=cx+t*dx
        if (side==0 and x<=fit['knots_x'][0]+TOL) or (side==-1 and x>=fit['knots_x'][-1]-TOL):
            if not any(abs(t-r['t'])<=TOL for r in out['roots']):
                out['roots'].append({'t':t,'point':[x,cy+t*dy],'piece':name,'line':[m,b]})
    out['roots'].sort(key=lambda r:r['t'])
    return out


def extended_fl(seg,curve,width,height):
    if curve is None: return {**compact_fl(fl_for_segment(seg,None,width,height,False)),'roots':{}}
    roots={s:extended_roots(seg,curve[s]) for s in ('sup','deep')}; reasons=[]
    for s,r in roots.items():
        if r['coincident_pieces']: reasons.append(s+'_coincident_boundary')
        elif len(r['roots'])!=1: reasons.append(s+('_no_intersection' if not r['roots'] else '_multiple_intersections'))
    if reasons: return {**compact_fl({'fl_failure_conditions':reasons,'fl_failure_conditions_pure':reasons}),'roots':roots}
    local={k:curve[k] for k in ('support','paired_x','max_sample_gap')}
    local.update({s:roots[s]['roots'][0]['line'] for s in ('sup','deep')})
    raw=fl_for_segment(seg,local,width,height,True)
    xlo,xhi=sorted(roots[s]['roots'][0]['point'][0] for s in ('sup','deep'))
    probes=[xlo,xhi]+[x for x in curve['sup']['knots_x'] if xlo<x<xhi]
    if any(value_at(curve['sup'],x)>=value_at(curve['deep'],x)-TOL for x in probes):
        # Keep raw intersections but reject geometrically inverted continuation.
        for k in ('fl_failure_conditions','fl_failure_conditions_pure'):
            raw[k]=list(raw[k])+['inverted_boundary_continuation']
    return {**compact_fl(raw),'roots':roots}


def validity(seg,curve):
    if curve is None: return False,['no_apo_boundaries']
    xs=np.linspace(seg['p0'][0],seg['p1'][0],21);ys=np.linspace(seg['p0'][1],seg['p1'][1],21)
    inside=np.mean([value_at(curve['sup'],x)<=y<=value_at(curve['deep'],x) for x,y in zip(xs,ys)])
    supported=np.mean([min(abs(x-np.asarray(curve['paired_x'])))<=curve['max_sample_gap'] for x in xs])
    reasons=[]
    if inside<.5: reasons.append('candidate_outside_band_region')
    if supported<.5: reasons.append('candidate_local_support_gap')
    return bool(reasons),reasons


def measure_candidates(candidates,curve,width,height):
    new=[];pa_records=[]
    for c in candidates:
        s=c['segment']; f=extended_fl(s,curve,width,height); inv,reasons=validity(s,curve)
        local=None
        if curve:
            local={k:curve[k] for k in ('support','paired_x','max_sample_gap')}
            local.update({side:line_at(curve[side],s['centroid'][0]) for side in ('sup','deep')})
        pa=pa_for_segment(s,local,width,curve is not None)
        new.append({'label_id':c['label_id'],'segment':s,'invalid':inv,'invalid_conditions':reasons,'F0':f,'PA':pa})
        # Frozen PA selector, independent of FL acceptance.
        pa_records.append({'segment':s,**pa,'intersection_deep':f['intersection_deep'],'fl_accept_pure':False,'fl_accept_strict':False})
    ps=select_candidates(pa_records)
    ids=[c['segment']['label_id'] for c in ps['pa']]
    mt=None;positions=[];mt_reasons=[]
    if curve:
        lo,hi=curve['support']; positions=[lo+q*(hi-lo) for q in (.25,.5,.75)]
        if not all(min(abs(x-np.asarray(curve['paired_x'])))<=curve['max_sample_gap'] for x in positions):
            mt_reasons.append('measurement_in_support_gap')
        if any(not 0<=value_at(curve[side],x)<height for x in positions for side in ('sup','deep')):
            mt_reasons.append('fitted_band_outside_image')
        if any(value_at(curve['deep'],x)<=value_at(curve['sup'],x) for x in positions):
            mt_reasons.append('nonpositive_thickness')
        if not mt_reasons:
            mt=float(np.mean([value_at(curve['deep'],x)-value_at(curve['sup'],x) for x in positions]))
    else:mt_reasons.append('no_apo_boundaries')
    return new,{'MT':mt,'MT_reasons':mt_reasons,'PA':ps['pa_mean_deg'],'PA_ids':ids,'PA_invalid':[c['label_id'] for c in new if c['label_id'] in ids and c['invalid']],'mt_positions':positions}


def middepth_x(seg,curve):
    fit=copy.deepcopy(curve['sup'])
    fit['knots_y']=list(np.mean([curve[s]['knots_y'] for s in ('sup','deep')],axis=0))
    for name in ('left_ray','right_ray'): fit[name]=list(np.mean([curve[s][name] for s in ('sup','deep')],axis=0))
    r=extended_roots(seg,fit)
    return r['roots'][0]['point'][0] if len(r['roots'])==1 and not r['coincident_pieces'] else None


def trajectory_duplicate(a,b,cfg):
    da=abs(a['direction_deg']-b['direction_deg'])%180;da=min(da,180-da)
    delta=np.subtract(a['centroid'],b['centroid']); perp=[]
    for s in (a,b):
        th=math.radians(s['direction_deg']);perp.append(abs(delta@np.array([-math.sin(th),math.cos(th)])))
    return da<=cfg['duplicate_angle_deg'] and max(perp)<=cfg['duplicate_perp_px']


def spatial_select(candidates,curve,cfg):
    support=curve['support'] if curve else None
    result=select(candidates,support,'UW'); span=support[1]-support[0] if support else 1.
    eligible=[c for c in candidates if accepted(c,'U')[0] and region(c['segment']['centroid'][0],support) is not None]
    mid={c['label_id']:middepth_x(c['segment'],curve) for c in eligible}
    buckets=[[None]+[c for c in eligible if region(c['segment']['centroid'][0],support)==i and mid[c['label_id']] is not None] for i in range(3)]
    anchors=[support[0]+u*span for u in cfg['anchors']] if support else []
    best=None;chosen=[]; tested=0
    for comb in itertools.product(*buckets):
        cs=[c for c in comb if c is not None];good=True
        for a,b in itertools.combinations(cs,2):
            if (abs(a['segment']['centroid'][0]-b['segment']['centroid'][0])<cfg['min_center_separation_fraction']*span or
                abs(mid[a['label_id']]-mid[b['label_id']])<cfg['min_middepth_separation_fraction']*span or
                trajectory_duplicate(a['segment'],b['segment'],cfg)): good=False;break
        if not good: continue
        tested+=1
        score=(-sum(cfg['weights'][i] for i,c in enumerate(comb) if c is not None),
               sum(abs(c['segment']['centroid'][0]-anchors[i])/span for i,c in enumerate(comb) if c is not None),
               sum(c['F0']['extrapolation_ratio'] for c in cs),tuple(c['label_id'] if c is not None else -1 for c in comb))
        if best is None or score<best: best=score;chosen=cs
    # Reuse aggregation accounting; do not duplicate a line to fill an empty slot.
    selected=select(chosen,support,'UW')
    result.update({k:v for k,v in selected.items() if k not in ('eligible_ids','outside_spatial_support_ids')})
    result.update(variant='anchored_independent_UW',anchors_x=anchors,middepth_x={str(k):v for k,v in mid.items()},
                  feasible_combinations=tested,spatial_missing_reason='no independent candidate satisfying fixed spatial rule')
    return result
