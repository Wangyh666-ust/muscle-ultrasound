"""Isolated Round36 postprocessing. No labels, expert values or model training."""
import copy
import math
import numpy as np
from scipy import ndimage
from scripts.round29_core import value_at, measure_candidates, validity
from scripts.round28_core import select, accepted, region, REGIONS


def axis_delta(a,b):
    return (np.asarray(a)-b+90)%180-90


def weighted_axis_median(angles,weights):
    a=np.asarray(angles,float);w=np.asarray(weights,float)
    medoid=min(range(len(a)),key=lambda k:(float(np.sum(abs(axis_delta(a,a[k]))*w)),float(a[k])))
    origin=a[medoid];d=axis_delta(a,origin);order=np.argsort(d,kind='stable')
    k=np.searchsorted(np.cumsum(w[order]),w.sum()/2,side='left')
    return float((origin+d[order[min(k,len(order)-1)]])%180)


def strip_curve(curve):
    if curve is None:return None
    c=copy.deepcopy(curve)
    for s in ('sup','deep'):c[s].pop('residuals_px',None)
    return c


def equal_pair(a,b,h,cfg):
    la,ha=a['support'];lb,hb=b['support'];lo=max(la,lb);hi=min(ha,hb)
    if hi<=lo or (hi-lo)/(max(ha,hb)-min(la,lb))<cfg['dedup_support_iou']:return False
    xs=np.linspace(lo,hi,33)
    return max(abs(value_at(a[s],x)-value_at(b[s],x)) for s in ('sup','deep') for x in xs)<=cfg['dedup_max_error_height_fraction']*h


def merge_pools(row,cfg):
    proposals=[]
    for d in ('D0','D1'):
        scores={s['pair_id']:s for s in row['scores'][d]['scores']}
        for p in row['pools'][d]:
            if p['curve'] is None:continue
            proposals.append({'source_decoder':d,'source_pair_id':p['pair_id'],'curve':strip_curve(p['curve']),
                              'old_quality':scores[p['pair_id']]['quality'],'sources':[[d,p['pair_id']]]})
    # Representative is fixed before scoring, with source identities preserved.
    proposals.sort(key=lambda p:(-p['old_quality'],p['source_decoder'],p['source_pair_id']))
    kept=[]
    for p in proposals:
        dupe=next((q for q in kept if equal_pair(p['curve'],q['curve'],row['height'],cfg)),None)
        if dupe is not None:dupe['sources']+=p['sources']
        else:kept.append(p)
    for i,p in enumerate(kept):p['pair_id']=i
    return kept


def support(curve,x):
    x=np.asarray(x);px=np.asarray(curve['paired_x']);ix=np.searchsorted(px,x)
    nearest=np.minimum(abs(x-px[np.clip(ix-1,0,len(px)-1)]),abs(x-px[np.clip(ix,0,len(px)-1)]))
    return (x>=curve['support'][0])&(x<=curve['support'][1])&(nearest<=curve['max_sample_gap'])


def interpolated_y(fit,x):
    # Calls here only evaluate observed support, not extension rays.
    return np.interp(x,fit['knots_x'],fit['knots_y'])


def independent_segments(segments,h,cfg):
    reps=[];groups=[]
    for s in sorted(segments,key=lambda z:(-z['length_px'],z['label_id'])):
        found=None
        for i,t in enumerate(reps):
            if abs(axis_delta(s['direction_deg'],t['direction_deg']))>cfg['duplicate_angle_deg']:continue
            delta=np.subtract(s['centroid'],t['centroid'])
            offsets=[abs(delta@np.array([-math.sin(math.radians(q['direction_deg'])),math.cos(math.radians(q['direction_deg']))])) for q in (s,t)]
            if max(offsets)<=cfg['duplicate_offset_height_fraction']*h:found=i;break
        if found is None:reps.append(s);groups.append([s['label_id']])
        else:groups[found].append(s['label_id'])
    return reps,groups


def prepare_evidence(mask,segments,w,h,cfg):
    labs,_=ndimage.label(mask,structure=np.ones((3,3),bool));ids=[s['label_id'] for s in segments]
    ev=np.isin(labs,ids);ys,xs=np.nonzero(ev)
    reps,groups=independent_segments(segments,h,cfg)
    weights=[min(s['length_px'],cfg['length_weight_cap_width_fraction']*w) for s in reps]
    direction=weighted_axis_median([s['direction_deg'] for s in reps],weights) if reps else None
    good=[(s,g,wt) for s,g,wt in zip(reps,groups,weights) if abs(axis_delta(s['direction_deg'],direction))<=cfg['angular_inlier_deg']]
    pixels={}
    for s,g,wt in good:
        yy,xx=np.nonzero(np.isin(labs,g));pixels[s['label_id']]=(xx,yy)
    return {'xs':xs,'ys':ys,'tracks':good,'pixels':pixels,'all_count':len(segments),'independent_count':len(reps),'reliable_count':len(good)}


def physical_prior_penalty(mt,scale,cfg):
    if not scale or mt is None:return 0.0,False
    lo,hi=cfg['flat_interval_mm'];distance=max(lo-mt,mt-hi,0.)
    # Bounded weak preference, never alters the measured thickness or creates a line.
    return cfg['max_score_penalty']*(1-math.exp(-.5*(distance/cfg['soft_shoulder_mm'])**2)),True


def candidate_scores(pool,ev,probs,row,cfg):
    if not pool:return []
    lo=min(p['curve']['support'][0] for p in pool);hi=max(p['curve']['support'][1] for p in pool)
    xs,ys=ev['xs'],ev['ys'];keep=(xs>=lo)&(xs<=hi);xs=xs[keep];ys=ys[keep]
    bins=np.minimum(((xs-lo)/max(hi+1-lo,1)*3).astype(int),2)
    den=[int(np.sum(bins==b)) for b in range(3)];active=[b for b in range(3) if den[b]]
    h,w=row['height'],row['width'];out=[]
    for p in pool:
        cv=p['curve'];inside=support(cv,xs)&(ys>=interpolated_y(cv['sup'],xs))&(ys<=interpolated_y(cv['deep'],xs))
        oldcontain=float(np.mean([inside[bins==b].mean() for b in active])) if active else 0.
        fractions=[[],[],[]];track_records=[]
        for s,g,wt in ev['tracks']:
            b=region(s['centroid'][0],[lo,hi])
            if b is None:continue
            xx,yy=ev['pixels'][s['label_id']]
            ins=support(cv,xx)&(yy>=interpolated_y(cv['sup'],xx))&(yy<=interpolated_y(cv['deep'],xx))
            frac=float(ins.mean()) if len(ins) else 0
            fractions[b].append((frac,wt))
            track_records.append({'id':s['label_id'],'region':b,'inside':frac,'weight':wt,'members':g})
        occupied=[b for b in range(3) if fractions[b]];rw=cfg['c1']['region_weights'];total=sum(rw[b] for b in occupied)
        c1=sum(rw[b]*np.average([f for f,t in fractions[b]],weights=[t for f,t in fractions[b]]) for b in occupied)/total if total else 0.
        gate=len(track_records)>=cfg['c1']['min_independent_tracks_for_gate'] and len(occupied)>=cfg['c1']['min_active_regions_for_gate']
        px=np.asarray(cv['paired_x']);xx=np.rint(px).astype(int);confidence=[];identity=[]
        ordered=True
        for side,ch in [('sup',1),('deep',2)]:
            yy=np.rint(interpolated_y(cv[side],px)).astype(int);valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            if not valid.all():ordered=False
            if valid.any():
                # Search a 2-pixel ORIGINAL-coordinate neighborhood; no mask expansion.
                samples=np.stack([probs[:,np.clip(yy[valid]+dy,0,h-1),xx[valid]] for dy in (-2,-1,0,1,2)])
                v=samples.max(0);confidence.append(float(np.maximum(v[0],v[ch]).mean()));identity.append(float(v[ch].mean()))
            else:confidence.append(0.);identity.append(0.)
        ordered=ordered and all(value_at(cv['sup'],x)<value_at(cv['deep'],x) for x in np.linspace(*cv['support'],33))
        span=max(hi-lo+1,1);cov=min(1.,len(px)/span)
        wt=cfg['c1']['weights'];newscore=wt['containment']*c1+wt['support']*cov+wt['boundary_probability']*np.mean(confidence)+wt['identity_probability']*np.mean(identity)
        oldscore=oldcontain*p['old_quality']
        mpx=float(np.mean([value_at(cv['deep'],x)-value_at(cv['sup'],x) for x in np.linspace(*cv['support'],5)[1:-1]]))
        mt=mpx/row['px_per_mm'] if row.get('px_per_mm') else None
        penalty,enabled=physical_prior_penalty(mt,row.get('px_per_mm'),cfg['mt_prior'])
        out.append({'pair_id':p['pair_id'],'old_score':oldscore,'old_containment':oldcontain,'quality':p['old_quality'],
            'c1_score':float(newscore) if track_records else oldscore,'containment':float(c1),'support_fraction':cov,
            'boundary_probability':float(np.mean(confidence)),'identity_probability':float(np.mean(identity)),
            'gate_applied':gate,'allowed':bool(ordered and (not gate or c1>=cfg['c1']['min_containment'])),
            'tracks':track_records,'MT_mm':mt,'prior_penalty':penalty,'prior_enabled':enabled})
    return out


def choose(pool,scores,row,use_c1=False,use_prior=False):
    candidates=[s for s in scores if not use_c1 or s['allowed']]
    if not candidates:return None
    lookup={p['pair_id']:p for p in pool}
    def rank(s):
        p=lookup[s['pair_id']];cv=p['curve'];spacing=np.median([value_at(cv['deep'],x)-value_at(cv['sup'],x) for x in cv['paired_x']])
        score=s['c1_score' if use_c1 else 'old_score']-(s['prior_penalty'] if use_prior else 0)
        return (-score,-p['old_quality'],abs(spacing-.358*row['height']),s['pair_id'])
    return lookup[min(candidates,key=rank)['pair_id']]


def measure(curve,segments,row,pair_id):
    cs,m=measure_candidates([{'label_id':s['label_id'],'segment':s} for s in segments],curve,row['width'],row['height'])
    selection=select(cs,curve['support'] if curve else None,'UW');scale=row.get('px_per_mm') or 1.
    values={'MT':m['MT']/scale if m['MT'] is not None else None,'PA':m['PA'],'FL':selection['value_px']/scale if selection['value_px'] is not None else None}
    out={'curve':curve,'selected_pair_id':pair_id,'values':values,'selection':selection,'PA_ids':m['PA_ids'],'PA_invalid':m['PA_invalid'],
         'MT_reasons':m['MT_reasons'],'mt_positions':m['mt_positions'],'stage':'boundary_failure' if curve is None else 'output' if values['FL'] is not None else 'no_FL',
         'candidates':[{'segment':c['segment'],'invalid':c['invalid'],'invalid_conditions':c['invalid_conditions'],'raw_FL':c['F0']['fl_px_raw'],
                        'extrapolation_ratio':c['F0']['extrapolation_ratio'],'G1':c['F0']['G1']} for c in cs]}
    return out,cs


def consensus_measure(base,segments,row,cfg):
    out=copy.deepcopy(base);curve=base['curve'];out['direction_diagnostic']={'regions':[],'changed_ids':[],'rejected_anchor_ids':[]}
    if curve is None:return out
    valid=[s for s in segments if not validity(s,curve)[0]];reps,groups=independent_segments(valid,row['height'],cfg['c1'])
    regions=[]
    for b in range(3):
        pool=[s for s in reps if region(s['centroid'][0],curve['support'])==b]
        weights=[min(s['length_px'],cfg['c1']['length_weight_cap_width_fraction']*row['width']) for s in pool]
        med=weighted_axis_median([s['direction_deg'] for s in pool],weights) if pool else None
        inds=[i for i,s in enumerate(pool) if abs(axis_delta(s['direction_deg'],med))<=cfg['direction']['inlier_deg']]
        active=len(inds)>=cfg['direction']['min_independent_inliers']
        med=weighted_axis_median([pool[i]['direction_deg'] for i in inds],[weights[i] for i in inds]) if active else None
        q={'region':REGIONS[b],'independent_count':len(pool),'inlier_ids':[pool[i]['label_id'] for i in inds],'direction_deg':med,'active':active}
        regions.append(q)
    adjusted=[]
    for s in segments:
        t=copy.deepcopy(s);b=region(s['centroid'][0],curve['support']);q=regions[b] if b is not None else None
        if q and q['active'] and not validity(s,curve)[0] and abs(axis_delta(q['direction_deg'],s['direction_deg']))<=cfg['direction']['max_anchor_change_deg']:
            t['direction_deg']=q['direction_deg'];t['original_direction_deg']=s['direction_deg']
            # Endpoints remain observations, never paint synthetic segments as C1 mask.
            if abs(axis_delta(t['direction_deg'],s['direction_deg']))>1e-8:out['direction_diagnostic']['changed_ids'].append(s['label_id'])
        adjusted.append(t)
    fresh,cs=measure(curve,adjusted,row,base['selected_pair_id']);byid={c['label_id']:c for c in cs}
    s=copy.deepcopy(base['selection']);ids=base['selection']['selected_ids'];keep=[i for i in ids if accepted(byid[i],'U')[0]]
    weights=[w for i,w in zip(ids,s['weights']) if i in keep];sw=sum(weights);weights=[v/sw for v in weights] if sw else []
    bins=[region(byid[i]['segment']['centroid'][0],curve['support']) for i in keep]
    s.update(selected_ids=keep,weights=weights,selected_regions=[REGIONS[b] for b in bins],
        value_px=sum(byid[i]['F0']['fl_px_raw']*v for i,v in zip(keep,weights)) if keep else None,
        missing_regions=[REGIONS[b] for b in range(3) if b not in bins],three_regions_complete=len(set(bins))==3,
        selected_invalid=[i for i in keep if byid[i]['invalid']],selected_over_80=[i for i in keep if 'excessive_extrapolation' in byid[i]['F0']['G1']['conditions']],
        max_extrapolation=max((byid[i]['F0']['extrapolation_ratio'] for i in keep),default=None),
        eligible_ids=[c['label_id'] for c in cs if accepted(c,'U')[0]],
        selected_no_visible_support=[i for i in keep if byid[i]['F0']['visible_length_px']<=0])
    pa_ids=[i for i in base['PA_ids'] if byid[i]['PA']['pa_deg'] is not None]
    out.update(candidates=fresh['candidates'],selection=s,PA_ids=pa_ids,PA_invalid=[i for i in pa_ids if byid[i]['invalid']])
    out['values']['FL']=s['value_px']/(row.get('px_per_mm') or 1.) if s['value_px'] is not None else None
    out['values']['PA']=float(np.mean([byid[i]['PA']['pa_deg'] for i in pa_ids])) if pa_ids else None
    out['stage']='output' if out['values']['FL'] is not None else 'no_FL'
    out['direction_diagnostic'].update(regions=regions,rejected_anchor_ids=[i for i in ids if i not in keep])
    return out
