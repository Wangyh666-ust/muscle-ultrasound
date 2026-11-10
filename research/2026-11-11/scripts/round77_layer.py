"""Candidate-only subject-layer evidence. No labels or reference measurements."""
from copy import deepcopy
import math
import numpy as np
from scripts.round63_paths import sample

def signature(p):return None if p is None else [list(p[s]['ids']) for s in ('upper','lower')]
def same(a,b):
    if a is None or b is None:return a is None and b is None
    return all(np.array_equal(a[s][k],b[s][k]) for s in ('upper','lower') for k in ('ids','x','y','confidence','observed','kind'))
def sortkey(p):return (-p['score'],-np.ptp(p['upper']['x'])-np.ptp(p['lower']['x']),p['upper']['ids'],p['lower']['ids'])
def tail(mm):
    if mm is None:return 0.
    d=max(10-mm,mm-31,0.);return float(.4*(-math.expm1(-.5*(d/2.)**2)))

def path_distance(a,b,width):
    lo=max(a['x'][0],b['x'][0]);hi=min(a['x'][-1],b['x'][-1])
    if hi-lo<.08*width:return None
    x=np.linspace(lo,hi,31);return float(np.quantile(abs(np.interp(x,a['x'],a['y'])-np.interp(x,b['x'],b['y'])),.9))

def body_evidence(pairs,paths,prob,segments,width,bounds,cfg):
    groups=[];pathmap={};stats=[]
    for role,pool in enumerate(paths):
        for p in pool:
            observed=np.asarray(p['observed'],bool);x,y=np.asarray(p['x'])[observed],np.asarray(p['y'])[observed]
            if not len(x):continue
            pr=sample(prob[role+1],x,y);op=sample(prob[2-role],x,y);un=sample(prob[0],x,y)
            certified=(len(x)/width>=cfg['min_span'] and np.ptp(x)/width>=cfg['min_span'] and float(np.mean(pr))>=cfg['min_role'] and float(np.mean(pr-op))>=cfg['min_contrast'] and float(np.mean(un>=.35))>=.65)
            stats.append(dict(role=role,ids=p['ids'],observed=len(x),span=float(np.ptp(x)/width),role_mean=float(pr.mean()),opposite_mean=float(op.mean()),union_hit=float(np.mean(un>=.35)),certified=certified))
            if not certified:continue
            near=next((i for i,q in enumerate(groups) if (d:=path_distance(p,q['path'],width)) is not None and d<=cfg['dedup_roi']*width),None)
            if near is None:near=len(groups);groups.append(dict(path=p,roles={role}))
            else:groups[near]['roles'].add(role)
            pathmap[id(p)]=near
    # IDs can repeat across geometric variants; paths are looked up geometrically.
    def group(p,role):
        best=[]
        for i,g in enumerate(groups):
            if role not in g['roles']:continue
            d=path_distance(p,g['path'],width)
            if d is not None and d<=cfg['dedup_roi']*width:best.append((d,i))
        return min(best)[1] if best else None
    bands={}
    for p in pairs:
        u,d=group(p['upper'],0),group(p['lower'],1)
        if u is None or d is None or u==d:continue
        lo,hi=p['interval'];internal=[]
        for i,g in enumerate(groups):
            if i in (u,d):continue
            q=g['path'];use=np.asarray(q['observed'],bool)&(q['x']>=lo)&(q['x']<=hi);x,y=q['x'][use],q['y'][use]
            if len(x)<cfg['min_span']*width:continue
            inside=(y>np.interp(x,p['upper']['x'],p['upper']['y'])+.01*width)&(y<np.interp(x,p['lower']['x'],p['lower']['y'])-.01*width)
            # One certified physical track, never a union of separate bright lines.
            if inside.sum()>=cfg['min_span']*width and np.ptp(x[inside])>=cfg['min_span']*width:internal.append(i)
        if internal:continue
        key=(u,d)
        if key not in bands or p['score']>bands[key]['score']:bands[key]=p
    audit=dict(paths=stats,certified_physical_tracks=len(groups),adjacent_bands=len(bands))
    if not bands:return None,dict(**audit,reason='no_certified_adjacent_band')
    bandrows=list(bands.values());lo=max(p['interval'][0] for p in bandrows);hi=min(p['interval'][1] for p in bandrows)
    if hi-lo<.12*width:return None,dict(**audit,reason='no_shared_observed_comparison_window')
    # A physical layer may have several partial-chain hypotheses. Collapse their
    # geometrically equivalent corridors before counting competing assignments.
    # Reuse the registered side-match tolerance; no new threshold is calibrated.
    xx=np.linspace(lo,hi,31);distinct=[]
    for p in sorted(bandrows,key=lambda q:-q['score']):
        duplicate=any(max(float(np.quantile(abs(np.interp(xx,p[s]['x'],p[s]['y'])-np.interp(xx,q[s]['x'],q[s]['y'])),.9)) for s in ('upper','lower'))<=cfg['match_roi']*width for q in distinct)
        if not duplicate:distinct.append(p)
    audit['distinct_adjacent_layers']=len(distinct)
    bandrows=distinct
    # A wide candidate can skip a credible internal boundary, leaving exactly
    # one valid adjacent layer. That is a useful hypothesis, not a missing rival.
    alternatives=[]
    for p in pairs:
        if all(max(float(np.quantile(abs(np.interp(xx,p[s]['x'],p[s]['y'])-np.interp(xx,q[s]['x'],q[s]['y'])),.9)) for s in ('upper','lower'))>cfg['match_roi']*width for q in alternatives):alternatives.append(p)
    if len(alternatives)<2:return None,dict(**audit,reason='only_one_original_physical_hypothesis')
    # Saturate repeated spatial evidence; do not divide by candidate band area.
    cells={};cell=.02*width
    for s in segments:
        length=min(float(s.get('length_px',np.linalg.norm(np.subtract(s['p1'],s['p0'])))),.12*width)
        mass=length*float(s.get('linearity',1))*float(s.get('filter_quality',1))/31
        for point in np.linspace(s['p0'],s['p1'],31):
            if not lo<=point[0]<=hi:continue
            key=tuple(np.floor((point-np.asarray([bounds[0],0]))/cell).astype(int))
            if key not in cells or cells[key]['mass']<mass:cells[key]=dict(point=point,mass=mass,id=s['label_id'])
    if not cells:return None,dict(**audit,reason='no_C1_in_shared_window')
    obs=list(cells.values());xy=np.asarray([o['point'] for o in obs]);mass=np.asarray([o['mass'] for o in obs]);x,y=xy.T
    bins=np.clip(((x-bounds[0])/max(np.ptp(bounds),1)*3).astype(int),0,2);active=sorted(set(bins));ids={o['id'] for o in obs}
    if len(ids)<3 or len(active)<2:return None,dict(**audit,reason='C1_subject_insufficient_spatial_support',ids=len(ids),regions=active)
    for b,rw in enumerate((.25,.50,.25)):
        use=bins==b
        if use.any():mass[use]*=rw/max(mass[use].sum(),1e-12)
    memberships=np.asarray([(y>=np.interp(x,p['upper']['x'],p['upper']['y']))&(y<=np.interp(x,p['lower']['x'],p['lower']['y'])) for p in bandrows])
    count=memberships.sum(0);exclusive=count==1;unassigned=float(mass[count==0].sum()/mass.sum());ambiguous=float(mass[count>1].sum()/mass.sum())
    if unassigned+ambiguous>.4:return None,dict(**audit,reason='subject_assignment_ambiguous',unassigned_mass=unassigned,ambiguous_mass=ambiguous)
    evidence=np.asarray([(mass*member*exclusive).sum() for member in memberships]);order=np.argsort(-evidence);total=evidence.sum()
    shares=evidence/max(total,1e-12);win=int(order[0]);gap=float(shares[win]-(shares[order[1]] if len(order)>1 else 0.))
    audit.update(common_window=[lo,hi],band_mass=shares.tolist(),dominance=float(shares[win]),dominance_gap=gap,subject_band_ids=signature(bandrows[win]))
    if shares[win]<cfg['dominance'] or gap<cfg['dominance_gap']:return None,dict(**audit,reason='no_dominant_subject_layer')
    keep=memberships[win]&exclusive
    return dict(pair=bandrows[win],xy=xy[keep],mass=mass[keep],interval=[lo,hi]),dict(**audit,reason='dominant_layer_supported')

def rank(pairs,paths,prob,segments,width,bounds,cfg=None,use_prior=False):
    subject,au=body_evidence(pairs,paths,prob,segments,width,bounds,cfg) if cfg else (None,dict(reason='L_disabled'))
    ranked=[];details=[]
    for p in pairs:
        q=deepcopy(p);extra=tail(p['MT_mm']) if use_prior else 0.;q['score']-=extra
        bonus=0.;newC=None;matches=[]
        if subject:
            x,y=subject['xy'].T;mass=subject['mass'];known=(x>=p['interval'][0])&(x<=p['interval'][1])
            if known.sum()>=3 and mass[known].sum()>=.5*mass.sum():
                x,y,mass=x[known],y[known],mass[known]
                contain=float(mass@((y>=np.interp(x,p['upper']['x'],p['upper']['y']))&(y<=np.interp(x,p['lower']['x'],p['lower']['y'])))/mass.sum())
                for s in ('upper','lower'):
                    a,b=p[s],subject['pair'][s];common=np.intersect1d(a['x'][a['observed']],b['x'][b['observed']])
                    if len(common)<.08*width:matches=[];break
                    distance=abs(np.interp(common,a['x'],a['y'])-np.interp(common,b['x'],b['y']))
                    matches.append(float(np.mean(np.exp(-.5*(distance/(cfg['match_roi']*width))**2))))
                if matches:
                    newC=contain*float(np.mean(matches));oldbonus=.20*(p['c1']-.5) if p['c1_active'] else 0.
                    bonus=.20*(newC-.5)-oldbonus;q['score']+=bonus
        ranked.append(q);details.append(dict(ids=signature(p),base_score=p['score'],score=q['score'],MT_mm=p['MT_mm'],old_C=p['c1'],new_C=newC,side_match=matches,L_delta=bonus,extra_prior=extra))
    ranked.sort(key=sortkey);details.sort(key=lambda d:-d['score'])
    selected=deepcopy(ranked[0]) if ranked else None
    if selected:
        other=next((p for p in ranked[1:] if abs(p['MT_px']-selected['MT_px'])>.02*width),None)
        selected['margin']=float(selected['score']-other['score']) if other else None
        selected['ambiguous']=bool(other and selected['margin']<.025)
    return selected,dict(subject=au,candidates=details,margin=selected.get('margin') if selected else None,ambiguous=selected.get('ambiguous') if selected else None)
