"""Full-scope frozen-geometry completion and actual-polyline measurement estimates."""
from copy import deepcopy
import cv2,time
from scripts.round71_common import *
from scripts.round71_geometry import complete
from scripts.round71_prepare import view_mask
from scripts.round70_geometry import guard_edit
from scripts.round63_measure import geometry,mapped_segments,reference_score,aggregate


def normalize(p,confidence):
    q=deepcopy(p)
    for k in ('x','y','observed','kind'):q[k]=np.asarray(q[k],bool if k=='observed' else int if k=='kind' else float)
    q['confidence']=np.asarray(confidence,float);return q


def base_pair(old,old69):
    source=old['arms']['W']['pair'];proposal=old['arms']['WJ']['pair'];audit={}
    if source is None:return None,dict(reason='no_original_pair')
    result=dict(interval=list(source['interval']),width=old['width'])
    for s in SIDES:
        q=old69['arms']['WM']['eligibility'][s];p=normalize(source[s],q['fused_probability'])
        assert np.array_equal(p['x'],old69['arms']['W0']['pair'][s]['x']) and np.max(abs(p['y']-old69['arms']['W0']['pair'][s]['y']))<1e-8
        fused,role=np.asarray(q['fused_probability']),np.asarray(q['role_probability'])
        W=(p['kind']==3)&~p['observed'];p['anchor_eligible']=p['observed']|(W&(fused>=.65)&(role>=.65))
        p['protected_W']=W&(fused>=.5)&(role>=.5)
        events=[e for e in proposal[s]['events'] if e['type']=='R68_J']
        accept=True;reasons=[]
        for event in events:
            allowed,reason=guard_edit(p,proposal[s],event,old['width']);accept&=allowed;reasons.append(reason)
        candidate=deepcopy(proposal[s]) if events and accept else deepcopy(p)
        candidate=normalize(candidate,np.interp(candidate['x'],p['x'],p['confidence']))
        pos=np.searchsorted(p['x'],candidate['x']);valid=pos<len(p['x']);exact=np.zeros(len(pos),bool)
        exact[valid]=(abs(p['x'][pos[valid]]-candidate['x'][valid])<1e-8)&(abs(p['y'][pos[valid]]-candidate['y'][valid])<1e-8)
        candidate['anchor_eligible']=np.zeros(len(pos),bool);candidate['anchor_eligible'][exact]=p['anchor_eligible'][pos[exact]]
        candidate['source_qualified']=candidate['observed'].copy()
        candidate['source_qualified'][exact]|=(p['protected_W'][pos[exact]])
        result[s]=candidate;audit[s]=dict(J_present=bool(events),J_accepted=bool(events and accept),decisions=reasons,
             anchors=int(candidate['anchor_eligible'].sum()),new_J_points_not_used_as_anchors=int((~exact).sum()))
    result['interval']=[max(result[s]['x'][0] for s in SIDES),min(result[s]['x'][-1] for s in SIDES)]
    return result,audit


def local_values(pair,mask,segments,scale,width,uncertainty_cutoff,qualified):
    if pair is None:return dict(values=dict(MT=None,PA=None,FL=None),finite_values=dict(MT=None,PA=None,FL=None),
          weights={},qualified_values=dict(MT=None,PA=None,FL=None),contributors=[],reason='no_pair')
    g=geometry(pair,segments,scale,width);finite=dict(MT=g['MT'],PA=g['PA'],FL=g['FL_finite'])
    values=dict(MT=g['MT'],PA=g['PA'],FL=g['FL_ray'])
    xx=pair['interval'][0]+np.asarray([.25,.5,.75])*np.ptp(pair['interval'])
    support_ok=True
    for s in SIDES:
        p=pair[s];support_ok&=bool((np.interp(xx,p['x'],p['qualified'].astype(float))>.999).all())
    qvalues=dict(MT=finite['MT'] if support_ok else None,PA=None,FL=None);contributors=[]
    for c in g['contributors']:
        row=deepcopy(c);row['PA_qualified']=False;row['FL_qualified']=False
        lower=pair['lower'];cx=float(c['centroid'][0]);row['PA_qualified']=bool(np.interp(cx,lower['x'],lower['qualified'].astype(float))>.999)
        if c.get('finite_points'):
            flags=[];info=[]
            for s,point in zip(SIDES,c['finite_points']):
                p=pair[s];x,y=point;qual=bool(np.interp(x,p['x'],p['qualified'].astype(float))>.999)
                flags.append(qual);info.append(dict(side=s,point=point,uncertainty=float(np.interp(x,p['x'],p['uncertainty'])),kind=int(round(np.interp(x,p['x'],p['kind']))),qualified=qual))
            row['FL_qualified']=all(flags);row['intersection_audit']=info
        contributors.append(row)
    for name,field,flag in [('PA','PA','PA_qualified'),('FL','FL_finite','FL_qualified')]:
        filtered=[{**c,field:c[field] if c[flag] else None} for c in contributors]
        qvalues[name],_=aggregate(filtered,field)
    # A source-wide crossing is a structural failure for all qualified estimates.
    probes=np.linspace(*pair['interval'],max(2,int(np.ptp(pair['interval']))+1))
    ordered=bool(np.all(np.interp(probes,pair['upper']['x'],pair['upper']['y'])<np.interp(probes,pair['lower']['x'],pair['lower']['y'])))
    if not ordered:qvalues=dict(MT=None,PA=None,FL=None)
    return dict(values=values,finite_values=finite,qualified_values=qvalues,weights=g['weights'],contributors=contributors,
                unit=g.get('unit','px'),ordered=ordered,measurement='exact completed polyline, R63 geometry; exploratory, not default formal',
                qualified_uses_finite_intersections_only=True,MT_positions=xx)


def work(rec,domain_entry,lock):
    cv2.setNumThreads(1);tic=time.monotonic();relative=Path(rec['dataset'])/Path(rec['image_id']).stem/(rec['view']+'.json.gz')
    dest=OUT/'records'/relative
    if dest.exists():
        r=readgz(dest);assert r['lock_sha']==sha(OUT/'parameter_lock.json');return dict(key=rec['key'],view=rec['view'],resumed=True)
    old,old69=readgz(R68/'records'/relative),readgz(R69/'records'/relative)
    with np.load(ROOT/domain_entry['path']) as z:
        shape=tuple(z['shape']);mask=np.unpackbits(z['bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)
    native=view_mask(mask,rec,old['shape']);yy,xx=np.where(native)
    domain=[float(xx.min()),float(xx.max())] if len(xx) else [0.,float(old['shape'][1]-1)]
    width=max(1.,np.ptp(domain));base,ba=base_pair(old,old69);metadata=j(R63/'metadata.json')[rec['key']]
    segments=mapped_segments(metadata,rec);scale=metadata['px_per_mm']*rec['matrix'][0][0] if metadata['px_per_mm'] else None
    row={k:old[k] for k in ('key','dataset','image_id','view','manual','partition','group','input','width','shape')}
    row.update(arms={},domain=domain,domain_meta=domain_entry['meta'],domain_sha=domain_entry['sha'],base_audit=ba,
               lock_sha=sha(OUT/'parameter_lock.json'),source_sha=sha(R68/'records'/relative),C1_segments_fixed=len(segments))
    if rec['manual']:row['referenceLines']=old['referenceLines']
    for arm in ARMS:
        pair=None;audits={};geo={}
        if base is not None:
            pair=dict(interval=[max(domain[0],max(base[s]['x'][0] for s in SIDES)),min(domain[1],min(base[s]['x'][-1] for s in SIDES))],width=width)
            for side_index,s in enumerate(SIDES):
                if arm=='B':
                    p=deepcopy(base[s]);p.update(uncertainty=np.zeros(len(p['x'])),existing=np.ones(len(p['x']),bool),qualified_source=p['source_qualified'].copy());au=dict(reason='frozen_protected_baseline')
                else:p,au=complete(base[s],domain,width,lock['selected'],gaps=arm in ('G','GE'),ends=arm in ('E','GE'))
                if p is None:pair=None;audits[s]=au;break
                x,y=np.asarray(p['x']),np.asarray(p['y']);kind=np.asarray(p['kind']);new=~np.asarray(p['existing'])
                distance=np.zeros(len(x));left=x<base[s]['x'][0];right=x>base[s]['x'][-1]
                distance[left]=base[s]['x'][0]-x[left];distance[right]=x[right]-base[s]['x'][-1]
                gap=new&~left&~right
                if gap.any():distance[gap]=np.min(abs(x[gap,None]-base[s]['x'][None]),axis=1)
                uncertainty=np.asarray(p['uncertainty']).copy()
                for b in lock['uncertainty_multiplier'][str(side_index)]:
                    lower=(distance/width>b['lo']) if b['lo'] else (distance>=0)
                    inside=new&lower&(distance/width<=b['hi']);uncertainty[inside]*=b['multiplier']
                # Calibration bins can differ; reporting confidence must never improve merely by moving farther away.
                for end in (left,right):
                    indices=np.flatnonzero(new&end);order=indices[np.argsort(distance[indices])]
                    uncertainty[order]=np.maximum.accumulate(uncertainty[order])
                p['uncertainty']=uncertainty
                inside=(x>=0)&(x<native.shape[1])&(y>=0)&(y<native.shape[0]);onmask=np.zeros(len(x),bool)
                ix=np.clip(np.rint(x).astype(int),0,native.shape[1]-1);iy=np.clip(np.rint(y).astype(int),0,native.shape[0]-1);onmask[inside]=native[iy[inside],ix[inside]]
                cutoff=2*max(old['shape'])/512.
                qualified=(uncertainty<=cutoff)&onmask
                loc=np.searchsorted(x,base[s]['x']);assert np.array_equal(x[loc],base[s]['x']) and np.array_equal(y[loc],base[s]['y'])
                qualified[loc]&=base[s]['source_qualified'];p['qualified']=qualified;pair[s]=p
                columns=np.arange(int(np.ceil(domain[0])),int(np.floor(domain[1]))+1);covered=np.isin(columns,np.rint(x).astype(int))
                geo[s]=dict(complete=bool(covered.all()),target_columns=len(columns),covered_columns=int(covered.sum()),
                      target_coverage=float(covered.mean()),new_columns=int(new.sum()),new_gap_columns=int((kind==4).sum()),new_extension_columns=int((kind==5).sum()),
                      qualified_new=int((qualified&new).sum()),existing_preserved=True,onmask_fraction=float(onmask.mean()),
                      new_outside_domain_mask=int((new&~onmask).sum()),uncertainty_max=float(uncertainty.max(initial=0)))
                audits[s]=au
            if pair:
                pair['interval']=[max(pair[s]['x'][0] for s in SIDES),min(pair[s]['x'][-1] for s in SIDES)]
                if pair['interval'][1]<=pair['interval'][0]:pair=None
        measured=local_values(pair,native,segments,scale,width,None,None)
        result=dict(pair=pair,completion=geo,audit=audits,measurement=measured)
        if rec['manual']:
            class Ref:
                def __getitem__(self,key):
                    if key=='x':return np.asarray(old['referenceLines'][0]['x'])
                    if key=='ys':return np.asarray([r['y'] for r in old['referenceLines']])
                    if key=='radius':return old['arms']['S']['reference']['upper']['radius']
                    if key=='reference_width':return old['width']
                    raise KeyError(key)
            result['reference']=reference_score(pair,Ref())
        row['arms'][arm]=result
    # Same actual x coordinates for a boundary-only MT comparison; no extrapolated np.interp outside baseline.
    if base:
        pos=np.linspace(*base['interval'],5)[1:-1];row['fixed_MT_positions']=pos;row['fixed_MT']={}
        for arm in ARMS:
            pair=row['arms'][arm]['pair']
            row['fixed_MT'][arm]=float(np.mean(np.interp(pos,pair['lower']['x'],pair['lower']['y'])-np.interp(pos,pair['upper']['x'],pair['upper']['y']))/(scale or 1)) if pair else None
    row['seconds']=time.monotonic()-tic;write(dest,row)
    return dict(key=rec['key'],view=rec['view'],resumed=False,seconds=row['seconds'])
