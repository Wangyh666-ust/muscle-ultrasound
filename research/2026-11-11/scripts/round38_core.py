"""Round38 measurements and inverse-coordinate evaluation. No reference reads."""
import copy, itertools, math
import cv2,numpy as np
from scipy import ndimage
from scripts.round38_common import *
from scripts.round37_core import raw_components,filter_components,duplicate,sensitivity
from scripts.round37_run import scores
from scripts.round36_core import merge_pools,choose,measure as legacy_measure
from scripts.round35_decoder import identity_pool,map_curve
from scripts.round29_core import boundary_pool,value_at,line_at
from scripts.round27_experiment import boundary
from scripts.round28_core import select,accepted,region,REGIONS
from scripts.fascicle_pa_fl import select_candidates
from src.umud_baseline.apo_baseline import load_frozen_rule

def decode(masks,cfg):
    rule,_=load_frozen_rule();bcfg=j(ROOT/'manifest/round29_boundary_spatial_v1.json')['boundary']
    _,base=boundary(masks[0],rule,masks.shape[1]);p0=boundary_pool(masks[0],base,bcfg);p1=identity_pool(masks,bcfg)
    row=dict(height=masks.shape[1],pools={},scores={})
    for name,p in [('D0',p0),('D1',p1)]:
        row['pools'][name]=p['pairs'];row['scores'][name]={'scores':[{'pair_id':v['pair_id'],'quality':v['quality']} for v in p['pairs']]}
    pool=merge_pools(row,j(ROOT/'configs/round36_candidate_consensus.json')['merge'])
    diag={n:{'raw_tracks':p['original_track_count'],'tracks':p['track_count'],'pairs':len(p['pairs']),'fit_failures':[q['failure'] for q in p['pairs'] if q['curve'] is None]} for n,p in [('D0',p0),('D1',p1)]}
    return pool,diag

def state(mask,roi,cfg):
    ex,raw=raw_components(mask);segs,audit=filter_components(raw,roi,cfg['reference_filter'],'scale')
    labs,_=ndimage.label(mask,structure=np.ones((3,3),bool));evidence=np.isin(labs,[s['label_id'] for s in segs])
    return dict(segments=segs,evidence=evidence,audit=audit,roi=roi,raw_count=len(raw))

def base_measure(pool,st,row):
    sc=scores(pool,st['evidence']);p=choose(pool,sc,row);cv=p['curve'] if p else None
    out,cs=legacy_measure(cv,st['segments'],row,p['pair_id'] if p else None)
    for c,f in zip(out['candidates'],cs):
        c.update(PA=f['PA']['pa_deg'],intersection_sup=f['F0']['intersection_sup'],intersection_deep=f['F0']['intersection_deep'])
    out['candidate_scores']=sc;out['roi']=st['roi'];out['raw_components']=st['raw_count'];out['filter_count']=len(st['segments'])
    out['FL_weights']={str(i):w for i,w in zip(out['selection']['selected_ids'],out['selection']['weights'])}
    out['PA_weights']={str(i):1/len(out['PA_ids']) for i in out['PA_ids']}
    out['exploratory_values']={p:None for p in ('MT','PA','FL')};out['aggregation']=None
    return out,cs

def qualified(base,cs):
    out=copy.deepcopy(base);valid=[c for c in cs if not c['invalid'] and out['curve'] is not None]
    pa=[dict(segment=c['segment'],**c['PA'],intersection_deep=c['F0']['intersection_deep'],fl_accept_pure=False,fl_accept_strict=False) for c in valid]
    result=select_candidates(pa);out['PA_ids']=[c['segment']['label_id'] for c in result['pa']];out['PA_invalid']=[]
    out['selection']=select(valid,out['curve']['support'] if out['curve'] else None,'UW')
    out['values']['PA']=result['pa_mean_deg'];out['values']['FL']=out['selection']['value_px']
    out['FL_weights']={str(i):w for i,w in zip(out['selection']['selected_ids'],out['selection']['weights'])}
    out['PA_weights']={str(i):1/len(out['PA_ids']) for i in out['PA_ids']}
    out['exploratory_values']={p:base['values'][p] if out['values'][p] is None else None for p in ('MT','PA','FL')}
    out['exploratory_weights']={p:base[p+'_weights'] if out['values'][p] is None else {} for p in ('PA','FL')}
    out['qualified_count']=len(valid)
    return out

def weighted_median(x,w):
    order=np.argsort(x,kind='stable');k=np.searchsorted(np.cumsum(w[order]),w.sum()/2);return float(x[order[min(k,len(x)-1)]])

def huber_weights(x,weights,floor,cfg):
    x=np.asarray(x,float);w=np.asarray(weights,float);w=w/w.sum();center=weighted_median(x,w)
    scale=max(floor,1.4826*weighted_median(abs(x-center),w));final=w.copy()
    for _ in range(cfg['iterations']):
        residual=abs(x-center);factor=np.minimum(1.,cfg['huber_delta']*scale/np.maximum(residual,1e-12))
        final=w*factor;final/=final.sum();new=float(final@x)
        if abs(new-center)<1e-10:break
        center=new
    return final,float(final@x),scale

def representative_lines(reps,roi,cfg):
    lo,_,hi,_=roi['box'];span=max(hi-lo,1);buckets=[[None]+[c for c in reps if region(c['segment']['centroid'][0],[lo,hi])==b] for b in range(3)]
    best=None;chosen=[]
    for comb in itertools.product(*buckets):
        cc=[c for c in comb if c is not None]
        if any(abs(a['segment']['centroid'][0]-b['segment']['centroid'][0])<cfg['min_rep_spacing_roi']*span for a,b in itertools.combinations(cc,2)):continue
        rank=(-sum(cfg['region_weights'][i] for i,c in enumerate(comb) if c is not None),sum(abs(c['segment']['centroid'][0]-(lo+cfg['anchors'][i]*span))/span for i,c in enumerate(comb) if c is not None),tuple(c['label_id'] if c else -1 for c in comb))
        if best is None or rank<best:best=rank;chosen=cc
    return chosen

def aggregate(base,cs,row,cfg):
    out=copy.deepcopy(base);cv=out['curve'];ac=cfg['aggregate'];roi=out['roi'];lo,_,hi,_=roi['box'];span=max(hi-lo,1)
    if cv is None:return out
    valid=[c for c in cs if not c['invalid'] and region(c['segment']['centroid'][0],[lo,hi]) is not None]
    reps=[];groups=[]
    for c in sorted(valid,key=lambda c:(-c['segment']['length_px'],c['label_id'])):
        ix=next((i for i,r in enumerate(reps) if duplicate(c['segment'],r['segment'],span,cfg['reference_filter'])),None)
        if ix is None:reps.append(c);groups.append([c['label_id']])
        else:groups[ix].append(c['label_id'])
    qs={};risks={}
    for c in reps:
        s=c['segment'];sid=str(c['label_id']);d=sensitivity(s,cv,row,roi,cfg['reference_filter']);risks[sid]=d
        density=min(1.,s['area_px']/max(s['length_px']*max(s['width_px'],1),1))
        q=min(s['length_px']/span,ac['length_cap_roi'])*max(density,ac['weight_floor'])*s['linearity']
        qs[sid]=q/(1+ac['quality_sensitivity_weight']*d['risk'])
    diag=dict(groups=groups,quality=qs,sensitivity=risks,metrics={})
    for metric in ('FL','PA'):
        rr=[c for c in reps if accepted(c,'U')[0]] if metric=='FL' else [c for c in reps if c['PA']['pa_deg'] is not None]
        binned=[region(c['segment']['centroid'][0],[lo,hi]) for c in rr];active=sorted(set(binned));weights={};regions=[]
        if rr:
            xx=np.array([(c['segment']['centroid'][0]-lo)/span for c in rr]);kernel=np.exp(-.5*((xx[:,None]-np.array(ac['anchors'])[None,:])/ac['spatial_sigma'])**2)
            kernel[:,[i for i in range(3) if i not in active]]=0;kernel/=kernel.sum(1)[:,None]
            vals=np.array([c['F0']['fl_px_raw'] if metric=='FL' else c['PA']['pa_deg'] for c in rr]);quality=np.array([qs[str(c['label_id'])] for c in rr])
            rw=np.array([ac['region_weights'][i] if i in active else 0 for i in range(3)]);rw/=rw.sum();final=np.zeros(len(rr))
            floor=ac['scale_floor_FL_roi']*span if metric=='FL' else ac['scale_floor_PA_deg']
            for b in active:
                ww,v,sc=huber_weights(vals,quality*kernel[:,b],floor,ac);final+=rw[b]*ww
                regions.append(dict(region=REGIONS[b],own_evidence_ids=[c['label_id'] for c,bb in zip(rr,binned) if bb==b],value=v,scale=sc))
            weights={str(c['label_id']):float(w) for c,w in zip(rr,final)};out['values'][metric]=float(final@vals)
            disp=float(np.sqrt(final@((vals-final@vals)**2)));effective=float(1/np.sum(final**2))
        else:out['values'][metric]=None;disp=None;effective=0.
        out[metric+'_weights']=weights
        diag['metrics'][metric]=dict(regions=regions,missing_regions=[REGIONS[i] for i in range(3) if i not in active],contributing_ids=[c['label_id'] for c in rr],dispersion=disp,effective_tracks=effective)
    flreps=[c for c in reps if accepted(c,'U')[0]];chosen=representative_lines(flreps,roi,ac)
    selected=select(chosen,[lo,hi],'UW');selected['value_px']=out['values']['FL'];selected['variant']='per_track_robust_measurements'
    selected['missing_regions']=diag['metrics']['FL']['missing_regions'];selected['three_regions_complete']=not selected['missing_regions']
    selected['representative_only']=True;selected['contributing_ids']=diag['metrics']['FL']['contributing_ids']
    selected['representative_missing_regions']=[REGIONS[i] for i in range(3) if i not in [region(c['segment']['centroid'][0],[lo,hi]) for c in chosen]]
    out['selection']=selected;out['PA_ids']=[int(k) for k in out['PA_weights']];out['PA_invalid']=[];out['aggregation']=diag
    return out

def to_original(out,t,original):
    """Only map selected measurements; never re-decode, re-rank or use original GT."""
    q=copy.deepcopy(out);iv=inverse_spec(t);scale=original.get('px_per_mm') or 1.;cv=map_curve(out['curve'],iv);q['curve']=cv
    by={}
    for c in q['candidates']:
        s=map_segment(c['segment'],iv);c['segment']=s
        for part in ('intersection_sup','intersection_deep'):
            if c[part] is not None:c[part]=map_point(c[part],iv)
        if c['raw_FL'] is not None and c['intersection_sup'] is not None and c['intersection_deep'] is not None:c['raw_FL']=float(np.linalg.norm(np.subtract(c['intersection_sup'],c['intersection_deep'])))
        if c['PA'] is not None and cv:
            angle=math.degrees(math.atan(line_at(cv['deep'],s['centroid'][0])[0]));delta=abs(s['direction_deg']-angle)%180;c['PA']=float(min(delta,180-delta))
        by[str(s['label_id'])]=c
    for metric,field in [('PA','PA'),('FL','raw_FL')]:
        weights=out[metric+'_weights'];value=sum(w*by[k][field] for k,w in weights.items()) if weights else None
        q['values'][metric]=value/(scale if metric=='FL' else 1) if value is not None else None
        ew=out.get('exploratory_weights',{}).get(metric,{})
        if ew:q['exploratory_values'][metric]=sum(w*by[k][field] for k,w in ew.items())/(scale if metric=='FL' else 1)
    q['values']['MT']=out['values']['MT']/t['sy']/scale if out['values']['MT'] is not None else None
    q['selection']['value_px']=q['values']['FL']*scale if q['values']['FL'] is not None else None
    q['mt_positions']=[x*iv['sx']+iv['tx'] for x in out['mt_positions']]
    box=out['roi']['box'];q['roi']={**out['roi'],'box':map_point(box[:2],iv)+map_point(box[2:],iv)}
    q['stage']='boundary_failure' if cv is None else 'output' if all(v is not None for v in q['values'].values()) else 'partial_or_missing'
    if q['aggregation']:
        for p in ('FL','PA'):
            weights=q[p+'_weights'];values=[by[k]['raw_FL']/scale if p=='FL' else by[k]['PA'] for k in weights]
            q['aggregation']['metrics'][p]['dispersion_original_units']=float(np.sqrt(sum(w*(v-q['values'][p])**2 for w,v in zip(weights.values(),values)))) if values else None
    return q

def run_arms(pools,states,row,cfg,original,t):
    defs={'R':(0,0,False,False),'G':(0,0,True,False),'N':(0,2,True,False),'V':(0,1,True,False),
          'A':(0,0,True,True),'NV':(0,3,True,False),'NVA':(0,3,True,True),'BNVA':(1,3,True,True)}
    cache={};out={}
    for arm,(bi,fi,guard,agg) in defs.items():
        if (bi,fi) not in cache:cache[bi,fi]=base_measure(pools[bi],states[fi],row)
        base,cs=cache[bi,fi];a=qualified(base,cs) if guard else base
        if agg:a=aggregate(a,cs,row,cfg)
        out[arm]=to_original(a,t,original)
    return out
