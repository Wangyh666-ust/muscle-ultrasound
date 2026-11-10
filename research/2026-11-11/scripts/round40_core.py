"""Prespecified qualification, smooth rays, boundary-to-FL sensitivity. No GT."""
from scripts.round39_core import *
from scripts.round29_core import extended_fl
from scripts.round28_core import accepted,region

def smooth_extensions(pool,roi):
    out=copy.deepcopy(pool);span=roi['box'][2]-roi['box'][0]
    for p in out:
        a=np.asarray(p['observations']);cv=p['curve'];p['extension_diagnostics']=[]
        for col,side in ((1,'sup'),(2,'deep')):
            for right,name in ((False,'left_ray'),(True,'right_ray')):
                at=-1 if right else 0;x=a[at,0];w=np.exp(-.5*((a[:,0]-x)/(.25*span))**2)
                xx=a[:,0]-x;A=np.column_stack([xx,np.ones(len(xx))]);fit=np.linalg.lstsq(A*np.sqrt(w[:,None]),a[:,col]*np.sqrt(w),rcond=None)[0]
                slope=float(fit[0]);cv[side][name]=[slope,float(cv[side]['knots_y'][at]-slope*x)]
                p['extension_diagnostics'].append(dict(side=side,ray=name,observed_points=len(a),bandwidth=.25*span,effective_points=float(w.sum()**2/(w@w))))
    return out

def evidence_gates(pool,st,row,cfg,cal,oldgates):
    ev=prepare_evidence(st['raw_mask'],st['segments'],row['width'],row['height'],MCONF['c1']);out=[]
    lo,_,hi,_=st['roi']['box'];span=hi-lo;ac=cfg['aggregate'];rw=np.array(ac['region_weights'])
    for p,g in zip(pool,oldgates):
        cv=p['curve'];tracks=[];fs=[[],[],[]];m=np.zeros(3);numer=np.zeros(3)
        for s,group,wt in ev['tracks']:
            b=region(s['centroid'][0],[lo,hi])
            if b is None:continue
            xx,yy=ev['pixels'][s['label_id']];obs=support(cv,xx);o=float(obs.mean()) if len(obs) else 0.
            c=float(((yy[obs]>=interpolated_y(cv['sup'],xx[obs]))&(yy[obs]<=interpolated_y(cv['deep'],xx[obs]))).mean()) if obs.any() else None
            if o>=.5:fs[b].append((c,wt))
            density=min(1.,s['area_px']/max(s['length_px']*max(s['width_px'],1),1))
            qual=min(s['length_px']/span,ac['length_cap_roi'])*max(density,ac['weight_floor'])*s['linearity']
            kernel=np.exp(-.5*(((s['centroid'][0]-lo)/span-np.array(ac['anchors']))/ac['spatial_sigma'])**2);kernel/=kernel.sum()
            mass=qual*o*kernel;m+=mass;numer+=mass*(c if c is not None else 0.)
            tracks.append(dict(id=s['label_id'],members=group,observable_fraction=o,conditional_inside=c,region=b,base_quality=qual))
        active=[b for b in range(3) if fs[b]];n=sum(len(v) for v in fs);hard=n>=3 and len(active)>=2
        c=sum(rw[b]*np.average([v for v,w in fs[b]],weights=[w for v,w in fs[b]]) for b in active)/sum(rw[b] for b in active) if active else None
        structural=[r for r in g['reasons'] if r!='confident_C1_contradiction'];reasons=structural+(['confident_C1_contradiction'] if hard and c<.6 else [])
        reliability=m/(m+cal['lambda_evidence']);cb=np.divide(numer,m,out=np.zeros(3),where=m>0)
        factor=float(rw@((1-reliability)+reliability*cb));score=p['old_quality']*factor
        out.append(dict(pair_id=p['pair_id'],allowed=not reasons,reasons=reasons,C1_gate=hard,C1_containment=c,tracks=tracks,independent_tracks=n,
            structural_allowed=not structural,structural_reasons=structural,soft_score=score,boundary_quality=p['old_quality'],soft_factor=factor,
            regional_mass=m.tolist(),regional_reliability=reliability.tolist(),regional_conditional_inside=cb.tolist(),low_evidence=not hard))
    return out

def measure_soft(pool,st,row,cfg,gates):
    sc=scores(pool,st['evidence']);lookup={g['pair_id']:g for g in gates}
    ranked=[{**s,'old_score':lookup[s['pair_id']]['soft_score']} for s in sc if lookup[s['pair_id']]['structural_allowed']]
    rr={**row,'height':pool[0].get('ranking_height',row['height'])} if pool else row
    picked=choose(pool,ranked,rr);base,cs=base_measure([picked] if picked else [],st,row);out=aggregate(qualified(base,cs),cs,row,cfg)
    out['candidate_scores']=ranked;out['pair_eligibility']=gates
    ranked=sorted(ranked,key=lambda x:-x['old_score']);out['qualification_diagnostic']=dict(low_evidence=lookup[picked['pair_id']]['low_evidence'] if picked else True,
        best_second_score_margin=(ranked[0]['old_score']-ranked[1]['old_score']) if len(ranked)>1 else None,eligible_pairs=len(ranked),forced_boundary=False)
    return out

def perturb_boundary(cv,side,offset=0.,ray=None,angle=0.):
    q=copy.deepcopy(cv);f=q[side]
    if offset:
        f['knots_y']=[y+offset for y in f['knots_y']]
        for r in ('left_ray','right_ray'):f[r][1]+=offset
    if ray:
        at=0 if ray=='left_ray' else -1;x=f['knots_x'][at];y=f['knots_y'][at]
        slope=math.tan(math.atan(f[ray][0])+math.radians(angle));f[ray]=[slope,y-slope*x]
    return q

def propagated_risk(seg,cv,row,span,cfg,cal,base):
    probes=[];rf=cfg['reference_filter'];step=rf['sensitivity']['endpoint_reference_pixels']*span/rf['filter']['reference_roi_width']
    for da in rf['sensitivity']['angle_offsets']:
        s=copy.deepcopy(seg);s['direction_deg']=(s['direction_deg']+da)%180;probes.append(('C1_angle',s,cv))
    normal=np.array([-math.sin(math.radians(seg['direction_deg'])),math.cos(math.radians(seg['direction_deg']))])
    for sign in (-1,1):
        s=copy.deepcopy(seg)
        for k in ('p0','p1','centroid'):s[k]=(np.array(s[k])+sign*step*normal).tolist()
        probes.append(('C1_shift',s,cv))
    for side in ('sup','deep'):
        for sign in (-1,1):probes.append((side+'_shift',seg,perturb_boundary(cv,side,offset=sign*cal['boundary_offset_roi']*span)))
        for ray in ('left_ray','right_ray'):
            for da in cal['boundary_angle_offsets_deg']:probes.append((side+'_'+ray,seg,perturb_boundary(cv,side,ray=ray,angle=da)))
    vals=[];failures={};root_switches=0;center=extended_fl(seg,cv,row['width'],row['height'])
    for name,s,curve in probes:
        f=extended_fl(s,curve,row['width'],row['height']);ok,reasons=accepted({'F0':f},'U')
        if ok and f['fl_px_raw'] is not None and np.isfinite(f['fl_px_raw']):vals.append(float(f['fl_px_raw']))
        else:
            for reason in reasons:failures[reason]=failures.get(reason,0)+1
        root_switches+=any([r.get('piece') for r in f.get('roots',{}).get(side,{}).get('roots',[])]!=[r.get('piece') for r in center.get('roots',{}).get(side,{}).get('roots',[])] for side in ('sup','deep'))
    n=len(probes);lost=n-len(vals);u=max((abs(v-base)/span for v in vals),default=1.);p=lost/n
    return dict(u=u,p=p,n=n,valid=len(vals),lost=lost,unresolved=not vals,risk_factor=1/(1+(u/.03)**2+2*p),failures=failures,root_switches=root_switches,
        min_probe_FL=min(vals) if vals else None,max_probe_FL=max(vals) if vals else None,reliable=bool(vals and u<=.03 and lost==0))

def apply_u(out,cfg,cal,cache=None):
    q=copy.deepcopy(out)
    if q['curve'] is None or not q['aggregation']:return q
    ids,vals,quality,kernel,active,span=fl_inputs(q,cfg,True)
    if not ids:return q
    by={c['segment']['label_id']:c for c in q['candidates']};diag={}
    row=dict(width=q['roi'].get('image_width',1),height=1)
    # Width/height are used by the frozen geometry's image-bound checks.
    row=q['_native_dimensions']
    for i,v in zip(ids,vals):diag[str(i)]=propagated_risk(by[i]['segment'],q['curve'],row,span,cfg,cal,float(v))
    adjusted=quality*np.array([diag[str(i)]['risk_factor'] for i in ids]);final,regions=combine_fl(vals,adjusted,kernel,active,span,cfg['aggregate'],cal,True,True)
    q['values']['FL']=float(final@vals);q['FL_weights']={str(i):float(w) for i,w in zip(ids,final)};q['selection']['value_px']=q['values']['FL']
    q['aggregation']['metrics']['FL'].update(regions=regions,dispersion=float(np.sqrt(final@((vals-final@vals)**2))),effective_tracks=float(1/(final@final)))
    q['boundary_sensitivity']=dict(tracks=diag,all_tracks_unreliable=not any(d['reliable'] for d in diag.values()),unreliable_weight=sum(w for i,w in q['FL_weights'].items() if not diag[i]['reliable']))
    return q
