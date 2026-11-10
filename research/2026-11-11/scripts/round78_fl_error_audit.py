"""Read-only FL attribution on sealed R78/R77 predictions. Oracle calculations are diagnostics only."""
from pathlib import Path
from collections import Counter
import json,math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.round78_mt_replay import ROOT,OUT as R78,R77,SEEDS,j,write,readgz,sha
from scripts.round29_core import extended_fl,line_at
OUT=ROOT/'review/round78_FL_error_audit_20261110'

def expert_boundary(geo,offset):
    cv={}
    for side,role in [('upper','sup'),('lower','deep')]:
        p=np.asarray(geo[side]['points'],float)-[offset,0]
        slopes=np.diff(p[:,1])/np.diff(p[:,0]);assert np.all(np.diff(p[:,0])>0)
        cv[role]=dict(knots_x=p[:,0].tolist(),knots_y=p[:,1].tolist(),support=[p[0,0],p[-1,0]],
            left_ray=[slopes[0],p[0,1]-slopes[0]*p[0,0]],right_ray=[slopes[-1],p[-1,1]-slopes[-1]*p[-1,0]])
    lo=max(cv[s]['support'][0] for s in ('sup','deep'));hi=min(cv[s]['support'][1] for s in ('sup','deep'))
    cv.update(support=[lo,hi],paired_x=[],max_sample_gap=12)
    return cv

def expert_trace_diagnostic(geo,offset,curve,w,h):
    """Fit straight lines to the same expert trajectories in the visible ROI, not a new predictor."""
    out=[];x0,_,x1,_=geo['scale']['roi'];scale=geo['scale']['px_per_mm']
    for trace in geo['curves']:
        p=np.asarray(trace['points'],float)-[offset,0];full=np.asarray(trace['full_points'],float)-[offset,0]
        # Label vertices are sparse. Interpolate first, rather than fitting the few vertices in the ROI.
        xx=np.linspace(max(x0,p[0,0]),min(x1,p[-1,0]),201);yy=np.interp(xx,p[:,0],p[:,1]);xy=np.column_stack([xx,yy])
        center=xy.mean(0);_,_,v=np.linalg.svd(xy-center,full_matrices=False);d=v[0];d=d if d[0]>=0 else -d
        seg=dict(centroid=center.tolist(),direction_deg=float(np.degrees(np.arctan2(d[1],d[0]))),p0=xy[0].tolist(),p1=xy[-1].tolist(),length_px=float(np.linalg.norm(xy[-1]-xy[0])))
        f=extended_fl(seg,curve,w,h);raw=f.get('fl_px_raw')
        out.append(dict(FL_reference=trace['FL_mm'],reference_PA=trace['PA_deg'],full_chord_mm=float(np.linalg.norm(full[-1]-full[0])/scale),
            arclength_minus_chord_mm=float(trace['FL_mm']-np.linalg.norm(full[-1]-full[0])/scale),
            visible_expert_straight_angle=seg['direction_deg'],visible_expert_straight_with_expert_boundary_FL=None if raw is None else raw/scale))
    return out

def evaluate(r,src,g,seed,geo):
    row=r['row'];o=r['outputs'][seed]['M'];cv=src['curves']['P'];pred=o['values']['FL'];ref=row['reference']['FL'];scale=row['px_per_mm'];by={str(s['label_id']):s for s in src['segments'][seed]}
    ecv=expert_boundary(geo,row['annotation_offset']) if geo else None;ts=[]
    for saved in g['tracks'][seed]['P']:
        sid=str(saved['id']);w=float(o['FL_weights'].get(sid,0))
        if w<=0:continue
        s=by[sid];f=extended_fl(s,cv,row['width'],row['height']);raw=f.get('fl_px_raw');assert raw is not None and abs(raw/scale-saved['FL'])<1e-7
        roots={};outside=False
        for role,side in [('sup','upper'),('deep','lower')]:
            root=f['roots'][role]['roots'][0];x,y=root['point'];p=src['pairs']['P'][side];q=np.asarray(p['qualified'],bool)
            distance=float(np.min(np.hypot(np.asarray(p['x'])[q]-x,np.asarray(p['y'])[q]-y))/scale) if q.any() else None
            outside=outside or not (0<=x<row['width'] and 0<=y<row['height'])
            roots[side]=dict(point=root['point'],piece=root['piece'],boundary_angle_deg=float(np.degrees(np.arctan(root['line'][0]))),distance_to_qualified_mm=distance)
        oracle=extended_fl(s,ecv,row['width'],row['height']) if ecv else None;oracle_raw=oracle.get('fl_px_raw') if oracle else None
        local_angle=np.degrees(np.arctan(line_at(cv['deep'],s['centroid'][0])[0]));pa=abs((s['direction_deg']-local_angle+90)%180-90)
        ts.append(dict(id=sid,weight=w,FL=raw/scale,direction_deg=s['direction_deg'],local_PA_deg=float(pa),length_visible_mm=s['length_px']/scale,
            extrapolation_ratio=f.get('extrapolation_ratio'),roots=roots,any_root_outside_image=outside,max_root_support_distance_mm=max((z['distance_to_qualified_mm'] for z in roots.values() if z['distance_to_qualified_mm'] is not None),default=None),
            risk=o['boundary_sensitivity']['tracks'].get(sid),fixed_weight_expert_boundary_FL=None if oracle_raw is None else oracle_raw/scale,
            oracle_failure=None if oracle_raw is not None else (oracle.get('fl_failure_conditions') if oracle else 'no_expert_boundaries')))
    assert (pred is None)==(not ts)
    if ts:
        assert abs(sum(t['weight'] for t in ts)-1)<1e-8;assert abs(sum(t['weight']*t['FL'] for t in ts)-pred)<1e-7
    for t in ts:
        t['FL_without_this_track_fixed_other_weights']=(pred-t['weight']*t['FL'])/(1-t['weight']) if t['weight']<1-1e-8 else None
        t['removal_change_mm']=None if t['FL_without_this_track_fixed_other_weights'] is None else t['FL_without_this_track_fixed_other_weights']-pred
    oracle_complete=bool(ts) and all(t['fixed_weight_expert_boundary_FL'] is not None for t in ts)
    oracle=sum(t['weight']*t['fixed_weight_expert_boundary_FL'] for t in ts) if oracle_complete else None
    longest=max(ts,key=lambda t:t['FL']) if ts else None;flattest=min(ts,key=lambda t:abs((t['direction_deg']+90)%180-90)) if ts else None
    return dict(values=o['values'],reference=row['reference'],FL_error=None if pred is None else pred-ref,FL_absolute_error=None if pred is None else abs(pred-ref),tracks=ts,
        n_tracks=len(ts),effective_tracks=1/sum(t['weight']**2 for t in ts) if ts else 0,weight_outside_image=sum(t['weight'] for t in ts if t['any_root_outside_image']),
        weight_both_boundary_rays=sum(t['weight'] for t in ts if all(isinstance(z['piece'],str) and 'ray' in z['piece'] for z in t['roots'].values())),
        longest_id=longest['id'] if longest else None,flattest_id=flattest['id'] if flattest else None,
        without_longest=longest['FL_without_this_track_fixed_other_weights'] if longest else None,
        without_flattest=flattest['FL_without_this_track_fixed_other_weights'] if flattest else None,
        fixed_weights_expert_boundary_FL=oracle,oracle_complete=oracle_complete,oracle_missing_weight=sum(t['weight'] for t in ts if t['fixed_weight_expert_boundary_FL'] is None),
        diagnostic_only=True,source_PA_FL_unchanged=True)

def figure(r,src,g,audit,seed):
    row=r['row'];sid=row['image_id'];a=audit[seed];ts=a['tracks'];cv=src['curves']['P'];by={str(s['label_id']):s for s in src['segments'][seed]};geo=j(ROOT/f'validation/young_expert_v03/geometry/{sid}.json') if row['dataset']=='young78' else None
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.5,5.5),gridspec_kw=dict(width_ratios=[1.05,1.]));ax.imshow(Image.open(ROOT/row['image_path']).convert('L'),cmap='gray')
    if cv:
        for role,c in [('sup','#20c9ff'),('deep','#ff5db5')]:ax.plot(cv[role]['knots_x'],cv[role]['knots_y'],color=c,lw=1.7)
    for p in g['reference_lines'] or []:
        p=np.asarray(p);ax.plot(p[:,0],p[:,1],'w--',lw=1,alpha=.8)
    if geo:
        for c in geo['curves']:
            p=np.asarray(c['full_points'])-[row['annotation_offset'],0];ax.plot(p[:,0],p[:,1],color='#e6e6a9',ls=':',lw=.9,alpha=.8)
    important={t['id'] for t in sorted(ts,key=lambda t:t['weight'],reverse=True)[:3]};important.add(a['longest_id'])
    for t in ts:
        s=by[t['id']];p=np.asarray([s['p0'],s['p1']]);color='#fcaa46' if t['id']==a['flattest_id'] else '#3def9a'
        ax.plot(p[:,0],p[:,1],color=color,lw=1+5*t['weight'])
        if t['id'] in important:
            u,d=t['roots']['upper']['point'],t['roots']['lower']['point'];ax.plot([u[0],d[0]],[u[1],d[1]],color=color,ls='--',lw=.6,alpha=.65)
            ax.text(*s['centroid'],f"#{t['id']}",color=color,fontsize=9)
    if geo:x0,y0,x1,y1=geo['scale']['roi'];ax.set_xlim(x0-35,x1+35);ax.set_ylim(min(row['height'],y1+5),max(0,y0-10))
    ax.set_title('Actual C1 contributors and FL boundary geometry\nWhite: expert boundaries; pale dots: expert fascicle traces',fontsize=10);ax.axis('off')
    tt=sorted(ts,key=lambda t:t['FL']);yy=np.arange(len(tt));bx.barh(yy,[t['FL'] for t in tt],color=['#f5aa42' if t['id']==a['flattest_id'] else '#59a5ce' for t in tt],alpha=.85)
    bx.set_yticks(yy,[f"#{t['id']} | {t['weight']*100:.1f}%" for t in tt]);bx.axvline(a['reference']['FL'],color='#249c61',ls='--',lw=1.5,label=f"Reference {a['reference']['FL']:.2f}")
    bx.axvline(a['values']['FL'],color='#b33e55',lw=1.5,label=f"Weighted output {a['values']['FL']:.2f}");bx.set_xlabel('Straight-line intersection FL (mm)');bx.grid(axis='x',alpha=.2);bx.legend(fontsize=9)
    oracle=a['fixed_weights_expert_boundary_FL'];text='not available' if oracle is None else f'{oracle:.2f} mm'
    fig.suptitle(f"{sid} | seed {seed} | FL {a['values']['FL']:.2f} / reference {a['reference']['FL']:.2f} mm\nFixed C1 and weights + expert boundaries (diagnostic only): {text}",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.9));dest=OUT/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/f'{sid}_{seed}.png',dpi=135);plt.close(fig)

def stats(cases,seed,ds=None):
    rr=[r['audit'][seed] for r in cases if (ds is None or r['dataset']==ds) and r['audit'][seed]['values']['FL'] is not None]
    bad=[a for a in rr if a['FL_absolute_error']>10];oracle=[a for a in rr if a['oracle_complete']];obad=[a for a in bad if a['oracle_complete']]
    def comparison(rs):
        return dict(n=len(rs),baseline_MAE=float(np.mean([a['FL_absolute_error'] for a in rs])) if rs else None,
            expert_boundary_fixed_weights_MAE=float(np.mean([abs(a['fixed_weights_expert_boundary_FL']-a['reference']['FL']) for a in rs])) if rs else None,
            improved=sum(abs(a['fixed_weights_expert_boundary_FL']-a['reference']['FL'])<a['FL_absolute_error']-1e-8 for a in rs),
            worse=sum(abs(a['fixed_weights_expert_boundary_FL']-a['reference']['FL'])>a['FL_absolute_error']+1e-8 for a in rs))
    removal={}
    for key in ('without_flattest','without_longest'):
        q=[a for a in rr if a[key] is not None];removal[key]=dict(n=len(q),baseline_MAE=float(np.mean([a['FL_absolute_error'] for a in q])),counterfactual_MAE=float(np.mean([abs(a[key]-a['reference']['FL']) for a in q])),
            improved=sum(abs(a[key]-a['reference']['FL'])<a['FL_absolute_error']-1e-8 for a in q),worse=sum(abs(a[key]-a['reference']['FL'])>a['FL_absolute_error']+1e-8 for a in q))
    return dict(available=len(rr),bad_over10=len(bad),over10=sum(a['FL_error']>10 for a in rr),under10=sum(a['FL_error']< -10 for a in rr),
        reference_boundary_diagnostic=comparison(oracle),bad_reference_boundary_diagnostic=comparison(obad),drop_one_posthoc_not_algorithm=removal)

def main():
    OUT.mkdir(parents=True,exist_ok=True);cases=[];maximum=0.;sourcechecks={}
    for file in sorted((R78/'results').glob('*/*/identity.json.gz')):
        r=readgz(file);row=r['row']
        if not row['reference']:continue
        src=readgz(ROOT/r['source']);g=j(R77/f"gallery/data/{row['dataset']}__{row['image_id']}__identity.json")
        geo=j(ROOT/f"validation/young_expert_v03/geometry/{row['image_id']}.json") if row['dataset']=='young78' else None
        audit={seed:evaluate(r,src,g,seed,geo) for seed in SEEDS}
        for seed in SEEDS:
            if audit[seed]['tracks']:maximum=max(maximum,abs(sum(t['weight']*t['FL'] for t in audit[seed]['tracks'])-audit[seed]['values']['FL']))
        detail=dict(dataset=row['dataset'],id=row['image_id'],source=file.relative_to(ROOT).as_posix(),source_sha=sha(file),reference=row['reference'],audit=audit)
        if geo:
            detail['expert_visible_straight_diagnostic']=expert_trace_diagnostic(geo,row['annotation_offset'],expert_boundary(geo,row['annotation_offset']),row['width'],row['height'])
        cases.append(detail);sourcechecks[detail['source']]=detail['source_sha']
        if any(a['FL_absolute_error'] is not None and a['FL_absolute_error']>10 for a in audit.values()):
            for seed in SEEDS:figure(r,src,g,audit,seed)
    write(OUT/'analysis.json',dict(status='posthoc_read_only_diagnosis_not_new_algorithm',seeds=SEEDS,cohort=112,reference_available_FL={s:sum(c['audit'][s]['values']['FL'] is not None for c in cases) for s in SEEDS},
        max_weighted_FL_reconstruction_error=maximum,results={s:{ds:stats(cases,s,None if ds=='combined112' else ds) for ds in ('external34','young78','combined112')} for s in SEEDS},
        cases=cases,source_hashes=sourcechecks,training=False,inference=False,default_changed=False,PA_FL_values_changed=False,
        oracle_warning='Expert boundary substitution keeps every original C1 contributor and its original weight. Missing roots are reported, never silently renormalized. It diagnoses sensitivity, not a deployable method or a pure causal partition.',
        reference_warning='New78 derived expert-polylines include assisted endpoint completion; different spatial samples from dense C1 lines. Source curvature and local straight extrapolation differ; do not label every nonmatched line as false detection.'))
    print('FL_ERROR_AUDIT_READY',len(cases),'maximum reconstruction',maximum)

if __name__=='__main__':main()
