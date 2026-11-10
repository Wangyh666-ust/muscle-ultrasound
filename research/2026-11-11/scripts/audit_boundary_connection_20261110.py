"""Read-only replay of frozen candidate construction, with labels used only afterwards."""
from scripts.round43_cpu_imports import install
install()
import argparse, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import numpy as np
from scripts.round77_common import ROOT, j, write, readgz, sha, load_context, NEW
from scripts.round63_paths import (decode, bridge, tangent, image_evidence, path_quality,
                                  hermite, refine_bridge, sample, choose_pair)
from scripts.round64_seams import reconstruct as seams, crop_path
from scripts.round65_core import keep
from scripts.round77_layer import rank, signature

OUT=ROOT/'review/boundary_connection_audit_20261110'
SIDES=('upper','lower')
_MANUAL=None


def original(p,c):
    q=deepcopy(p);s=c['matrix'][0,0];offset=c['matrix'][:,2];box=c['frame']['input_box']
    q['x']=(np.asarray(p['x'])+box[0]-offset[0])/s
    q['y']=(np.asarray(p['y'])+box[1]-offset[1])/s
    return q


def references(row):
    global _MANUAL
    if row['dataset']=='young78':
        g=j(ROOT/f"validation/young_expert_v03/geometry/{row['image_id']}.json")
        out={};lo,_,hi,_=g['scale']['roi']
        for side in SIDES:
            p=np.asarray(g[side]['source_dense'],float).copy();p[:,0]-=row['annotation_offset']
            p=p[(p[:,0]>=lo)&(p[:,0]<hi)];out[side]=p
        return out,'expert_source_supported_only'
    if row['dataset']=='apo153':
        if _MANUAL is None:
            _MANUAL={r['key']:r for r in j(ROOT/'runs/frozen_boundary_20261110/manifest.json')['rows'] if r['view']=='identity' and r['manual']}
        fr=_MANUAL.get(row['dataset']+'|'+row['image_id'])
        if fr:
            with np.load(ROOT/fr['input']['path']) as z:
                x=z['x']+fr['input']['input_box'][0];ys=z['ys']+fr['input']['input_box'][1]
            return {s:np.c_[x,ys[i]] for i,s in enumerate(SIDES)},'historical_manual_boundary'
    return None,'no_source_boundary_reference'


def bridge_details(a,b,field,gray,c,cfg):
    p,why=bridge(a,b,field,gray,c['valid'],c['width'],cfg)
    w=c['width'];gap=float(b['x'][0]-a['x'][-1]);info=dict(reason=why or 'accepted',gap_native=gap)
    if why=='overlap_different_trend':
        xx=np.arange(np.ceil(b['x'][0]),np.floor(a['x'][-1])+1)
        ya,yb=[np.interp(xx,z['x'],z['y']) for z in (a,b)]
        info.update(overlap_columns=len(xx),separation_max=float(abs(ya-yb).max()),distance_limit=cfg['near_distance']*w,
                    trend_difference_p90_deg=float(np.degrees(np.quantile(abs(np.arctan(np.gradient(ya))-np.arctan(np.gradient(yb))),.9))),trend_limit_deg=25.)
    if why=='abrupt_bridge' and gap>=1:
        x0,x1=a['x'][-1],b['x'][0];y0,y1=a['y'][-1],b['y'][0]
        m0,m1=tangent(a,True,w,cfg),tangent(b,False,w,cfg)
        xx=np.linspace(x0,x1,max(9,int(np.ceil(gap))+1));y,dy,ddy=hermite(xx,x0,y0,m0,x1,y1,m1)
        ry,rd,rdd,changed=refine_bridge(xx,y,dy,ddy,field,gray,w,cfg)
        budget=cfg['max_curvature']+12*w*(2*w/1046.)/gap**2;chord=y0+(y1-y0)*(xx-x0)/gap;variants=[]
        for alpha in ((0.,.25,.5,1.) if changed else (0.,)):
            yy=y+alpha*(ry-y);d=dy+alpha*(rd-dy);dd=ddy+alpha*(rdd-ddy)
            curvature=float(np.max(w*abs(dd)/(1+d*d)**1.5));turn=float(np.degrees(np.ptp(np.arctan(d))));bulge=float(abs(yy-chord).max())
            variants.append(dict(alpha=alpha,curvature=curvature,curvature_limit=budget,turn_deg=turn,turn_limit_deg=cfg['max_turn_deg'],bulge=bulge,bulge_limit=cfg['maximum_bulge']*w,
                                 failed=[n for n,v in [('curvature',curvature>budget),('turn',turn>cfg['max_turn_deg']),('bulge',bulge>cfg['maximum_bulge']*w)] if v]))
        info.update(endpoint_slopes=[m0,m1],variants=variants)
    return info


def peak_near(c,gt,head,radius):
    m=c['matrix'];box=c['frame']['input_box'];x=gt[:,0]*m[0,0]+m[0,2]-box[0];y=gt[:,1]*m[1,1]+m[1,2]-box[1]
    radius=radius*m[0,0]
    offsets=np.linspace(-radius,radius,9)
    return np.max([sample(c['prob'][head],x,y+d) for d in offsets],axis=0)


def one(row,view):
    dst=OUT/f"records/{row['dataset']}/{row['image_id']}/{view}.json.gz"
    if dst.exists():return dict(id=row['image_id'],view=view,resumed=True)
    start=time.monotonic();c=load_context(row,view);cfg=j(ROOT/'runs/round63_boundary_paths_v1/config.json')
    scfg=j(ROOT/'runs/round64_boundary_continuation_v1/controls/S_confirmation_lock.json')['selected']
    frozen=readgz(ROOT/f"runs/round77_subject_layer_v1/results/{row['dataset']}/{row['image_id']}/{view}.json.gz")
    decoded=decode(c['prob'],c['image'],c['valid'],c['width'],cfg,c['mapped'],c['scale']);gray=image_evidence(c['image'],c['width'])
    pools=[];seam_audits=[]
    for i,side in enumerate(SIDES):
        field=c['prob'][i+1]*(.4+.6*c['prob'][0]);extra,au=seams(decoded['tracks']['rebuilt'][i],field,gray,c['valid'],c['width'],scfg,cfg)
        pools.append(keep(decoded['tracks']['rebuilt'][i],extra,c['width']));seam_audits.append(dict(added=len(extra),rejections=dict(Counter(au))))
    _,candidates=choose_pair(pools,c['width'],cfg,c['mapped'],c['scale'],True,True)
    chosen,rank_audit=rank(candidates,pools,c['prob'],c['mapped'],c['width'],c['domain'],None,bool(c['scale']))
    expected=frozen['ranking_audit']['P'].get('selected_ids')
    selected_ids=signature(chosen)
    pair=frozen['pairs']['P'];curve=frozen['curves']['P'];gt,gtclass=references(row)
    # This is a descriptive screen, not a new qualification/measurement rule.
    span=float(np.ptp(c['domain'])/c['matrix'][0,0]);tau=2*max(c['image'].shape)/512/c['matrix'][0,0]
    mincols=max(12,int(np.ceil(.08*span)))
    sides={};detail=[]
    for i,side in enumerate(SIDES):
        role=('sup','deep')[i];ps=[original(p,c) for p in pools[i]]
        field=c['prob'][i+1]*(.4+.6*c['prob'][0]);sel=pair[side] if pair else None;fit=curve[role] if curve else None
        sideout=dict(pool_paths=len(ps),weak_paths=len(decoded['tracks']['weak'][i]),rebuilt_paths=len(decoded['tracks']['rebuilt'][i]),
                     direct_rejections=decoded['audit'][i]['rejected'],seams=seam_audits[i],unused_complementary=[],confirmed_correct_omitted=[],fit_discrepancy=None,source_reference=None)
        if sel:
            sx=np.asarray(sel['x']);sy=np.asarray(sel['y']);obs=np.asarray(sel['observed'],bool);q=np.asarray(sel['qualified'],bool)
            selected_lineage=sel.get('ids',chosen[side]['ids'] if chosen else ())
            ox=sx[obs];oy=sy[obs];sideout['selected_ids']=selected_lineage;sideout['selected_id_source']='saved_final' if 'ids' in sel else 'replayed_pre_completion_rank'
            sideout['observed_range']=[float(ox.min()),float(ox.max())] if len(ox) else None
            if fit:
                xx=sx[q&(sx>=fit['support'][0])&(sx<=fit['support'][1])];yy=np.interp(xx,sx,sy)
                error=abs(np.interp(xx,fit['knots_x'],fit['knots_y'])-yy)
                sideout['fit_discrepancy']=dict(columns=len(xx),max_px=float(error.max()) if len(xx) else None,p90_px=float(np.quantile(error,.9)) if len(xx) else None,
                    over_tolerance=int((error>tau).sum()),large_flag=bool(len(xx) and (error>tau).sum()>=mincols),FL_support=fit['support'])
            for path in ps:
                if set(path['ids'])&set(selected_lineage):continue
                po=np.asarray(path['observed'],bool);px=path['x'][po];py=path['y'][po]
                if len(ox) and len(px):
                    outside=(px<ox.min())|(px>ox.max());side_direction='left' if px[0]<ox.min() else 'right'
                    gap=max(float(ox.min()-px[-1]),float(px[0]-ox.max()),0.)
                    # Current distance scale only flags possible continuations; it does not prove anatomical identity.
                    at=px[-1] if side_direction=='left' else px[0];end=0 if side_direction=='left' else -1
                    near=(abs(ox-ox[end])<=.05*c['width']/c['matrix'][0,0])
                    if near.sum()>=3:sl=np.polyfit(ox[near]-ox[end],oy[near],1)[0]
                    else:sl=0.
                    predicted=oy[end]+sl*(at-ox[end]);actual=np.interp(at,px,py);residual=abs(actual-predicted)/np.sqrt(1+sl*sl)
                    if outside.sum()>=mincols and gap<=cfg['max_gap']*c['width']/c['matrix'][0,0] and residual<=cfg['near_distance']*c['width']/c['matrix'][0,0]:
                        sideout['unused_complementary'].append(dict(ids=path['ids'],outside_columns=int(outside.sum()),direction=side_direction,gap_px=gap,endpoint_forecast_distance_px=float(residual),anatomical_identity_unconfirmed=True))
                if gt is not None and len(gt[side]) and len(px):
                    ref=gt[side];inside=(px>=ref[0,0])&(px<=ref[-1,0]);expected_y=np.interp(px,ref[:,0],ref[:,1]);correct=inside&(abs(py-expected_y)<=tau)
                    if len(ox):
                        idx=np.searchsorted(ox,px);a=np.clip(idx-1,0,len(ox)-1);b=np.clip(idx,0,len(ox)-1);nearest=np.where(abs(ox[a]-px)<=abs(ox[b]-px),a,b)
                        already=(abs(ox[nearest]-px)<=1.01/c['matrix'][0,0])&(abs(oy[nearest]-py)<=tau)
                    else:already=np.zeros(len(px),bool)
                    missing=correct&~already
                    if missing.sum()>=mincols:
                        sideout['confirmed_correct_omitted'].append(dict(ids=path['ids'],correct_omitted_columns=int(missing.sum()),range=[float(px[missing].min()),float(px[missing].max())],mean_error_px=float(abs(py[missing]-expected_y[missing]).mean())))
            if gt is not None and len(gt[side]):
                ref=gt[side];raw=peak_near(c,ref,i+1,tau);union=peak_near(c,ref,0,tau)
                pred=np.interp(ref[:,0],sx,sy);inside=(ref[:,0]>=sx[0])&(ref[:,0]<=sx[-1]);fitinside=np.zeros(len(ref),bool) if not fit else (ref[:,0]>=fit['support'][0])&(ref[:,0]<=fit['support'][1])
                fiterror=None if not fit else abs(np.interp(ref[:,0],fit['knots_x'],fit['knots_y'])-ref[:,1])
                sideout['source_reference']=dict(columns=len(ref),role_over05=float(np.mean(raw>.5)),union_over05=float(np.mean(union>.5)),
                    completed_correct_coverage=float(np.mean(inside&(abs(pred-ref[:,1])<=tau))),
                    FL_fit_correct_coverage=float(np.mean(fitinside&(fiterror<=tau))) if fiterror is not None else 0.,
                    raw_strong_beyond_FL_support=int(((raw>.5)&~fitinside).sum()),
                    completed_good_but_fit_bad=bool(fiterror is not None and np.mean(inside&(abs(pred-ref[:,1])<=tau))>=.9 and np.mean(fitinside&(fiterror<=tau))<.8))
        if row['image_id'] in ('ANON_68','ANON_78') and view=='identity':
            weak=decoded['tracks']['weak'][i]
            for a in weak:
                for b in weak:
                    if b['x'][0]<=a['x'][0] or b['x'][-1]<=a['x'][-1]:continue
                    report=dict(side=side,left_ids=a['ids'],right_ids=b['ids'],left_range=original(a,c)['x'][[0,-1]],right_range=original(b,c)['x'][[0,-1]],
                                direct=bridge_details(a,b,field,gray,c,cfg),trimmed=[])
                    for ltrim in scfg['seam_trim_fractions']:
                        for rtrim in scfg['seam_trim_fractions']:
                            if not ltrim and not rtrim:continue
                            ma=a['x']<=a['x'][-1]-ltrim*c['width'];mb=b['x']>=b['x'][0]+rtrim*c['width']
                            if ma.sum()<8 or mb.sum()<8:continue
                            aa,bb=crop_path(a,ma),crop_path(b,mb)
                            if aa['confidence'][-1]<.5 or bb['confidence'][0]<.5:continue
                            report['trimmed'].append(dict(trim=[ltrim,rtrim],**bridge_details(aa,bb,field,gray,c,cfg)))
                    detail.append(report)
        sides[side]=sideout
    candidate_scores=[]
    for p in candidates:
        base=.5*(path_quality(p['upper'],c['width'])+path_quality(p['lower'],c['width']))+.1*np.ptp(p['interval'])/c['width']
        candidate_scores.append(dict(ids=signature(p),quality_coverage=base,C1_active=p['c1_active'],C1_value=p['c1'],C1_tracks=p['c1_tracks'],C1_bonus=.2*(p['c1']-.5) if p['c1_active'] else 0.,old_prior_penalty=p['prior_penalty'],MT_mm=p['MT_mm'],original_score=p['score']))
    write(dst,dict(dataset=row['dataset'],id=row['image_id'],view=view,reference_class=gtclass,tolerance_px=tau,min_screen_columns=mincols,
        sides=sides,selected_ids=selected_ids,expected_ranking_ids=expected,ranking_replay_matches=(selected_ids==expected) if expected is not None else None,
        source_result_sha=sha(ROOT/f"runs/round77_subject_layer_v1/results/{row['dataset']}/{row['image_id']}/{view}.json.gz"),
        probability_binding_manifest='probability_binding_validation.json',
        candidate_scores=candidate_scores if row['image_id'] in ('ANON_68','ANON_78') else [],detail=detail,seconds=time.monotonic()-start))
    return dict(id=row['image_id'],view=view,seconds=time.monotonic()-start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    protocol=j(ROOT/'runs/round77_subject_layer_v1/protocol.json');rows=protocol['rows'];tasks=[]
    for row in rows:
        if args.pilot and row['image_id'] not in ('ANON_68','ANON_78'):continue
        views=('identity',) if args.pilot or row['reference'] is None else ('identity','translate2_-1','brightness105','scale1015',NEW)
        tasks.extend((row,v) for v in views)
    if not (OUT/'protocol.json').exists():
        write(OUT/'protocol.json',dict(training=False,new_inference=False,default_changed=False,subagents=False,originals=547,views=995,
          frozen='R77P/R78M/R79B geometry and candidates; diagnostic only; no changed thresholds or outputs',
          screen='Omitted independent same-role path adds >=max(12,8% tissue width) observed columns, gap <= existing20% input-width limit, endpoint forecast distance <= existing4% input-width distance. Screen is NOT proof of same layer.',
          confirmed='An existing same-role omitted path has >=max(12,8% tissue width) correct columns against source-supported expert boundary, not already represented by selected observed geometry. GT used after decoding only.',
          fit_gap='At least same minimum columns of qualified completed path differ from actual FL four-piece fit by >2 native network-grid pixels, inside fit support. Descriptive discrepancy, not automatic error.',
          source_manual='143 apo153 expert references plus78 source-supported expert polylines; no reference geometry for remaining326 originals',
          code_hashes={p:sha(ROOT/p) for p in ['scripts/round63_paths.py','scripts/round64_seams.py','scripts/round68_geometry.py','scripts/round77_layer.py','scripts/round64_measure_adapter.py','scripts/round79_gallery.py']}))
    start=time.monotonic();done=0;failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(one,*task) for task in tasks]
        for f in as_completed(fs):
            try:f.result()
            except Exception as e:
                failures.append(dict(error=str(e),type=type(e).__name__))
                print('AUDIT_FAILURE',failures[-1],flush=True)
            done+=1
            if done%50==0 or done==len(tasks):print('BOUNDARY_CONNECTION_AUDIT',done,len(tasks),'seconds',round(time.monotonic()-start),flush=True)
    write(OUT/('pilot_done.json' if args.pilot else 'done.json'),dict(views=done,failures=failures,seconds=time.monotonic()-start))
    assert not failures,failures


if __name__=='__main__':main()
