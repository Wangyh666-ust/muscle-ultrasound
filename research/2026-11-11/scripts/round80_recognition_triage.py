"""Source-supported probability-to-measurement audit. No fitting rules/models are changed."""
from scripts.round80_common import ROOT,j,write,readgz,sha,np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter,defaultdict
import time
from scripts.round77_common import load_context
from scripts.audit_boundary_connection_20261110 import references,original
from scripts.round63_paths import decode,image_evidence,sample
from scripts.round64_seams import reconstruct as seams
from scripts.round65_core import keep
from scripts.round29_core import value_at

OUT=ROOT/'review/recognition_chain_triage_20261110'
SIDES=('upper','lower')
LABELS=('input_or_evidence_excluded','candidate_present_not_used','qualified_path_fit_distortion','local_response_no_retained_track',
        'role_head_weak_union_strong','nearby_displaced_response','weak_or_absent_local_response','mixed_selection_or_support')


def probability_near(c,gt,field,radius):
    m=c['matrix'];box=c['frame']['input_box'];x=gt[:,0]*m[0,0]+m[0,2]-box[0];y=gt[:,1]*m[1,1]+m[1,2]-box[1]
    radius*=m[0,0];n=int(np.ceil(radius/.5));offset=np.arange(-n,n+1,dtype=float)*.5
    offset=offset[abs(offset)<=radius+1e-8];offset=np.unique(np.r_[offset,-radius,radius])
    return np.max([sample(field,x,y+d) for d in offset],axis=0)


def path_hit(paths,gt,tol,observed=True):
    x,y=gt.T;hit=np.zeros(len(x),bool)
    for p in paths:
        px=np.asarray(p['x']);py=np.asarray(p['y']);q=np.asarray(p['observed'],bool) if observed else np.asarray(p['qualified'],bool)
        px,py=px[q],py[q]
        if not len(px):continue
        a=np.clip(np.searchsorted(px,x),0,len(px)-1);b=np.clip(a-1,0,len(px)-1);i=np.where(abs(px[a]-x)<=abs(px[b]-x),a,b)
        hit|=(abs(px[i]-x)<=1.01)&(abs(py[i]-y)<=tol)
    return hit


def stats(v):
    return dict(median=float(np.median(v)),mean=float(np.mean(v)),maximum=float(np.max(v)),over05=float(np.mean(v>=.5)),over018=float(np.mean(v>=.18)))


def work(row):
    dst=OUT/f"records/{row['dataset']}__{row['image_id']}.json.gz"
    if dst.exists():return readgz(dst)
    gt,gtclass=references(row);assert gt is not None
    c=load_context(row,'identity');cfg=j(ROOT/'runs/round63_boundary_paths_v1/config.json');scfg=j(ROOT/'runs/round64_boundary_continuation_v1/controls/S_confirmation_lock.json')['selected']
    metadata=j(ROOT/f"runs/round75_density_measurement_v1/inference/{row['dataset']}/{row['image_id']}/identity/boundary.json")
    assert sha(ROOT/metadata['data_path'])==metadata['sha']
    assert metadata['frame']['input_box']==c['frame']['input_box']
    frozen=readgz(ROOT/f"runs/round77_subject_layer_v1/results/{row['dataset']}/{row['image_id']}/identity.json.gz")
    decoded=decode(c['prob'],c['image'],c['valid'],c['width'],cfg,c['mapped'],c['scale']);gray=image_evidence(c['image'],c['width']);sides={}
    tol=2*max(c['image'].shape)/512;mincols=max(12,int(np.ceil(.08*np.ptp(c['domain']))))
    for i,side in enumerate(SIDES):
        points=gt[side];head=i+1;field=c['prob'][head]*(.4+.6*c['prob'][0])
        extra,seam_audit=seams(decoded['tracks']['rebuilt'][i],field,gray,c['valid'],c['width'],scfg,cfg)
        pools=keep(decoded['tracks']['rebuilt'][i],extra,c['width'])
        weak=[original(p,c) for p in decoded['tracks']['weak'][i]];pool=[original(p,c) for p in pools]
        raw=probability_near(c,points,c['prob'][head],tol);union=probability_near(c,points,c['prob'][0],tol);opposite=probability_near(c,points,c['prob'][3-head],tol)
        fused=probability_near(c,points,field,tol);broad=probability_near(c,points,field,2*tol)
        x,y=points.T;native_x=x-c['frame']['input_box'][0];native_y=y-c['frame']['input_box'][1]
        inframe=sample(c['valid'].astype(float),native_x,native_y)>.99;intissue=sample(c['native_mask'].astype(float),native_x,native_y)>.99
        w_hit=path_hit(weak,points,tol);p_hit=path_hit(pool,points,tol);pair=frozen['pairs']['P'];cv=frozen['curves']['P'];role=('sup','deep')[i]
        selected=np.zeros(len(x),bool);qualified=np.zeros(len(x),bool);complete=np.zeros(len(x),bool);formal=np.zeros(len(x),bool)
        if pair:
            selected=path_hit([pair[side]],points,tol);qualified=path_hit([pair[side]],points,tol,False)
            sp=pair[side];inside=(x>=sp['x'][0])&(x<=sp['x'][-1]);complete=inside&(abs(np.interp(x,sp['x'],sp['y'])-y)<=tol)
        if cv:
            pred=np.array([value_at(cv[role],float(xx)) for xx in x]);formal=abs(pred-y)<=tol
        failed=~formal;category=np.full(len(x),'correct_formal_geometry',dtype=object)
        for n in np.flatnonzero(failed):
            if not inframe[n]:why=LABELS[0]
            elif p_hit[n] and not qualified[n]:why=LABELS[1]
            elif qualified[n]:why=LABELS[2]
            elif fused[n]>=.18 and not p_hit[n]:why=LABELS[3]
            elif union[n]>=.5 and raw[n]<.18:why=LABELS[4]
            elif fused[n]<.18 and broad[n]>=.5:why=LABELS[5]
            elif fused[n]<.18:why=LABELS[6]
            else:why=LABELS[7]
            category[n]=why
        counts=Counter(category);full_flags={k:int(v) for k,v in counts.items() if k!='correct_formal_geometry'}
        # Descriptive flags can overlap; primary attribution is only a stage-localization heuristic.
        flags=dict(response_below_low_over20pct=bool(np.mean(fused<.18)>.2),strong_response_but_no_pool=bool(np.count_nonzero((fused>=.5)&~p_hit)>=mincols),
                   correct_pool_not_selected=bool(np.count_nonzero(p_hit&~selected)>=mincols),qualified_path_formal_failure=bool(np.count_nonzero(qualified&~formal)>=mincols))
        item=dict(reference_columns=len(x),range_x=[float(x[0]),float(x[-1])],tolerance_px=tol,input_evidence_fraction=float(inframe.mean()),tissue_mask_fraction=float(intissue.mean()),
             heads=dict(role=stats(raw),union=stats(union),opposite=stats(opposite),fused=stats(fused)),
             stage_correct_coverage=dict(weak_observed=float(w_hit.mean()),pool_observed=float(p_hit.mean()),selected_observed=float(selected.mean()),selected_qualified=float(qualified.mean()),completed_geometry=float(complete.mean()),formal_geometry_with_rays=float(formal.mean())),
             failed_columns=int(failed.sum()),failed_fraction=float(failed.mean()),stage_localization_counts=full_flags,flags=flags,
             weak_paths=len(weak),pool_paths=len(pool),min_flag_columns=mincols)
        if row['px_per_mm']:
            item['physical_radius_profiles']={str(rad):{name:stats(probability_near(c,points,c['prob'][h],rad*row['px_per_mm'])) for name,h in [('union',0),('role',head),('opposite',3-head)]} for rad in (.25,.5,1.)}
        if row['image_id'] in ('ANON_144','ANON_68','ANON_78'):
            item['column_details']=dict(x=x,y=y,role=raw,union=union,fused=fused,broad_fused=broad,pool_correct=p_hit,selected_correct=selected,formal_correct=formal,category=category.tolist())
        sides[side]=item
    result=dict(dataset=row['dataset'],id=row['image_id'],group=row['group'],reference_class=gtclass,sides=sides,
       probability_path=metadata['data_path'],probability_sha=metadata['sha'],reference_used_only_after_decoding=True,
       caveat='Stages locate evidence loss, not a causal partition of FL error or proof that training versus preprocessing caused a low response.')
    write(dst,result);return result


def summarize(rows):
    out={}
    for side in SIDES:
        rr=[r['sides'][side] for r in rows];cols=sum(r['reference_columns'] for r in rr)
        failures=Counter();flags=Counter()
        for r in rr:failures.update(r['stage_localization_counts']);flags.update({k:int(v) for k,v in r['flags'].items()})
        out[side]=dict(images=len(rr),source_columns=cols,mean_image_role_over05=float(np.mean([r['heads']['role']['over05'] for r in rr])),
            mean_image_fused_over018=float(np.mean([r['heads']['fused']['over018'] for r in rr])),
            mean_stage_coverages={k:float(np.mean([r['stage_correct_coverage'][k] for r in rr])) for k in rr[0]['stage_correct_coverage']},
            failed_columns=sum(r['failed_columns'] for r in rr),stage_localization_counts=dict(failures),flag_sides=dict(flags),
            flag_caution='Flags overlap; source columns and repeated images are not independent subjects.')
    return out


def main():
    rows=j(ROOT/'runs/round77_subject_layer_v1/protocol.json')['rows'];wanted=[r for r in rows if references(r)[0] is not None];assert len(wanted)==221
    if not (OUT/'protocol.json').exists():
        write(OUT/'protocol.json',dict(scope='221 original images with source-supported upper/lower traces; other326 cannot be assigned reference-based errors',training=False,inference=False,default_changed=False,subagents=False,
           frame_validation='cached probability SHA plus exact input-box binding; no augmented/input changes',
           thresholds='Existing decoder low.18/high.5, 2 network grid pixels localization tolerance; broader4-grid neighborhood only separates possible displacement.',
           rule_order=LABELS,reference_scope='143 historical manual boundaries +78 expert source-drawn portions only, not assisted extension',
           non_causal=True,FL_limit='Visible boundary-column error localization does not measure each mechanism contribution to FL error outside the image.',
           classification='Condition on current formal boundary being outside reference tolerance. Source-evidence exclusion, existing correct pool/qualification, local response without retained trace, identity-head weakness, neighboring response or low response. Report overlapping flags and all continuous rates too.'))
    start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(work,r) for r in wanted]
        for f in as_completed(fs):
            records.append(f.result())
            if len(records)%40==0:print('RECOGNITION_TRIAGE',len(records),221,round(time.monotonic()-start),flush=True)
    summary={ds:summarize([r for r in records if ds=='combined221' or r['dataset']==ds]) for ds in ('young78','apo153','combined221')}
    write(OUT/'summary.json',summary);write(OUT/'done.json',dict(images=len(records),sides=442,seconds=time.monotonic()-start))
    print('RECOGNITION_TRIAGE_DONE',json_safe_summary(summary))


def json_safe_summary(summary):
    return {ds:{side:dict(mean_role_over05=d['mean_image_role_over05'],flags=d['flag_sides'],failed_columns=d['failed_columns'],localization=d['stage_localization_counts']) for side,d in v.items()} for ds,v in summary.items()}


if __name__=='__main__':main()
