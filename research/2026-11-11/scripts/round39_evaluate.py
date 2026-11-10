"""Prespecified paired/grouped evaluation. Outputs frozen before reading references."""
import sys,copy,math,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict,Counter
from scripts.round39_common import *
from scripts.round39_core import ARMS,domain
from scripts.round38_evaluate import diff,worst,summary_worst,transform_summary,complete,finite,segs,METRICS
from scripts.round36_evaluate import coverage as coverage_summary,measurement_errors
from scripts.round25_evaluate import REF,validate_reference,f1_score
from scripts.round27_experiment import check_equal
EVAL_ARMS=ARMS+('R','A','NVA')

def reduce_arm(a):
    fields=('values','curve','selection','FL_weights','PA_weights','PA_ids','PA_invalid','roi','raw_components','filter_count','exploratory_values','stage','selected_pair_id','pair_eligibility','rejected_curve')
    q={k:a[k] for k in fields if k in a};q['segments']=segs(a)
    return q

def summary(records):
    q=summary_worst(records)
    q.update(any_availability_flip_count=sum(r['any_availability_flip'] for r in records),any_metric_missing_count=sum(r['any_metric_missing'] for r in records),missing_all_original_metrics=sum(r['all_original_missing'] for r in records))
    return q

def paired_bootstrap(records,split,after,before,partition,seed):
    by={a:{r['key']:r for r in records if r['arm']==a and r['seed']==seed and (partition=='all' or split['assignment'][r['key']]==partition)} for a in (before,after)}
    comps=defaultdict(list)
    for k in by[before]:comps[split['component_ids'][k]].append(k)
    metadata={c['id']:c for c in split['components']};strata=defaultdict(list)
    for cid in sorted(comps):strata[tuple(metadata[cid]['datasets'])].append(cid)
    metrics=('joint_pass','original_complete','any_availability_flip','severe','any_metric_missing')
    # Sum numerator and common denominator within cluster; resample clusters paired.
    arrays=[]
    for cids in strata.values():
        arr=[]
        for cid in cids:
            row=[]
            for m in metrics:
                keys=[k for k in comps[cid] if m!='joint_pass' or by[before][k]['reference_complete']]
                row.extend([sum(int(by[after][k][m])-int(by[before][k][m]) for k in keys),len(keys)])
            arr.append(row)
        arrays.append(np.array(arr,float))
    original=sum(a.sum(0) for a in arrays);rng=np.random.default_rng(2026110339);boots=np.zeros((5000,2*len(metrics)))
    for a in arrays:
        # All same-cluster images/transforms remain together; seeds never pooled.
        ix=rng.integers(0,len(a),size=(5000,len(a)));boots+=a[ix].sum(1)
    out={}
    for i,m in enumerate(metrics):
        samples=np.divide(boots[:,2*i],boots[:,2*i+1],out=np.full(5000,np.nan),where=boots[:,2*i+1]>0)
        out[m]=dict(delta_fraction=float(original[2*i]/original[2*i+1]),denominator=int(original[2*i+1]),ci95=np.nanquantile(samples,[.025,.975]).tolist())
    return dict(seed=seed,after=after,before=before,partition=partition,components=len(comps),strata_sizes={'+'.join(k):len(v) for k,v in strata.items()},metrics=out,interpretation='Descriptive clustered intervals on previously observed data; single-component strata cannot estimate between-component variation.')

def natural(rows,split,rois):
    pairs=[];groups=j(S01/'groups.json')
    for seed,rr in rows.items():
        for e in groups['pairs']:
            if e['pixel_identical']:continue
            r,s=rr[e['a']],rr[e['b']]
            for arm in EVAL_ARMS:
                a=r['arms'][arm];b=copy.deepcopy(s['arms'][arm]);ratio=(s.get('px_per_mm') or 1)/(r.get('px_per_mm') or 1)
                for p,dim in [('MT','height'),('FL','width')]:
                    if b['values'][p] is not None:b['values'][p]*=ratio*r[dim]/s[dim]
                b.pop('curve',None);d=diff(a,b,r,rois[e['a']]);d.update(seed=seed,arm=arm,a=e['a'],b=e['b'],group=e['group'],component=split['component_ids'][e['a']],strict=e['strict'],partition=split['assignment'][e['a']])
                d['either_endpoint_missing']=not (complete(a) and complete(b));pairs.append(d)
    summaries=[]
    for seed in SEEDS:
        for arm in EVAL_ARMS:
            for partition in ('all','development','confirmation'):
                for strict in (False,True):
                    pp=[p for p in pairs if p['seed']==seed and p['arm']==arm and (partition=='all' or p['partition']==partition) and (not strict or p['strict'])];q=transform_summary(pp);q.update(seed=seed,arm=arm,partition=partition,mode='strict' if strict else 'all_nonidentical',group_equal={},either_endpoint_missing=sum(p['either_endpoint_missing'] for p in pp))
                    for m in METRICS:
                        gs=defaultdict(list)
                        for p in pp:
                            v=p['parameters'][m]['absolute' if m=='PA' else 'ROI_absolute']
                            if v is not None:gs[p['component']].append(v)
                        q['group_equal'][m]=float(np.mean([np.mean(v) for v in gs.values()])) if gs else None
                    summaries.append(q)
    # Paired numerical effects use common observable edges. Missing remains a separate endpoint.
    contrasts=[]
    for seed in SEEDS:
        for after,before in [('J','B0'),('WS','B0'),('E','VC'),('J','R'),('WS','R')]:
            for partition in ('all','confirmation'):
                for strict in (False,True):
                    bb={(p['a'],p['b']):p for p in pairs if p['seed']==seed and p['arm']==before and (partition=='all' or p['partition']==partition) and (not strict or p['strict'])}
                    aa={(p['a'],p['b']):p for p in pairs if p['seed']==seed and p['arm']==after}
                    for metric in ('FL','PA','MT','missing'):
                        clusters=defaultdict(list);not_comparable=0
                        for edge,b in bb.items():
                            a=aa[edge]
                            if metric=='missing':v=int(a['either_endpoint_missing'])-int(b['either_endpoint_missing'])
                            else:
                                field='absolute' if metric=='PA' else 'ROI_absolute';x=b['parameters'][metric][field];y=a['parameters'][metric][field]
                                if x is None or y is None:not_comparable+=1;continue
                                v=y-x
                            clusters[b['component']].append(v)
                        v=np.array([np.mean(x) for x in clusters.values()]);ci=None
                        if len(v):
                            meta={c['id']:tuple(c['datasets']) for c in split['components']};strata=defaultdict(list)
                            for cid,vs in clusters.items():strata[meta[cid]].append(float(np.mean(vs)))
                            rng=np.random.default_rng(2026110339);boots=np.zeros(5000)
                            for vs in strata.values():
                                av=np.array(vs);boots+=av[rng.integers(0,len(av),size=(5000,len(av)))].sum(1)
                            boots/=len(v);ci=np.quantile(boots,[.025,.975]).tolist()
                        contrasts.append(dict(seed=seed,after=after,before=before,partition=partition,mode='strict' if strict else 'all_nonidentical',metric=metric,groups=len(v),pairs_total=len(bb),not_numerically_comparable=not_comparable,delta=float(v.mean()) if len(v) else None,ci95=ci))
    gzwrite(DEST/'evaluation/natural_pairs.json.gz',pairs);write(DEST/'evaluation/natural_summary.json',summaries);write(DEST/'evaluation/natural_contrasts.json',contrasts)
    return summaries

def accuracy(rows):
    refs,unclear=validate_reference(REF,{r['image_id']:r for r in rows[SEEDS[0]].values() if r['dataset']=='apo153'})
    assert sha(REF)==j(R37/'evaluation/summary.json')['boundary_reference_sha256']
    results=[];clipping=[]
    for seed in SEEDS:
        ep=ROOT/f'runs/round31_fascicle_guided_pair_v1/evaluation/{seed}/external34.json';expected=j(ROOT/'runs/round32_chain_foundation_v1/source_bindings.json')['consumed_verified_sources'];assert sha(ep)==expected[ep.relative_to(ROOT).as_posix()]
        experts={r['image_id']:r['reference'] for r in j(ep)['rows']}
        for ds in DS:
            rr=[r for r in rows[seed].values() if r['dataset']==ds]
            for r in rr:
                if ds=='external34':r['reference']=experts[r['image_id']]
            for arm in EVAL_ARMS:
                q=dict(seed=seed,dataset=ds,arm=arm,coverage=coverage_summary(rr,arm))
                if ds=='external34':q['errors']=measurement_errors(rr,arm)
                if ds=='apo153':
                    ok=[]
                    for r in rr:
                        if r['image_id'] not in refs:continue
                        a=r['arms'][arm];good,cov,_=f1_score(dict(status='candidate' if a['curve'] else 'rejected',curve=a['curve']),refs[r['image_id']]);a['reference_correct']=bool(good);ok.append(bool(good))
                    q['boundary']=dict(clear_n=len(ok),correct_n=sum(ok),unclear_n=len(unclear))
                results.append(q)
            gzwrite(DEST/f'evaluation/originals/{seed}/{ds}.json.gz',rr)
    # Both-boundary valid-domain clipping on all available clear GT, evaluation only.
    for r in rows[SEEDS[0]].values():
        if r['dataset']!='apo153' or r['image_id'] not in refs:continue
        ref=refs[r['image_id']];rec=j(cachepath(r,'identity','c1').with_suffix('.json'));box=domain(rec['roi'],r['width'],r['height'])
        xx=np.array(ref['reference_x']);side={}
        for s in ('sup','deep'):
            yy=np.array(ref['reference_'+s]);inside=(xx>=box[0])&(xx<box[2])&(yy>=box[1])&(yy<box[3]);side[s]=dict(reference_points=len(xx),outside_count=int((~inside).sum()),outside_fraction=float((~inside).mean()))
        clipping.append(dict(image_id=r['image_id'],box=box,roi_fallback=rec['roi']['fallback'],sides=side))
    write(DEST/'evaluation/roi_reference_audit_source.json',clipping);write(DEST/'evaluation/original_summary.json',results)
    return results

def main():
    start=time.time();verify_preparation();verify_lock();assert j(DEST/'experiment_success.json')['records']==7470
    files={p.relative_to(DEST).as_posix():sha(p) for p in (DEST/'predictions').rglob('*.json.gz')};assert len(files)==7470
    for rel,h in files.items():assert j((DEST/rel).with_suffix('.sha.json'))['sha256']==h
    write(DEST/'evaluation/predictions_frozen_before_reference.json',dict(files=files,lock_sha256=sha(DEST/'implementation_lock.json')))
    split=j(DEST/'partitions.json');rois=j(R37/'roi.json');targets=j(PROTOCOL)['targets'];worsts=[];summaries=[];dsums=[];tsums=[];extreme=[];allrows={}
    for seed in SEEDS:
        rows={};td=defaultdict(list)
        for i,r in enumerate(ordered_inputs()):
            base=gzread(outpath(r,'identity',seed));oldbase=gzread(predpath(r,'identity',seed));orig=base['arms'];orig.update({a:oldbase['arms'][a] for a in ('R','A','NVA')})
            record={k:v for k,v in base.items() if k not in ('arms','pools')};record['arms']={a:reduce_arm(v) for a,v in orig.items()};rows[key(r)]=record
            variants={}
            for tr in transforms(r)[1:]:
                new=gzread(outpath(r,tr,seed));old=gzread(predpath(r,tr,seed));variants[tr]={**new['arms'],**{a:old['arms'][a] for a in ('R','A','NVA')}}
            for arm in EVAL_ARMS:
                dd=[]
                for tr,v in variants.items():
                    d=diff(orig[arm],v[arm],base,rois[key(r)]);td[arm,tr].append(d)
                    if tr in GENTLE:dd.append((tr,d))
                    for metric,p in d['parameters'].items():
                        val=p['absolute'] if metric=='PA' else p['relative']
                        if p['lost'] or p['new'] or (val is not None and val>(3 if metric=='PA' else .2)):
                            extreme.append(dict(seed=seed,arm=arm,key=key(r),transform=tr,metric=metric,gentle=tr in GENTLE,**p))
                w=worst(dd,orig[arm],base,orig['R'],targets);w.update(key=key(r),seed=seed,arm=arm,partition=split['assignment'][key(r)],component=split['component_ids'][key(r)],
                    any_availability_flip=any(p['lost'] or p['new'] for _,d in dd for p in d['parameters'].values()),any_metric_missing=any(p['missing'] for p in w['parameters'].values()),all_original_missing=all(v is None for v in orig[arm]['values'].values()))
                worsts.append(w)
            if (i+1)%50==0:print('EVAL',seed,i+1,flush=True)
        allrows[seed]=rows
        for arm in EVAL_ARMS:
            rr=[r for r in worsts if r['seed']==seed and r['arm']==arm]
            for partition in ('all','development','confirmation'):summaries.append(dict(seed=seed,arm=arm,partition=partition,**summary([r for r in rr if partition=='all' or r['partition']==partition])))
            for ds in DS:dsums.append(dict(seed=seed,arm=arm,dataset=ds,**summary([r for r in rr if r['key'].split('|')[0]==ds])))
        for (arm,tr),rr in td.items():tsums.append(dict(seed=seed,arm=arm,transform=tr,gentle=tr in GENTLE,**transform_summary(rr)))
    gzwrite(DEST/'evaluation/worst_by_image.json.gz',worsts);write(DEST/'evaluation/stability_summary.json',summaries);write(DEST/'evaluation/stability_by_dataset.json',dsums);write(DEST/'evaluation/transform_summary.json',tsums);gzwrite(DEST/'evaluation/extreme_cases.json.gz',extreme)
    contrasts=[('J','B0'),('V','B0'),('C','B0'),('VC','V'),('VC','C'),('Q','VC'),('E','VC'),('W','B0'),('S','B0'),('WS','W'),('WS','S'),('WS','B0')]
    cis=[paired_bootstrap(worsts,split,a,b,p,s) for s in SEEDS for p in ('all','confirmation') for a,b in contrasts]
    write(DEST/'evaluation/paired_intervals.json',cis)
    natural(allrows,split,rois);accuracy(allrows)
    # Exact historical reference replay of reported worst metrics, independent of new method ranking.
    historical=j(R38/'evaluation/stability_summary.json')
    for q in summaries:
        if q['partition']!='all':continue
        name='BNVA' if q['arm']=='B0' else q['arm']
        if name not in ('BNVA','R','A','NVA'):continue
        old=next(x for x in historical if x['seed']==q['seed'] and x['arm']==name)
        for f in ('reference_complete_n','original_complete_n','joint_pass_count','missing_flip_count','severe_image_count','parameters'):check_equal(q[f],old[f],'historical evaluation '+f)
    write(DEST/'evaluation_success.json',dict(records=7470,new_arm_outputs=74700,historical_references=3,seed_count=2,images=469,seconds=time.time()-start,historical_evaluation_replayed=True))
    print('EVALUATION COMPLETE',round(time.time()-start),flush=True)

if __name__=='__main__':main()
