"""Prespecified stability endpoints; predictions frozen before reference access."""
import sys,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
from scripts.round41_measure import *
from scripts.round38_evaluate import diff,worst,complete,transform_summary,METRICS
from scripts.round39_evaluate import summary,reduce_arm,paired_bootstrap
from scripts.round36_evaluate import stats
SCOPES={'old11':GENTLE+NEW,'new5':NEW41,'combined16':GENTLE+NEW+NEW41,'stress4':STRESS}

def crossed_interval(records,split,after,before,partition,scope):
    """Resample image components once, then equal-average boundary conditions."""
    by={(s,a):{r['key']:r for r in records if r['boundary_seed']==s and r['arm']==a and r['scope']==scope and (partition=='all' or r['partition']==partition)} for s in SEEDS for a in (after,before)}
    meta={c['id']:tuple(c['datasets']) for c in split['components']};metrics=('joint_pass','original_complete','any_availability_flip','severe','any_metric_missing');clusters=defaultdict(lambda:np.zeros((2,len(metrics),2)))
    for si,s in enumerate(SEEDS):
        for k,b in by[s,before].items():
            a=by[s,after][k];v=clusters[split['component_ids'][k]]
            for mi,m in enumerate(metrics):
                if m=='joint_pass' and not b['reference_complete']:continue
                v[si,mi,0]+=int(a[m])-int(b[m]);v[si,mi,1]+=1
    strata=defaultdict(list)
    for cid,a in clusters.items():strata[meta[cid]].append(a)
    total=sum(clusters.values());boots=np.zeros((5000,2,len(metrics),2));rng=np.random.default_rng(2026110441)
    for arrays in strata.values():
        a=np.array(arrays);boots+=a[rng.integers(0,len(a),(5000,len(a)))].sum(1)
    output={}
    for i,m in enumerate(metrics):
        delta=total[:,i,0]/total[:,i,1];vals=(boots[:,:,i,0]/boots[:,:,i,1]).mean(1)
        output[m]=dict(delta_fraction=float(delta.mean()),ci95=np.quantile(vals,[.025,.975]).tolist(),boundary_deltas=delta.tolist(),boundary_denominators=total[:,i,1].astype(int).tolist())
    return dict(after=after,before=before,partition=partition,scope=scope,components=len(clusters),metrics=output,unit='same image component resampled jointly across both fixed boundary conditions; conditions equally weighted')

def natural(identities,split,rois):
    groups=j(S01/'groups.json');pairs=[];det=[]
    for s,rows in identities.items():
        for e in groups['pairs']:
            r,t=rows[e['a']],rows[e['b']]
            for arm in KINDS:
                a=r['arms'][arm];b=copy.deepcopy(t['arms'][arm])
                if e['pixel_identical']:
                    # Identical image pixels may have distinct externally supplied
                    # physical scale; compare native-pixel measurements.
                    va={m:(v*(r.get('px_per_mm') or 1) if m!='PA' and v is not None else v) for m,v in a['values'].items()}
                    vb={m:(v*(t.get('px_per_mm') or 1) if m!='PA' and v is not None else v) for m,v in b['values'].items()}
                    check_equal(va,vb,'identical pixels '+e['a']+' '+e['b']+' '+arm)
                    det.append(dict(boundary_seed=s,arm=arm,a=e['a'],b=e['b'],passed=True));continue
                ratio=(t.get('px_per_mm') or 1)/(r.get('px_per_mm') or 1)
                for m,dim in (('MT','height'),('FL','width')):
                    if b['values'][m] is not None:b['values'][m]*=ratio*r[dim]/t[dim]
                b.pop('curve',None);d=diff(a,b,r,rois[e['a']]);d.update(boundary_seed=s,arm=arm,a=e['a'],b=e['b'],group=e['group'],component=split['component_ids'][e['a']],strict=e['strict'],partition=split['assignment'][e['a']],either_endpoint_missing=not(complete(a) and complete(b)))
                pairs.append(d)
    gzwrite(DEST/'evaluation/training/natural_pairs.json.gz',pairs);write(DEST/'evaluation/training/identical_pixel_determinism.json',det)
    out=[]
    for s in SEEDS:
        for arm in KINDS:
            for partition in ('all','confirmation'):
                for strict in (False,True):
                    pp=[p for p in pairs if p['boundary_seed']==s and p['arm']==arm and (partition=='all' or p['partition']==partition) and (not strict or p['strict'])]
                    rec=dict(boundary_seed=s,arm=arm,partition=partition,mode='strict18' if strict else 'all117',pairs=len(pp),either_endpoint_missing=sum(p['either_endpoint_missing'] for p in pp),group_equal={})
                    for m in METRICS:
                        vals=defaultdict(list)
                        for p in pp:
                            v=p['parameters'][m]['absolute' if m=='PA' else 'ROI_absolute']
                            if v is not None:vals[p['component']].append(v)
                        rec['group_equal'][m]=float(np.mean([np.mean(v) for v in vals.values()])) if vals else None
                    out.append(rec)
    write(DEST/'evaluation/training/natural_summary.json',out)

def main():
    verify_lock();assert (DEST/'measurement_success.json').exists();inputs={}
    for r in ordered_inputs():
        for s in SEEDS:
            p=resultpath(r,s);meta=j(p.with_suffix('.sha.json'));assert sha(p)==meta['sha256'];inputs[p.relative_to(DEST).as_posix()]=meta['sha256']
    write(DEST/'evaluation/training/predictions_frozen_before_reference.json',dict(files=inputs,training_lock_sha256=sha(DEST/'training_lock.json')))
    split=j(DEST/'partitions.json');rois=j(R37/'roi.json');targets=j(CFG)['targets'];records=[];events=[];transforms_rows=defaultdict(list);identities={};recoveries=[];bridge=[];start=time.time()
    for s in SEEDS:
        rr={}
        for i,r in enumerate(ordered_inputs()):
            q=gzread(resultpath(r,s));orig=q['variants']['identity']['arms'];ref=gzread(predpath(r,'identity',s))['arms']['R'];ds=defaultdict(list)
            row={k:v for k,v in q.items() if k not in ('variants','sources')};row['arms']={a:reduce_arm(b) for a,b in orig.items()};rr[key(r)]=row
            for tr,v in q['variants'].items():
                if tr=='identity':continue
                for arm,a in v['arms'].items():
                    d=diff(orig[arm],a,q,rois[key(r)]);ds[arm].append((tr,d));transforms_rows[s,arm,tr].append(d)
                    for m,p in d['parameters'].items():
                        z=p['absolute'] if m=='PA' else p['relative']
                        if p['lost'] or p['new'] or (z is not None and z>(3 if m=='PA' else .2)):events.append(dict(key=key(r),boundary_seed=s,arm=arm,transform=tr,metric=m,**p))
                for arm in KINDS[1:]:
                    b=v['arms']['F'];a=v['arms'][arm]
                    if not complete(b) and complete(a):recoveries.append(dict(key=key(r),boundary_seed=s,arm=arm,transform=tr,before=b['values'],after=a['values'],boundary_recovery=b['curve'] is None))
                if v['historical_mask_exact'] is not None:bridge.append(dict(key=key(r),boundary_seed=s,transform=tr,mask_exact=v['historical_mask_exact'],measurement_diff=diff(v['arms']['historical_K'],v['arms']['F'],q,rois[key(r)])['parameters']))
            for arm in orig:
                for scope,trs in SCOPES.items():
                    if arm=='historical_K' and scope in ('new5','combined16'):continue
                    selected=[(tr,d) for tr,d in ds[arm] if tr in trs]
                    if not selected:continue
                    w=worst(selected,orig[arm],q,ref,targets);w.update(key=key(r),seed=s,boundary_seed=s,arm=arm,scope=scope,partition=split['assignment'][key(r)],component=split['component_ids'][key(r)],any_availability_flip=any(p['lost'] or p['new'] for _,d in selected for p in d['parameters'].values()),any_metric_missing=any(p['missing'] for p in w['parameters'].values()),all_original_missing=all(x is None for x in orig[arm]['values'].values()))
                    records.append(w)
            if i%50==0:print('TRAIN_EVAL',s,i,round(time.time()-start),flush=True)
        identities[s]=rr
    sums=[];dataset=[]
    for scope in SCOPES:
        for s in SEEDS:
            for arm in KINDS+('historical_K',):
                rs=[r for r in records if r['scope']==scope and r['boundary_seed']==s and r['arm']==arm]
                if not rs:continue
                for partition in ('all','development','confirmation'):sums.append(dict(scope=scope,boundary_seed=s,arm=arm,partition=partition,**summary([r for r in rs if partition=='all' or r['partition']==partition])))
                for d in DS:
                    dsrows=[r for r in rs if r['key'].split('|')[0]==d]
                    dataset.append(dict(scope=scope,boundary_seed=s,arm=arm,dataset=d,**(summary(dsrows) if dsrows else {'n':0,'not_applicable':True})))
    write(DEST/'evaluation/training/stability_summary.json',sums);write(DEST/'evaluation/training/stability_by_dataset.json',dataset);gzwrite(DEST/'evaluation/training/worst_by_image.json.gz',records)
    gzwrite(DEST/'evaluation/training/extreme_events.json.gz',events);gzwrite(DEST/'evaluation/training/variant_recoveries.json.gz',recoveries);gzwrite(DEST/'evaluation/training/environment_bridge.json.gz',bridge);gzwrite(DEST/'evaluation/training/identities.json.gz',identities)
    write(DEST/'evaluation/training/transform_summary.json',[dict(boundary_seed=s,arm=a,transform=tr,**transform_summary(v)) for (s,a,tr),v in transforms_rows.items()])
    intervals=[]
    for seed in SEEDS:
        for after,before in ((f'A_{seed}','F'),(f'B_{seed}',f'A_{seed}'),(f'B_{seed}','F')):
            for scope in SCOPES:
                for partition in ('all','confirmation'):intervals.append(dict(training_seed=seed,**crossed_interval(records,split,after,before,partition,scope)))
    write(DEST/'evaluation/training/primary_crossed_intervals.json',intervals);natural(identities,split,rois)
    print('TRAINING_STABILITY_EVALUATION_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
