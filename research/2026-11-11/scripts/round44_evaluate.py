"""All469 paired stability, missingness, extremes, true similar pairs."""
from scripts.round43_cpu_imports import install as cpu_install
cpu_install()
from collections import defaultdict
from scripts.round44_common import *
from scripts.round38_evaluate import diff,worst,complete,METRICS,transform_summary
from scripts.round39_evaluate import summary
from scripts.round39_verify import check_equal
from scripts.round41_eval_training import crossed_interval
# The same boundary pair is evaluated repeatedly across C1/group arms.
# Cache this pure diagnostic by all arguments; numeric formulas are unchanged.
import scripts.round38_evaluate as _stability
from scripts.round43_fast import memo as _exact_memo
if not getattr(_stability.boundary_delta,'_round43_memo',False):
    _stability.boundary_delta=_exact_memo(_stability.boundary_delta,512)
ARMS=tuple(k+'|'+g for k in KINDS for g in GROUPS)
E=DEST/'evaluation'

def readrow(r,s):
    q=gzread(resultpath(r,s));n=gzread(DEST/f"new/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz")
    q['variants'].update(n['variants']);return q

def main():
    register();lock=j(DEST/'lock.json')
    for p,h in lock['sources'].items():assert sha(ROOT/p)==h,p
    frozen={}
    for folder in ('measurements','new'):
        for r in rows():
            for s in SEEDS:
                p=DEST/f"{folder}/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz";m=j(p.with_suffix('.sha.json'));assert sha(p)==m['sha256'] and m['binding']['lock']==sha(DEST/'lock.json');frozen[p.relative_to(DEST).as_posix()]=m['sha256']
    assert len(frozen)==1876;write(E/'predictions_frozen_before_reference.json',dict(files=frozen,lock_sha256=sha(DEST/'lock.json')))
    oldworst={(x['key'],x['boundary_seed'],x['arm'],x['scope']):x for x in gzread(R43/'evaluation/worst_by_image.json.gz')}
    rois=j(R37/'roi.json');targets=j(CFG)['targets'];records=[];events=[];identities={};local={};recoveries=[];trstats=defaultdict(list);groups=[];parities=0
    for s in SEEDS:
        rr={};ll={}
        for i,r in enumerate(rows()):
            q=readrow(r,s);orig=q['variants']['identity']['arms'];loc=q['variants']['local_identity']['arms'];dd=defaultdict(list)
            base={k:q[k] for k in ('dataset','image_id','image_path','width','height','px_per_mm','group_id','pixel_sha256')}
            rr[key(r)]={**base,'arms':orig};ll[key(r)]={**base,'arms':loc}
            refrec=oldworst[key(r),s,KINDS[0]+'|R','combined16'];ref=dict(values={m:0. if refrec['parameters'][m]['reference_original_valid'] else None for m in METRICS})
            for tr,v in q['variants'].items():
                if key(r) in MISSING|TAIL:groups.append(dict(key=key(r),set='original_missing11' if key(r) in MISSING else 'padding_tail11',boundary_seed=s,transform=tr,field=v['field'],boundary_audit=v['boundary_audit'],arms={a:dict(values=b['values'],stage=b['stage'],selected_pair_id=b['selected_pair_id'],gates=b['gates'],curve=b['curve']) for a,b in v['arms'].items()}))
                for kind in KINDS:
                    b=v['arms'][kind+'|B']
                    for g in GROUPS[1:]:
                        a=v['arms'][kind+'|'+g];new=[m for m in METRICS if b['values'][m] is None and a['values'][m] is not None];lost=[m for m in METRICS if b['values'][m] is not None and a['values'][m] is None]
                        if new or lost:recoveries.append(dict(key=key(r),boundary_seed=s,arm=kind+'|'+g,transform=tr,new=new,lost=lost,before=b['values'],after=a['values'],curve=a['curve']))
                if tr in ('identity','local_identity'):continue
                anchor=loc if tr in NEW43+NEW44 else orig
                for arm,a in v['arms'].items():
                    d=diff(anchor[arm],a,q,rois[key(r)]);dd[arm].append((tr,d));trstats[s,arm,tr].append(d)
                    for m,p in d['parameters'].items():
                        z=p['absolute'] if m=='PA' else p['relative']
                        if p['lost'] or p['new'] or z is not None and z>(3 if m=='PA' else .2):events.append(dict(key=key(r),boundary_seed=s,arm=arm,transform=tr,metric=m,**p))
            for arm in ARMS:
                for scope,trs in SCOPES.items():
                    sel=[(tr,d) for tr,d in dd[arm] if tr in trs]
                    if not sel:continue
                    w=worst(sel,(loc if scope.startswith('new_') or scope=='r44_new' else orig)[arm],q,ref,targets)
                    w.update(key=key(r),seed=s,boundary_seed=s,arm=arm,scope=scope,partition=SPLIT['assignment'][key(r)],component=SPLIT['component_ids'][key(r)],
                        any_availability_flip=any(p['lost'] or p['new'] for _,d in sel for p in d['parameters'].values()),any_metric_missing=any(p['missing'] for p in w['parameters'].values()),all_original_missing=all(x is None for x in orig[arm]['values'].values()))
                    records.append(w)
                    if arm.split('|')[1] in ('H','B') and scope in OLD_SCOPES:
                        ow=oldworst[key(r),s,arm.split('|')[0]+'|'+('R' if arm.endswith('|H') else 'ID'),scope]
                        for f in ('parameters','joint_pass','original_complete','reference_complete','any_availability_flip','severe'):check_equal(w[f],ow[f],'R43 baseline stats '+f)
                        parities+=1
            if i%40==0:print('R44_EVAL',s,i,flush=True)
        identities[s]=rr;local[s]=ll
    gzwrite(E/'worst_by_image.json.gz',records);gzwrite(E/'extreme_events.json.gz',events);gzwrite(E/'recoveries.json.gz',recoveries)
    gzwrite(E/'identities.json.gz',identities);gzwrite(E/'local_identities.json.gz',local);gzwrite(E/'two11_full_audit.json.gz',groups)
    sums=[];datasets=[]
    for scope in SCOPES:
        for s in SEEDS:
            for arm in ARMS:
                rs=[x for x in records if x['scope']==scope and x['boundary_seed']==s and x['arm']==arm]
                for part in ('all','development','confirmation'):
                    sub=[r for r in rs if part=='all' or r['partition']==part]
                    if sub:sums.append(dict(scope=scope,boundary_seed=s,arm=arm,partition=part,**summary(sub)))
                for ds in DS:
                    sub=[r for r in rs if r['key'].split('|')[0]==ds]
                    if sub:datasets.append(dict(scope=scope,boundary_seed=s,arm=arm,dataset=ds,**summary(sub)))
    write(E/'stability_summary.json',sums);write(E/'stability_by_dataset.json',datasets)
    write(E/'transform_summary.json',[dict(boundary_seed=s,arm=a,transform=tr,**transform_summary(v)) for (s,a,tr),v in trstats.items()])
    # Main CI includes all images. Historical-reference denominator remains in summaries.
    allref=[{**r,'reference_complete':True} for r in records];ci=[]
    for kind in KINDS:
        for a,b in (('LE','B'),('L','B'),('E','B'),('LE','H')):
            for part in ('all','confirmation'):
                
                for scope in ('combined16','new_gray','r44_new','all23'):
                    ci.append(dict(kind=kind,**crossed_interval(allref,SPLIT,kind+'|'+a,kind+'|'+b,part,scope)))
    write(E/'paired_stability_intervals.json',ci)
    natural(identities,rois)
    write(E/'evaluation_success.json',dict(prediction_files=1876,records=len(records),old_baseline_statistics_exact=parities,outputs=sum(len(q['variants'])*len(ARMS) for r in rows() for s in SEEDS for q in [readrow(r,s)])))
    print('ROUND44_EVALUATED',len(records),flush=True)

def natural(identities,rois):
    pairs=[];det=[];gg=j(S01/'groups.json')
    for s,rr in identities.items():
        for e in gg['pairs']:
            r,t=rr[e['a']],rr[e['b']]
            for arm in ARMS:
                a=r['arms'][arm];b=copy.deepcopy(t['arms'][arm])
                if e['pixel_identical']:
                    va={m:(v*(r.get('px_per_mm') or 1) if m!='PA' and v is not None else v) for m,v in a['values'].items()};vb={m:(v*(t.get('px_per_mm') or 1) if m!='PA' and v is not None else v) for m,v in b['values'].items()}
                    check_equal(va,vb,'R43 identical '+e['a']+' '+e['b']+' '+arm);det.append(dict(boundary_seed=s,arm=arm,a=e['a'],b=e['b'],passed=True));continue
                ratio=(t.get('px_per_mm') or 1)/(r.get('px_per_mm') or 1)
                for m,dim in (('MT','height'),('FL','width')):
                    if b['values'][m] is not None:b['values'][m]*=ratio*r[dim]/t[dim]
                b.pop('curve',None);d=diff(a,b,r,rois[e['a']]);d.update(boundary_seed=s,arm=arm,a=e['a'],b=e['b'],group=e['group'],component=SPLIT['component_ids'][e['a']],strict=e['strict'],partition=SPLIT['assignment'][e['a']],either_endpoint_missing=not(complete(a) and complete(b)));pairs.append(d)
    gzwrite(E/'natural_pairs.json.gz',pairs);write(E/'identical_pixel_determinism.json',det);sums=[]
    for s in SEEDS:
        for arm in ARMS:
            for part in ('all','confirmation'):
                for strict in (False,True):
                    pp=[p for p in pairs if p['boundary_seed']==s and p['arm']==arm and(part=='all' or p['partition']==part) and(not strict or p['strict'])];rec=dict(boundary_seed=s,arm=arm,partition=part,mode='strict18' if strict else 'all117',pairs=len(pp),either_endpoint_missing=sum(p['either_endpoint_missing'] for p in pp),group_equal={})
                    for m in METRICS:
                        vals=defaultdict(list)
                        for p in pp:
                            v=p['parameters'][m]['absolute' if m=='PA' else 'ROI_absolute']
                            if v is not None:vals[p['component']].append(v)
                        rec['group_equal'][m]=float(np.mean([np.mean(v) for v in vals.values()])) if vals else None
                    sums.append(rec)
    write(E/'natural_summary.json',sums)

if __name__=='__main__':main()
