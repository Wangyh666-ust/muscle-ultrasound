"""Stability-first evaluation after all frozen predictions; no tuning."""
import sys,copy,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from scripts.round38_common import *
from scripts.round36_evaluate import stats,coverage,measurement_errors
from scripts.round25_evaluate import REF,validate_reference,f1_score
from scripts.s01_round36_stability import match_components,mapping
from scripts.round29_core import value_at
METRICS=('MT','PA','FL')

def finite(v):return v is not None and math.isfinite(v)
def complete(a):return all(finite(a['values'][p]) for p in METRICS)
def relative(a,b):return 2*abs(a-b)/max(abs(a)+abs(b),1e-12)
def segs(a):return a.get('segments',[c['segment'] for c in a.get('candidates',[])])

def boundary_delta(a,b,roi):
    if a is None or b is None:return dict(comparable=False,missing_flip=(a is None)!=(b is None))
    lo=max(a['support'][0],b['support'][0]);hi=min(a['support'][1],b['support'][1]);span=max(roi['box'][2]-roi['box'][0],1)
    union=max(a['support'][1],b['support'][1])-min(a['support'][0],b['support'][0])
    q=dict(comparable=hi>lo,missing_flip=False,support_overlap=max(0,hi-lo)/max(union,1),support_endpoint_change_ROI=max(abs(np.array(a['support'])-b['support']))/span)
    if hi>lo:
        xs=np.linspace(lo,hi,33)
        q['max_curve_change_ROI']=max(abs(value_at(a[s],x)-value_at(b[s],x)) for s in ('sup','deep') for x in xs)/span
        q['max_tangent_change_deg']=None
    return q

def diff(a,b,row,roi,match=False):
    scale=row.get('px_per_mm') or 1;span=max(roi['box'][2]-roi['box'][0],1)/scale
    q={'parameters':{},'complete_before':complete(a),'complete_after':complete(b)}
    for p in METRICS:
        x,y=a['values'][p],b['values'][p];both=finite(x) and finite(y)
        q['parameters'][p]=dict(before=x,after=y,absolute=abs(x-y) if both else None,relative=relative(x,y) if both and p!='PA' else None,
            ROI_absolute=abs(x-y)/span if both and p!='PA' else None,lost=finite(x) and not finite(y),new=not finite(x) and finite(y),both_missing=not finite(x) and not finite(y))
    if 'curve' in b:q['boundary']=boundary_delta(a['curve'],b['curve'],roi)
    if match:
        pairs=match_components(segs(a),segs(b),row['width'],row['height']);mp={x['label_id']:y['label_id'] for x,y in pairs};ai=a['selection']['selected_ids'];bi=b['selection']['selected_ids']
        q['representative_switch']=any(i not in mp for i in ai) or {mp[i] for i in ai if i in mp}!=set(bi)
        q['matched_tracks']=len(pairs);q['eligible_counts']=[len(segs(a)),len(segs(b))]
    return q

def worst(ds,a,baseline,ref,targets):
    out=dict(original_complete=complete(a),reference_complete=complete(ref),parameters={},joint_pass=True,any_missing_flip=False,severe=False)
    for p in METRICS:
        vv=[(name,d['parameters'][p]) for name,d in ds];val=a['values'][p];origok=finite(val)
        missing=not origok or any(not finite(v['after']) for _,v in vv)
        keyf='absolute' if p=='PA' else 'relative';vvnum=[(n,v[keyf]) for n,v in vv if v[keyf] is not None]
        worstname,maxval=max(vvnum,key=lambda x:x[1]) if vvnum else (None,None)
        roa=max((v['ROI_absolute'] for _,v in vv if v['ROI_absolute'] is not None),default=None)
        limit=targets['PA_deg'] if p=='PA' else targets[p+'_relative']
        passed=not missing and maxval is not None and maxval<=limit and (p!='FL' or roa<=targets['FL_ROI_absolute'])
        severe=maxval is not None and maxval>(targets['severe_PA_deg'] if p=='PA' else targets['severe_relative'])
        lost=origok and any(v['lost'] for _,v in vv)
        out['parameters'][p]=dict(original_valid=origok,reference_original_valid=finite(ref['values'][p]),worst=maxval,worst_transform=worstname,worst_ROI_absolute=roa,
            pass_target=passed,missing=missing,lost_any=lost,severe=severe,all_variants_observed=not missing)
        out['joint_pass'] &= passed;out['any_missing_flip'] |= lost;out['severe'] |= severe
    return out

def summary_worst(records):
    ref=[x for x in records if x['reference_complete']];own=[x for x in records if x['original_complete']]
    q=dict(n=len(records),reference_complete_n=len(ref),original_complete_n=len(own),joint_pass_count=sum(x['joint_pass'] for x in ref),
        joint_pass_fraction=sum(x['joint_pass'] for x in ref)/len(ref) if ref else None,
        pass_among_all_images=sum(x['joint_pass'] for x in records)/len(records),
        original_missing_on_reference_count=sum(not x['original_complete'] for x in ref),
        missing_flip_count=sum(x['any_missing_flip'] for x in own),missing_flip_fraction=sum(x['any_missing_flip'] for x in own)/len(own) if own else None,
        severe_image_count=sum(x['severe'] for x in records),parameters={})
    for p in METRICS:
        pp=[x['parameters'][p] for x in records];fixed=[x for x in pp if x['reference_original_valid']]
        q['parameters'][p]=dict(reference_n=len(fixed),original_valid=sum(x['original_valid'] for x in pp),pass_count=sum(x['pass_target'] for x in fixed),
            pass_fraction=sum(x['pass_target'] for x in fixed)/len(fixed) if fixed else None,lost_any=sum(x['lost_any'] for x in pp),
            missing_on_reference=sum(x['missing'] for x in fixed),severe=sum(x['severe'] for x in pp),
            worst_numeric_observed_only=stats([x['worst'] for x in pp if x['worst'] is not None]),
            worst_ROI_observed_only=stats([x['worst_ROI_absolute'] for x in pp if x['worst_ROI_absolute'] is not None]))
    q['joint_target_met']=q['joint_pass_fraction'] is not None and q['joint_pass_fraction']>=.95 and q['missing_flip_fraction']<=.01 and len(own)>=len(ref)
    return q

def transform_summary(rows):
    q=dict(n=len(rows),parameters={})
    for p in METRICS:
        v=[r['parameters'][p] for r in rows]
        q['parameters'][p]={k:stats([r[k] for r in v if r[k] is not None]) for k in ('absolute','relative','ROI_absolute')}
        q['parameters'][p].update(lost=sum(r['lost'] for r in v),new=sum(r['new'] for r in v),both_missing=sum(r['both_missing'] for r in v))
    bd=[r.get('boundary',{}) for r in rows];q['boundary']={k:stats([r[k] for r in bd if r.get(k) is not None]) for k in ('max_curve_change_ROI','support_overlap','support_endpoint_change_ROI')}
    return q

def natural(allrows,rois):
    groups=j(S01/'groups.json');out=[];details=[]
    for seed,rows in allrows.items():
        for arm in ARMS:
            pairs=[]
            for e in groups['pairs']:
                if e['pixel_identical']:continue
                r,s=rows[e['a']],rows[e['b']];a=r['arms'][arm];b=copy.deepcopy(s['arms'][arm]);ratio=(s.get('px_per_mm') or 1)/(r.get('px_per_mm') or 1)
                for p,dim in [('MT','height'),('FL','width')]:
                    if b['values'][p] is not None:b['values'][p]*=ratio*r[dim]/s[dim]
                b['segments']=[map_segment(x,mapping(r,s,e)) for x in segs(b)];b.pop('candidates',None);b.pop('curve',None)
                d=diff(a,b,r,rois[e['a']],True);d.update(a=e['a'],b=e['b'],group=e['group'],strict=e['strict'],seed=seed,arm=arm);pairs.append(d)
            for strict in (False,True):
                pp=[p for p in pairs if p['strict'] or not strict];q=transform_summary(pp);q.update(seed=seed,arm=arm,mode='strict' if strict else 'all_nonidentical',representative_switches=sum(p['representative_switch'] for p in pp),group_equal={})
                for metric in METRICS:
                    gs=defaultdict(list)
                    for p in pp:
                        v=p['parameters'][metric]['absolute' if metric=='PA' else 'ROI_absolute']
                        if v is not None:gs[p['group']].append(v)
                    q['group_equal'][metric]=float(np.mean([np.mean(v) for v in gs.values()])) if gs else None
                out.append(q)
            details+=pairs
    gzwrite(BASE/'evaluation/natural_pairs.json.gz',details);write(BASE/'evaluation/natural_summary.json',out)

def main():
    verify_code('evaluate');cfg=j(CFG);rois=j(R37/'roi.json');allrows={};worsts=[];transums=[];summaries=[];dataset_summaries=[];cases=[];fixedsums=[];paired=[]
    for seed in SEEDS:assert (BASE/f'chain_{seed}_success.json').exists(),'incomplete predictions'
    files={p.relative_to(BASE).as_posix():sha(p) for p in (BASE/'predictions').rglob('*.json.gz')};assert len(files)==7470
    write(BASE/'evaluation/predictions_frozen_before_reference.json',dict(files=files,config_sha256=sha(CFG)))
    for seed in SEEDS:
        rows={};armworst=defaultdict(list);td=defaultdict(list);fd=defaultdict(list);rowdetails=[]
        for i,r in enumerate(ordered_inputs()):
            base=gzread(predpath(r,'identity',seed));rows[key(r)]=base;originalarms=base['arms'];variants={tr:gzread(predpath(r,tr,seed)) for tr in transforms(r)[1:]}
            local={}
            for arm in ARMS:
                ds=[]
                for tr,v in variants.items():
                    d=diff(originalarms[arm],v['arms'][arm],base,rois[key(r)])
                    td[arm,tr].append(d)
                    if tr in GENTLE:ds.append((tr,d))
                    if v['fixed_boundary']:
                        f=diff(originalarms[arm],v['fixed_boundary'][arm],base,rois[key(r)]);fd[arm,tr].append(f)
                    for p in METRICS:
                        pp=d['parameters'][p];vv=pp['absolute'] if p=='PA' else pp['relative']
                        if pp['lost'] or (vv is not None and vv>(3 if p=='PA' else .2)):
                            cases.append(dict(seed=seed,arm=arm,key=key(r),transform=tr,metric=p,**pp,boundary=d['boundary'],gentle=tr in GENTLE))
                w=worst(ds,originalarms[arm],base,originalarms['R'],cfg['targets']);w.update(key=key(r),seed=seed,arm=arm);armworst[arm].append(w);local[arm]=w;rowdetails.append(w)
            for aa,bb in [('R','G'),('G','N'),('G','V'),('G','A'),('N','NV'),('V','NV'),('NV','NVA'),('NVA','BNVA')]:
                for p in METRICS:
                    av,bv=local[aa]['parameters'][p],local[bb]['parameters'][p]
                    paired.append(dict(seed=seed,key=key(r),before=aa,after=bb,metric=p,before_pass=av['pass_target'],after_pass=bv['pass_target'],before_worst=av['worst'],after_worst=bv['worst'],lost=av['all_variants_observed'] and not bv['all_variants_observed']))
            if (i+1)%50==0:print('EVAL',seed,i+1,469,flush=True)
        allrows[seed]=rows;worsts+=rowdetails
        for arm,rr in armworst.items():
            summaries.append(dict(seed=seed,arm=arm,**summary_worst(rr)))
            for dataset in DS:dataset_summaries.append(dict(seed=seed,arm=arm,dataset=dataset,**summary_worst([r for r in rr if r['key'].split('|')[0]==dataset])))
        for (arm,tr),rr in td.items():transums.append(dict(seed=seed,arm=arm,transform=tr,gentle=tr in GENTLE,**transform_summary(rr)))
        for (arm,tr),rr in fd.items():fixedsums.append(dict(seed=seed,arm=arm,transform=tr,scope='113 similarity-group images only',**transform_summary(rr)))
    gzwrite(BASE/'evaluation/worst_by_image.json.gz',worsts);write(BASE/'evaluation/stability_summary.json',summaries);write(BASE/'evaluation/stability_by_dataset.json',dataset_summaries);write(BASE/'evaluation/transform_summary.json',transums)
    write(BASE/'evaluation/fixed_boundary_summary.json',fixedsums);gzwrite(BASE/'evaluation/extreme_cases.json.gz',cases);gzwrite(BASE/'evaluation/paired_contrasts.json.gz',paired)
    natural(allrows,rois)
    # Reference checks are secondary and occur only after all predictions are frozen.
    assert sha(REF)==j(R37/'evaluation/summary.json')['boundary_reference_sha256'],'boundary reference changed'
    refs,unclear=validate_reference(REF,{r['image_id']:r for r in allrows[SEEDS[0]].values() if r['dataset']=='apo153'})
    accuracy=[]
    for seed in SEEDS:
        ep=ROOT/f'runs/round31_fascicle_guided_pair_v1/evaluation/{seed}/external34.json'
        expected=j(ROOT/'runs/round32_chain_foundation_v1/source_bindings.json')['consumed_verified_sources'];assert sha(ep)==expected[ep.relative_to(ROOT).as_posix()]
        experts={r['image_id']:r['reference'] for r in j(ep)['rows']}
        for ds in DS:
            rows=[r for r in allrows[seed].values() if r['dataset']==ds]
            for r in rows:
                if ds=='external34':r['reference']=experts[r['image_id']]
            for arm in ARMS:
                q=dict(seed=seed,dataset=ds,arm=arm,coverage=coverage(rows,arm))
                if ds=='external34':q['errors']=measurement_errors(rows,arm)
                if ds=='apo153':
                    clear=[r for r in rows if r['image_id'] in refs];correct=[]
                    for r in clear:
                        a=r['arms'][arm];ok,cov,_=f1_score(dict(status='candidate' if a['curve'] else 'rejected',curve=a['curve']),refs[r['image_id']]);a['reference_correct']=bool(ok);correct.append(bool(ok))
                    q['boundary']=dict(clear_n=len(clear),correct_n=sum(correct),unclear_n=len(unclear))
                accuracy.append(q)
            gzwrite(BASE/f'evaluation/originals/{seed}/{ds}.json.gz',rows)
    write(BASE/'evaluation/original_summary.json',accuracy)
    write(BASE/'evaluation_success.json',dict(image_seed_originals=938,image_seed_variants=7470,arm_outputs=59760,all_image_worst_records=len(worsts),extreme_records=len(cases),references_only_secondary=True))
    print('EVALUATION DONE',flush=True)

if __name__=='__main__':main()
