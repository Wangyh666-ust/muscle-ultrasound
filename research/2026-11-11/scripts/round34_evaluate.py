"""Frozen-prediction paired evaluation; no tuning, fixed denominators."""
import argparse,copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from scripts.round34_common import *
from scripts.round25_evaluate import REF,validate_reference,f1_score
from scripts.round25_boundary_fit import curve_errors
from src.umud_baseline.apo_baseline import support_valid_mask

def group_mean(rows,fn):
    g=defaultdict(list)
    for r in rows:
        v=fn(r)
        if v is not None:g[r['group_id']].append(float(v))
    return float(np.mean([np.mean(v) for v in g.values()])) if g else None

def error_metrics(rows,post,param,refmode='mean'):
    available=[r for r in rows if r['arms'][post]['values'][param] is not None]
    err=np.asarray([r['arms'][post]['values'][param]-r['reference'][param][refmode] for r in available],float)
    require(np.isfinite(err).all(),'nonfinite measurement error')
    return {'fixed_n':len(rows),'n':len(err),'MAE':float(abs(err).mean()) if len(err) else None,'bias':float(err.mean()) if len(err) else None,
            'absolute':stats(abs(err)),'gt20_ids':[r['image_id'] for r,e in zip(available,err) if abs(e)>20],
            'gt50_ids':[r['image_id'] for r,e in zip(available,err) if abs(e)>50]}

def compare(a,b,post,param,external=False):
    aa=unique(a,['image_id']);bb=unique(b,['image_id']);require(set(aa)==set(bb),'paired identity set')
    common=[];new=[];lost=[];bothmissing=[];details=[]
    for k,r in aa.items():
        q=bb[k];x=r['arms'][post]['values'][param];y=q['arms'][post]['values'][param]
        if x is None and y is None:bothmissing.append(k[0])
        elif x is None:new.append(k[0])
        elif y is None:lost.append(k[0])
        else:common.append(k[0])
        rec={'image_id':k[0],'before':x,'after':y,'delta':y-x if x is not None and y is not None else None}
        if external:
            gt=r['reference'][param]['mean'];rec.update(reference=gt,error_before=x-gt if x is not None else None,error_after=y-gt if y is not None else None,
                absolute_error_delta=abs(y-gt)-abs(x-gt) if x is not None and y is not None else None)
        details.append(rec)
    out={'fixed_n':len(a),'common_n':len(common),'common_ids':common,'new_ids':new,'lost_ids':lost,'both_missing_ids':bothmissing,
         'absolute_measurement_change':stats([abs(r['delta']) for r in details if r['delta'] is not None]),'per_image':details}
    if external:
        ar=[aa[(x,)] for x in common];br=[bb[(x,)] for x in common]
        out.update(paired_before=error_metrics(ar,post,param),paired_after=error_metrics(br,post,param),
            median_reference_before=error_metrics(ar,post,param,'median'),median_reference_after=error_metrics(br,post,param,'median'),
            improved_ids=[r['image_id'] for r in details if r['absolute_error_delta'] is not None and r['absolute_error_delta']< -1e-7],
            worsened_ids=[r['image_id'] for r in details if r['absolute_error_delta'] is not None and r['absolute_error_delta']> 1e-7])
    require(len(common)+len(new)+len(lost)+len(bothmissing)==len(a),'paired partition')
    return out

def coverage(rows,post,external=False):
    result={p:{'n':sum(r['arms'][post]['values'][p] is not None for r in rows),'fixed_n':len(rows),
            'group_equal':group_mean(rows,lambda r:r['arms'][post]['values'][p] is not None),
            'exploratory_error':error_metrics(rows,post,p) if external else None} for p in ('MT','PA','FL')}
    result['selection']={'FL_invalid':sum(len(r['arms'][post]['selection']['selected_invalid']) for r in rows),
        'PA_invalid':sum(len(r['arms'][post]['PA_invalid']) for r in rows),
        'FL_selected':sum(len(r['arms'][post]['selection']['selected_ids']) for r in rows),
        'FL_ge3_images':sum(len(r['arms'][post]['selection']['selected_ids'])>=3 for r in rows),
        'FL_missing_regions':dict(Counter(x for r in rows for x in r['arms'][post]['selection']['missing_regions'])),
        'FL_over80':sum(len(r['arms'][post]['selection']['selected_over_80']) for r in rows)}
    result['stages']=dict(Counter(r['arms'][post]['stage'] for r in rows));require(sum(result['stages'].values())==len(rows),'stage denominator')
    return result

def score_boundary(rows,refs):
    for r in rows:
        ref=refs.get(r['image_id']);r['target_unclear']=ref is None
        if ref is None:continue
        r['pool_usable']=any(p['curve'] and f1_score({'status':'candidate','curve':p['curve']},ref)[0] for p in r['pool'])
        for post in ('R','S'):
            a=r['arms'][post];ok,cov,_=f1_score({'status':'candidate' if a['curve'] else 'rejected','curve':a['curve']},ref)
            a['reference_correct']=ok;a['reference_coverage']=cov or {'correct':False,'sup_coverage':0.,'deep_coverage':0.}
            a['failure_class']='correct' if ok else 'selection_missed_usable_pair' if r['pool_usable'] else 'no_reference_usable_pair'

def boundary_summary(rows,post):
    clear=[r for r in rows if not r['target_unclear']];require(len(clear)==143,'143 fixed reference denominator')
    return {'all_n':153,'reference_n':143,'output_n':sum(r['arms'][post]['curve'] is not None for r in rows),
        'correct_n':sum(r['arms'][post]['reference_correct'] for r in clear),'correct_group_equal':group_mean(clear,lambda r:r['arms'][post]['reference_correct']),
        'pool_usable_n':sum(r['pool_usable'] for r in clear),'failure_classes':dict(Counter(r['arms'][post]['failure_class'] for r in clear)),
        'coverage':{s:{'image_equal':float(np.mean([r['arms'][post]['reference_coverage'][s+'_coverage'] for r in clear])),
            'group_equal':group_mean(clear,lambda r:r['arms'][post]['reference_coverage'][s+'_coverage'])} for s in ('sup','deep')}}

def boundary_paired(a,b,post,refs):
    aa=unique(a,['image_id']);bb=unique(b,['image_id']);cells={'both_correct':[],'A_only':[],'B_only':[],'neither':[]};per=[];pooled={'sup':{'A':[],'B':[]},'deep':{'A':[],'B':[]}}
    for iid,ref in refs.items():
        x=aa[(iid,)]['arms'][post];y=bb[(iid,)]['arms'][post];u=x['reference_correct'];v=y['reference_correct']
        cells['both_correct' if u and v else 'A_only' if u else 'B_only' if v else 'neither'].append(iid)
        ca,cb=x['curve'],y['curve']
        if ca is None or cb is None:continue
        ea=curve_errors(ca,ref['reference_x'],ref['reference_sup'],ref['reference_deep']);eb=curve_errors(cb,ref['reference_x'],ref['reference_sup'],ref['reference_deep'])
        valid=support_valid_mask(ca,ref['reference_x'])&support_valid_mask(cb,ref['reference_x'])
        valid&=np.isfinite(ea['se'])&np.isfinite(eb['se'])&np.isfinite(ea['de'])&np.isfinite(eb['de'])
        if not valid.any():continue
        item={'image_id':iid,'group_id':aa[(iid,)]['group_id'],'n':int(valid.sum()),'x':np.asarray(ref['reference_x'])[valid].tolist(),'sides':{}}
        for side,col in [('sup','se'),('deep','de')]:
            av=ea[col][valid];bv=eb[col][valid];item['sides'][side]={'A':float(av.mean()),'B':float(bv.mean()),'B_minus_A':float((bv-av).mean())}
            pooled[side]['A'].extend(av);pooled[side]['B'].extend(bv)
        per.append(item)
    require(sum(map(len,cells.values()))==143,'four-way count')
    keys=[(r['image_id'],x) for r in per for x in r['x']];require(len(set(keys))==len(keys),'common x unique')
    return {'cells':cells,'counts':{k:len(v) for k,v in cells.items()},'paired_net_correct':len(cells['B_only'])-len(cells['A_only']),
        'common_position_images':len(per),'unique_common_positions':len(keys),'per_image_common_positions':per,
        'common_distances':{s:{'image_equal':{k:float(np.mean([r['sides'][s][k] for r in per])) if per else None for k in ('A','B','B_minus_A')},
            'group_equal':{k:group_mean(per,lambda r:r['sides'][s][k]) for k in ('A','B','B_minus_A')},
            'position_equal_A':stats(pooled[s]['A']),'position_equal_B':stats(pooled[s]['B'])} for s in ('sup','deep')}}

def fixed_selection(a,b,post):
    bb=unique(b,['image_id']);out=[]
    for r in a:
        q=bb[(r['image_id'],)];sel=r['arms'][post]['selection'];cs={c['segment']['label_id']:c for c in q['arms'][post]['candidates']}
        ids=sel['selected_ids'];vals=[cs[k]['raw_FL'] for k in ids];usable=bool(ids) and all(v is not None for v in vals)
        out.append({'image_id':r['image_id'],'selected_ids_fixed_from_A':ids,'weights_fixed_from_A':sel['weights'],
            'A_FL':r['arms'][post]['values']['FL'],'B_same_lines_FL':sum(v*w for v,w in zip(vals,sel['weights']))/(r['px_per_mm'] or 1) if usable else None,
            'B_reselected_FL':q['arms'][post]['values']['FL'],'B_selected_ids':q['arms'][post]['selection']['selected_ids'],'missing_raw':not usable})
    return out

def freeze_gate():
    files={};allrows={}
    for seed in SEEDS:
        for arm in ('A','B','B_zero_hint'):
            source=BASE/f'chain/{seed}/{arm}';sg=j(source/'success.json');verify(sg['files'],source)
            files[(source/'success.json').relative_to(BASE).as_posix()]=sha(source/'success.json')
            for ds,n in DATASETS.items():
                for tr in (['identity'] if arm=='B_zero_hint' else j(CFG)['transforms']):
                    p=source/ds/(tr+'.json');rows=j(p);require(len(rows)==n,'chain count');unique(rows,['image_id'])
                    allrows[(seed,arm,ds,tr)]=rows;files[p.relative_to(BASE).as_posix()]=sha(p)
    write(BASE/'evaluation/predictions_frozen_before_reference.json',{'files':files,'expert_reference_read':False})
    return allrows

def main():
    out=BASE/'evaluation';out.mkdir(parents=True,exist_ok=True);(out/'success.json').unlink(missing_ok=True)
    datasets=freeze_gate()
    refs,unclear=validate_reference(REF,{r['image_id']:r for r in datasets[(SEEDS[0],'A','apo153','identity')]})
    # Prior independently frozen expert aggregation, with its historical hash.
    expected=j(R32/'source_bindings.json')['consumed_verified_sources'];expert_sources={}
    for seed in SEEDS:
        ep=R31/f'evaluation/{seed}/external34.json';require(sha(ep)==expected[ep.relative_to(ROOT).as_posix()],'expert aggregate changed');expert_sources[str(seed)]=sha(ep)
        er=unique(j(ep)['rows'],['image_id']);require(len(er)==34 and ('im_07_arch',) not in er,'external identities')
        for key,rows in datasets.items():
            if key[0]!=seed:continue
            if key[2]=='apo153':score_boundary(rows,refs)
            elif key[2]=='external34':
                require({r['image_id'] for r in rows}=={x[0] for x in er},'external fixed set')
                for r in rows:r['reference']=er[(r['image_id'],)]['reference']
    summary=[];paired=[];transform=[];ablation=[];fixed=[];details=[]
    for seed in SEEDS:
        for arm in ('A','B'):
            for ds in DATASETS:
                rows=datasets[(seed,arm,ds,'identity')]
                summary.append({'seed':seed,'model_arm':arm,'dataset':ds,'post':{p:{**boundary_summary(rows,p),'measurement':coverage(rows,p)} if ds=='apo153' else coverage(rows,p,ds=='external34') for p in ('R','S')}})
                details.extend({'seed':seed,'model_arm':arm,'dataset':ds,'image_id':r['image_id'],'group_id':r['group_id'],
                    'target_unclear':r.get('target_unclear'),'reference':r.get('reference'),
                    'arms':{p:{k:v for k,v in r['arms'][p].items() if k in ('values','reference_correct','reference_coverage','failure_class','selected_pair_id','stage','selection','PA_invalid')} for p in ('R','S')}} for r in rows)
                for tr in ('scale075','scale125','pad10'):
                    other=datasets[(seed,arm,ds,tr)]
                    transform.append({'seed':seed,'model_arm':arm,'dataset':ds,'transform':tr,
                        'scope':'boundary network + decoder with geometrically transformed fixed C1 pixels/candidates; not C1 network robustness',
                        'post':{p:{'measurement':{param:compare(rows,other,p,param,ds=='external34') for param in ('MT','PA','FL')},
                                  'boundary':boundary_paired(rows,other,p,refs) if ds=='apo153' else None,
                                  'coverage':boundary_summary(other,p) if ds=='apo153' else coverage(other,p,ds=='external34')} for p in ('R','S')}})
        for ds in DATASETS:
            a=datasets[(seed,'A',ds,'identity')];b=datasets[(seed,'B',ds,'identity')];zero=datasets[(seed,'B_zero_hint',ds,'identity')]
            paired.append({'seed':seed,'dataset':ds,'post':{p:{'measurement':{param:compare(a,b,p,param,ds=='external34') for param in ('MT','PA','FL')},
                'boundary':boundary_paired(a,b,p,refs) if ds=='apo153' else None} for p in ('R','S')}})
            ablation.append({'seed':seed,'dataset':ds,'direction':'trained B normal input -> same B with hint zero; not retrained A',
                'post':{p:{'measurement':{param:compare(b,zero,p,param,ds=='external34') for param in ('MT','PA','FL')},
                'boundary':boundary_paired(b,zero,p,refs) if ds=='apo153' else None} for p in ('R','S')}})
            fixed.extend({'seed':seed,'dataset':ds,'post':p,**r} for p in ('R','S') for r in fixed_selection(a,b,p))
        print('evaluation seed',seed,flush=True)
    overlap=j(BASE/'inputs/c1_training_overlap.json');clean_ids={r['image_id'] for r in overlap['rows'] if r['dataset']=='dev282' and r['status']=='no_known_overlap'}
    require(len(clean_ids)==276,'C1-clean dev sensitivity denominator');sensitivity=[]
    for seed in SEEDS:
        a=[r for r in datasets[(seed,'A','dev282','identity')] if r['image_id'] in clean_ids]
        b=[r for r in datasets[(seed,'B','dev282','identity')] if r['image_id'] in clean_ids]
        sensitivity.append({'seed':seed,'n':276,'scope':'Known C1-training-overlap exclusion only; main282 retained; not patient independence',
            'post':{p:{'A':coverage(a,p),'B':coverage(b,p),'paired':{k:compare(a,b,p,k) for k in ('MT','PA','FL')}} for p in ('R','S')}})
    write(out/'clean_dev_sensitivity.json',sensitivity)
    write(out/'summary.json',{'rows':summary,'primary':'A-R vs B-R; same-seed paired','secondary':'S','formal_error':None,'units':'unconfirmed; assumed_units_exploratory',
        'expert_reference_hashes':expert_sources,'apo_reference_sha256':sha(REF),'reference_unclear':sorted(unclear),'seeds_are_not_independent_patients':True})
    write(out/'paired.json',paired);write(out/'transforms.json',transform);write(out/'zero_hint_ablation.json',ablation);write(out/'fixed_selection.json',fixed);write(out/'details.json',details)
    cases=[]
    for pair in paired:
        if pair['dataset']!='external34':continue
        for post in ('R','S'):
            for param,c in pair['post'][post]['measurement'].items():
                common=[r for r in c['per_image'] if r['absolute_error_delta'] is not None]
                cases.append({'seed':pair['seed'],'post':post,'parameter':param,'common_n':c['common_n'],
                    'top_improved':sorted([r for r in common if r['absolute_error_delta']< -1e-7],key=lambda r:r['absolute_error_delta'])[:3],
                    'top_degraded':sorted([r for r in common if r['absolute_error_delta']>1e-7],key=lambda r:-r['absolute_error_delta'])[:3],
                    'new_and_lost':[r for r in c['per_image'] if r['image_id'] in c['new_ids']+c['lost_ids']],
                    'worst_before':sorted([r for r in c['per_image'] if r['error_before'] is not None],key=lambda r:-abs(r['error_before']))[:3],
                    'worst_after':sorted([r for r in c['per_image'] if r['error_after'] is not None],key=lambda r:-abs(r['error_after']))[:3],
                    'predeclared_anchors':[r for r in c['per_image'] if r['image_id'] in ('im_06_arch','im_12_arch','im_20_arch')]})
    write(out/'cases.json',cases)
    write(out/'success.json',{'files':{p.name:sha(p) for p in out.glob('*.json') if p.name not in ('success.json','validation_failed.json')}})
    print('EVALUATION COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:fail(BASE/'evaluation',e);raise
