"""Reference evaluation runs strictly after frozen Round36 predictions."""
import sys,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
import numpy as np
from scripts.round36_run import BASE,CFG,ARMS,j,sha,write,read_gzip,save_gzip,require,code_binding
from scripts.round25_evaluate import REF,validate_reference,f1_score
from scripts.round25_boundary_fit import curve_errors
from src.umud_baseline.apo_baseline import support_valid_mask
from scripts.round34_evaluate import compare,boundary_paired,group_mean

TOL={'PA':6.,'FL':12.,'MT':3.}
LABEL={'D0':'原D0','D1':'原D1基线','M':'D0/D1候选合并','C':'合并+C1选带','P':'合并+MT假设','F':'合并+多片段方向','CP':'合并+C1+MT假设','CF':'合并+C1+多片段方向','CPF':'全部组合'}


def stats(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a,.9)) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None,max=float(a.max()) if len(a) else None)


def measurement_errors(rows,arm,fallback=False,refmode='mean'):
    result={};per={p:[] for p in TOL};allcomplete=True;perimage=[];fills=Counter()
    for r in rows:
        errors={}
        for p,t in TOL.items():
            v=r['arms'][arm]['values'][p]
            if v is None and fallback:v=r['arms']['D1']['values'][p];fills[p]+=1
            if v is None:allcomplete=False;continue
            e=v-r['reference'][p][refmode];per[p].append((r['image_id'],e));errors[p]=e
        score=sum(abs(errors[p])/TOL[p] for p in TOL)/3 if len(errors)==3 else None
        perimage.append({'image_id':r['image_id'],'group_id':r['group_id'],'errors':errors,'normalized_score':score})
    primary=secondary=tertiary=0.
    for p,vals in per.items():
        err=np.array([e for i,e in vals]);ab=abs(err)
        result[p]={'n':len(vals),'fixed_n':len(rows),'MAE':float(ab.mean()) if len(ab) else None,'bias':float(err.mean()) if len(err) else None,
                   'absolute':stats(ab),'gt_tolerance_ids':[i for i,e in vals if abs(e)>TOL[p]],
                   'gt_twice_tolerance_ids':[i for i,e in vals if abs(e)>2*TOL[p]],'gt50_ids':[i for i,e in vals if abs(e)>50]}
        if len(ab):
            primary+=ab.mean()/TOL[p]/3;secondary+=np.median(ab)/TOL[p]/3;tertiary+=np.sqrt(np.mean(err**2))/TOL[p]/3
    return {'parameters':result,'full_cohort_score':float(primary+1e-6*secondary+1e-9*tertiary) if allcomplete else None,
            'primary':float(primary) if allcomplete else None,'fallback_cells':dict(fills),'per_image':perimage,'fixed_n':len(rows),
            'formal_competition_score':False,'reference_units_unconfirmed':True}


def coverage(rows,arm):
    aa=[r['arms'][arm] for r in rows];selection=[a['selection'] for a in aa]
    return {'n':len(rows),'boundary':sum(a['curve'] is not None for a in aa),
            'parameters':{p:sum(a['values'][p] is not None for a in aa) for p in TOL},
            'complete':sum(all(a['values'][p] is not None for p in TOL) for a in aa),
            'stages':dict(Counter(a['stage'] for a in aa)),
            'FL_invalid_images':sum(bool(s['selected_invalid']) for s in selection),
            'PA_invalid_images':sum(bool(a['PA_invalid']) for a in aa),
            'FL_over80_images':sum(bool(s['selected_over_80']) for s in selection),
            'FL_ge3':sum(len(s['selected_ids'])>=3 for s in selection),
            'missing_regions':dict(Counter(x for s in selection for x in s['missing_regions'])),
            'prior_applied':sum(a.get('prior_applied',False) for a in aa),
            'direction_changed_images':sum(bool(a.get('direction_diagnostic',{}).get('changed_ids')) for a in aa),
            'direction_rejected_anchors':sum(len(a.get('direction_diagnostic',{}).get('rejected_anchor_ids',[])) for a in aa),
            'FL_over_2_image_width':sum(a['values']['FL'] is not None and a['values']['FL']*(r.get('px_per_mm') or 1)>2*r['width'] for r,a in zip(rows,aa)),
            'FL_over_5_image_width':sum(a['values']['FL'] is not None and a['values']['FL']*(r.get('px_per_mm') or 1)>5*r['width'] for r,a in zip(rows,aa)),
            'distribution':{p:stats([a['values'][p] for a in aa if a['values'][p] is not None]) for p in TOL}}


def boundary_score(rows,refs,arm):
    clear=[r for r in rows if r['image_id'] in refs];errors={'sup':[],'deep':[],'thickness':[]}
    for r in clear:
        a=r['arms'][arm];cv=a['curve'];ref=refs[r['image_id']]
        ok,cov,_=f1_score({'status':'candidate' if cv else 'rejected','curve':cv},ref)
        a['reference_correct']=bool(ok);a['reference_coverage']=cov or {'sup_coverage':0.,'deep_coverage':0.}
        a['failure_class']='correct' if ok else 'selection_missed_usable_pair' if r['pool_usable'] else 'no_reference_usable_pair'
        if cv:
            xs=np.array(ref['reference_x']);valid=support_valid_mask(cv,xs);ee=curve_errors(cv,xs,ref['reference_sup'],ref['reference_deep'])
            valid &= np.isfinite(ee['se'])&np.isfinite(ee['de'])
            if valid.any():
                for s,k in [('sup','se'),('deep','de')]:errors[s].append(float(ee[k][valid].mean()))
                from scripts.round29_core import value_at
                gt=np.array(ref['reference_deep'])-np.array(ref['reference_sup']);pr=np.array([value_at(cv['deep'],x)-value_at(cv['sup'],x) for x in xs])
                errors['thickness'].append(float(np.abs(pr[valid]-gt[valid]).mean()))
    return {'reference_n':len(clear),'all_n':len(rows),'correct_n':sum(r['arms'][arm]['reference_correct'] for r in clear),
            'pool_usable_n':sum(r['pool_usable'] for r in clear),'failure_classes':dict(Counter(r['arms'][arm]['failure_class'] for r in clear)),
            'side_coverage':{s:float(np.mean([r['arms'][arm]['reference_coverage'].get(s+'_coverage',0) for r in clear])) for s in ('sup','deep')},
            'observed_pixel_errors_image_equal':{s:stats(v) for s,v in errors.items()},'pixel_errors_are_output_conditional':True}


def bootstrap_score(a,b):
    av={r['image_id']:r for r in a['per_image']};bv={r['image_id']:r for r in b['per_image']};groups=defaultdict(list)
    for iid,x in av.items():
        y=bv[iid]
        if x['normalized_score'] is not None and y['normalized_score'] is not None:groups[x['group_id']].append(y['normalized_score']-x['normalized_score'])
    if not groups:return None
    g=list(groups.values());rng=np.random.default_rng(20261103);means=[]
    for _ in range(2000):
        sample=[z for k in rng.integers(len(g),size=len(g)) for z in g[k]];means.append(np.mean(sample))
    return {'mean_delta':float(np.mean([z for v in g for z in v])),'cluster_bootstrap_95':np.quantile(means,[.025,.975]).tolist(),'groups':len(g),'replicates':2000,'not_independent_blind_validation':True}


def main():
    cfg=j(CFG);sg=j(BASE/'predictions/success.json');require(sg['binding_sha256']==sha(BASE/'binding.json'),'prediction binding')
    for rel,h in sg['files'].items():require(sha(BASE/rel)==h,'frozen predictions')
    out=BASE/'evaluation';out.mkdir(exist_ok=True)
    write(out/'predictions_frozen_before_reference.json',{'files':sg['files'],'binding_sha256':sg['binding_sha256']})
    refs,unclear=validate_reference(REF,{r['image_id']:r for r in read_gzip(BASE/f'predictions/{cfg["seeds"][0]}/apo153.json.gz')})
    summary=[];contrasts=[];refhash={};alldetails=[]
    clean_ids={r['image_id'] for r in j(ROOT/'runs/round34_boundary_guidance_v1/inputs/c1_training_overlap.json')['rows'] if r['dataset']=='dev282' and r['status']=='no_known_overlap'}
    require(len(clean_ids)==276,'clean C1 sensitivity')
    for seed in cfg['seeds']:
        erpath=ROOT/f'runs/round31_fascicle_guided_pair_v1/evaluation/{seed}/external34.json'
        if not erpath.exists():
            from scripts.round35_common import R31
            erpath=R31/f'evaluation/{seed}/external34.json'
        from scripts.round35_common import R32
        expected=j(R32/'source_bindings.json')['consumed_verified_sources']
        require(sha(erpath)==expected[erpath.relative_to(ROOT).as_posix()],'frozen external references')
        refhash[str(seed)]=sha(erpath);experts={r['image_id']:r['reference'] for r in j(erpath)['rows']}
        for ds,n in cfg['datasets'].items():
            rows=read_gzip(BASE/f'predictions/{seed}/{ds}.json.gz');require(len(rows)==n,'full cohort')
            if ds=='external34':
                for r in rows:r['reference']=experts[r['image_id']]
            if ds=='apo153':
                for r in rows:
                    r['target_unclear']=r['image_id'] not in refs
                    r['pool_usable']=any(f1_score({'status':'candidate','curve':p['curve']},refs[r['image_id']])[0] for p in r['pool']) if not r['target_unclear'] else None
            for arm in ARMS:
                rec={'seed':seed,'dataset':ds,'arm':arm,'coverage':coverage(rows,arm)}
                rec['cohort_diagnostics']={'C1_segments':stats([r['evidence']['all_count'] for r in rows]),
                    'C1_independent_tracks':stats([r['evidence']['independent_count'] for r in rows]),
                    'C1_reliable_tracks':stats([r['evidence']['reliable_count'] for r in rows]),
                    'less_than_3_C1_segments':sum(r['evidence']['all_count']<3 for r in rows),
                    'less_than_3_independent_tracks':sum(r['evidence']['independent_count']<3 for r in rows),
                    'merged_pair_count':stats([len(r['pool']) for r in rows])}
                if ds=='external34':
                    rec['raw_errors']=measurement_errors(rows,arm);rec['full_score']=measurement_errors(rows,arm,True)
                    rec['median_reference_score']=measurement_errors(rows,arm,True,'median')
                if ds=='apo153':rec['boundary']=boundary_score(rows,refs,arm)
                if ds=='dev282':rec['clean276_coverage']=coverage([r for r in rows if r['image_id'] in clean_ids],arm)
                summary.append(rec)
            for aa,bb in cfg['contrasts']:
                va=[{**r,'arms':{'M':r['arms'][aa]}} for r in rows];vb=[{**r,'arms':{'M':r['arms'][bb]}} for r in rows]
                contrast={'seed':seed,'dataset':ds,'before':aa,'after':bb,
                          'measurements':{p:compare(va,vb,'M',p,ds=='external34') for p in TOL}}
                if ds=='apo153':contrast['boundary']=boundary_paired(va,vb,'M',refs)
                if ds=='external34':
                    ea=measurement_errors(rows,aa,True);eb=measurement_errors(rows,bb,True)
                    contrast['full_score_delta']=eb['full_cohort_score']-ea['full_cohort_score']
                    contrast['bootstrap']=bootstrap_score(ea,eb)
                    ac={x['image_id']:x for x in ea['per_image']};bc={x['image_id']:x for x in eb['per_image']}
                    contrast['score_changes']=sorted([{'image_id':iid,'before':ac[iid]['normalized_score'],'after':bc[iid]['normalized_score'],
                        'delta':bc[iid]['normalized_score']-ac[iid]['normalized_score']} for iid in ac],key=lambda x:x['delta'])
                contrasts.append(contrast)
            save_gzip(out/f'details/{seed}/{ds}.json.gz',rows)
            print('EVALUATED',seed,ds,flush=True)
    write(out/'summary.json',{'rows':summary,'reference_hashes':refhash,'boundary_reference_sha256':sha(REF),
         'official_score_only_proxy':True,'units_unconfirmed':True,'all_938_evaluated':True,'training':False,'default_changed':False})
    write(out/'contrasts.json',contrasts)
    write_report(summary,contrasts)
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='success.json'}
    write(out/'success.json',{'files':files,'fixed_n_per_seed':469,'seeds':cfg['seeds'],'arms':list(ARMS)})
    require(code_binding()==j(BASE/'binding.json')['protected_code'],'protected code unchanged')
    print('EVALUATION_COMPLETE',len(summary),'summary rows',flush=True)


def write_report(summary,contrasts):
    lines=['Round36 全量候选合并与方向对照（2026-11-03）','',
      '范围：冻结两种子T3与同一C1；每种子外部34+边界开发153+肌束开发282=469图，共938条图像-种子记录，9组=8442组输出。无训练、无新推理、未改默认流程、未上传Kaggle。',
      'D0/D1为旧基线；M合并已有边界对候选并去重，未跨来源合成新的上下组合；C=M+C1新评分；P=M+厚度假设；F=M+多片段方向；CP/CF/CPF为组合。',
      'M重新在联合支持区间计算原S包含率×质量分数，平局统一按质量、旧高度间距偏好及ID排序；因此包含候选整合所需的统一排序，不能称为逐分支原排序完全不变。',
      '厚度P是用户15-25mm假设的敏感性实验，5mm软过渡，扣分最高0.1。不是已验证的解剖先验；只在68条external34种子记录有尺度时可启用，其余870条禁用，不把像素当毫米。',
      '方向使用各区独立轨迹的长度截顶加权轴向中位数，至少3个一致轨迹才启用；保持原FL/PA锚点，仅几何失效时显式缺位。',
      '外部参考单位/协议尚未正式确认，以下是按毫米/度假设计算的Kaggle公式代理分数，不是新提交成绩，也不是独立盲测。',
      '全34评分：实验缺失值回退同图冻结D1，另报原始输出覆盖；PA/FL/MT的平均绝对误差分别除6/12/3后均值，另加官方微小平局项。',
      '边界153有143图目标层明确，另10图仍展示但不判对错。开发282没有逐图PA/FL/MT数值真值，不能计算全三项真实测量误差。','',
      '外部34图（MAE顺序PA度/FL毫米/MT毫米；越低越好）','种子 | 组 | 原始完整输出 | 全34代理分数 | MAE | FL偏差 | FL最大绝对误差 | 填补单元']
    for r in summary:
        if r['dataset']!='external34':continue
        q=r['full_score'];p=q['parameters'];lines.append(f"{r['seed']} | {r['arm']} | {r['coverage']['complete']}/34 | {q['full_cohort_score']:.6f} | {p['PA']['MAE']:.3f}/{p['FL']['MAE']:.3f}/{p['MT']['MAE']:.3f} | {p['FL']['bias']:+.3f} | {p['FL']['absolute']['max']:.3f} | {q['fallback_cells']}")
    lines+=['','边界开发与肌束开发（均为原始输出，不作参考填补）','种子 | 组 | 边界符合/143 | 边界输出/153 | 开发282 MT/PA/FL | 开发FL最大值px | PA异常图']
    for seed in j(CFG)['seeds']:
        for arm in ARMS:
            a=next(r for r in summary if r['seed']==seed and r['dataset']=='apo153' and r['arm']==arm)
            d=next(r for r in summary if r['seed']==seed and r['dataset']=='dev282' and r['arm']==arm);c=d['coverage']
            lines.append(f"{seed} | {arm} | {a['boundary']['correct_n']}/143 | {a['coverage']['boundary']}/153 | {c['parameters']} | {c['distribution']['FL']['max']} | {c['PA_invalid_images']}")
    lines+=['','单因素与组合：外部完整评分变化（后减前，负为改善）']
    for c in contrasts:
        if c['dataset']=='external34':lines.append(f"{c['seed']} {c['before']}→{c['after']}: {c['full_score_delta']:+.6f}; 分组bootstrap95%={c['bootstrap']['cluster_bootstrap_95']}")
    lines+=['','改善/退化实例：全部组合CPF对D1，按三项合成误差变化排序；完整逐图记录保存在contrasts.json。']
    for c in contrasts:
        if c['dataset']=='external34' and (c['before'],c['after'])==('D1','CPF'):
            for q in c['score_changes'][:3]+c['score_changes'][-3:]:lines.append(f"{c['seed']} {q['image_id']}: {q['before']:.4f}→{q['after']:.4f} ({q['delta']:+.4f})")
    lines+=['','解释限制：参考143仅定义目标边界，不用于预测选带；未匹配到参考可用候选不自动等同于网络完全没有响应，仍可能是候选提取/支持范围问题。',
      '多片段方向可能压掉真实局部弯曲，数量增加不等于准确率提高。MT假设没有肌种条件，只作敏感性分析，不可单凭本轮提升升为默认。',
      '外部34已被多轮用于观察，bootstrap不消除复用导致的偏差；两种子也不是两个独立人群。',
      '数据文件：evaluation/summary.json、contrasts.json、details/；预测先冻结在predictions/，binding.json含代码、缓存与参数哈希。']
    txt='\n'.join(lines)+'\n';(BASE/'RESULTS.txt').write_text(txt,encoding='utf8')
    (ROOT/'review/ROUND36_CANDIDATE_CONSENSUS_RESULTS.txt').write_text(txt,encoding='utf8')

if __name__=='__main__':main()
