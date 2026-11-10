"""Paired full-regression report; rejected versions and missing values remain visible."""
from scripts.round80_common import *
from copy import deepcopy
from collections import Counter,defaultdict
import json
import scripts.round75_report as metrics
from scripts.round76_score_audit import official,calculate
from scripts.audit_boundary_connection_20261110 import references

REPORT_ARMS=('B','R0','R')


def guarded(records):
    output=[];checks=0
    for old in records:
        r=deepcopy(old);r['outputs']={s:dict(B=deepcopy(old['outputs'][s]['B']),R0=deepcopy(old['outputs'][s]['R'])) for s in SEEDS}
        unknown=old['audits']['R']['ranking']['reason'].startswith('C1_unknown')
        for seed in SEEDS:
            r['outputs'][seed]['R']=deepcopy(old['outputs'][seed]['B'] if unknown else old['outputs'][seed]['R'])
            if unknown:assert r['outputs'][seed]['R']==r['outputs'][seed]['B']
            checks+=1
        r['pairs']=dict(B=old['pairs']['B'],R0=old['pairs']['R'],R=old['pairs']['B'] if unknown else old['pairs']['R'])
        r['curves']=dict(B=old['curves']['B'],R0=old['curves']['R'],R=old['curves']['B'] if unknown else old['curves']['R'])
        r['guard_audit']=dict(unknown=unknown,preserve_original_selection=unknown,informed_by_initial_full=True,not_a_blind_confirmation=True)
        write(OUT/f"guarded_results/{r['row']['dataset']}/{r['row']['image_id']}/{r['view']}.json.gz",r);output.append(r)
    write(OUT/'guard_full_checks.json',dict(views=len(records),seed_cases=checks,
       guarded_outputs_equal_B=sum(r['outputs'][s]['R']==r['outputs'][s]['B'] for r in output for s in SEEDS),
       all_guarded_geometry_equal_B=all(r['pairs']['R']==r['pairs']['B'] and r['curves']['R']==r['curves']['B'] for r in output),
       original_R0_retained=True,no_new_numeric_parameters=True,informed_by_initial_full=True))
    return output


def comparison(rows,seed,arm,metric):
    before=[r for r in rows if r['outputs'][seed]['B']['values'][metric] is not None]
    common=[r for r in before if r['outputs'][seed][arm]['values'][metric] is not None]
    lost=[r['row']['image_id'] for r in before if r['outputs'][seed][arm]['values'][metric] is None]
    recovered=[r for r in rows if r['outputs'][seed]['B']['values'][metric] is None and r['outputs'][seed][arm]['values'][metric] is not None]
    mae={a:float(np.mean([abs(r['outputs'][seed][a]['values'][metric]-r['row']['reference'][metric]) for r in common])) if common else None for a in ('B',arm)}
    return dict(common=len(common),common_MAE=mae,lost=lost,recovered=[r['row']['image_id'] for r in recovered],
                recovered_MAE=float(np.mean([abs(r['outputs'][seed][arm]['values'][metric]-r['row']['reference'][metric]) for r in recovered])) if recovered else None,
                over10=sum(r['outputs'][seed][arm]['values'][metric]-r['row']['reference'][metric]>10 for r in common),under10=sum(r['outputs'][seed][arm]['values'][metric]-r['row']['reference'][metric]<-10 for r in common))


def boundaries(identities):
    rows=[]
    for r in identities:
        ref,kind=references(r['row'])
        if ref is None:continue
        box=r['frame']['input_box'];step=max(box[2]-box[0],box[3]-box[1])/512/r['matrix'][0][0];tau=2*step;record=dict(id=r['row']['image_id'],dataset=r['row']['dataset'],reference_class=kind,sides={})
        for side,role in [('upper','sup'),('lower','deep')]:
            gt=ref[side];x,y=gt.T;result={}
            for arm in REPORT_ARMS:
                pair=r['pairs'][arm];cv=r['curves'][arm]
                if pair is None:result[arm]=dict(complete_correct_coverage=0.,finite_fit_correct_coverage=0.,mae=None);continue
                p=pair[side];inside=(x>=p['x'][0])&(x<=p['x'][-1]);error=abs(np.interp(x,p['x'],p['y'])-y)
                finside=np.zeros(len(x),bool) if cv is None else (x>=cv[role]['support'][0])&(x<=cv[role]['support'][1]);fe=np.full(len(x),np.inf) if cv is None else abs(np.interp(x,cv[role]['knots_x'],cv[role]['knots_y'])-y)
                result[arm]=dict(complete_correct_coverage=float(np.mean(inside&(error<=tau))),finite_fit_correct_coverage=float(np.mean(finside&(fe<=tau))),mae=float(error[inside].mean()) if inside.any() else None)
            record['sides'][side]=result
        rows.append(record)
    assert len(rows)==221
    summary={}
    for side in ('upper','lower'):
        good=[r for r in rows if r['sides'][side]['B']['complete_correct_coverage']>=.9]
        summary[side]={arm:dict(reference_images=len(rows),mean_complete_correct_coverage=float(np.mean([r['sides'][side][arm]['complete_correct_coverage'] for r in rows])),
            baseline_good_sides=len(good),good_sides_degraded_over5points=sum(r['sides'][side][arm]['complete_correct_coverage']<r['sides'][side]['B']['complete_correct_coverage']-.05 for r in good)) for arm in REPORT_ARMS}
    write(OUT/'boundary_references.json',dict(summary=summary,rows=rows,source_supported_only=True))
    return summary


def main():
    verify_default();assert j(OUT/'evaluation_done.json')['views']==995
    old=[readgz(p) for p in sorted((OUT/'results').rglob('*.json.gz'))];assert len(old)==995
    rows=guarded(old);ident=[r for r in rows if r['view']=='identity'];refs=[r for r in ident if r['row']['reference']]
    metrics.ARMS=REPORT_ARMS;metrics.VIEWS=('identity','translate2_-1','brightness105','scale1015','combo098_translate-1_1')
    summary={};scores={};scorer=official()
    for ds in ('external34','young78','combined112'):
        rr=[r for r in refs if ds=='combined112' or r['row']['dataset']==ds];summary[ds]={};scores[ds]={}
        for seed in SEEDS:
            summary[ds][seed]=dict(metrics={a:{m:metrics.metric(rr,seed,a,m) for m in ('MT','PA','FL')} for a in REPORT_ARMS},
                                   comparisons={a:{m:comparison(rr,seed,a,m) for m in ('MT','PA','FL')} for a in REPORT_ARMS if a!='B'})
            common=[r for r in rr if all(r['outputs'][seed][a]['values'][m] is not None for a in REPORT_ARMS for m in ('MT','PA','FL'))]
            scores[ds][seed]=dict(fixed_common_complete={a:calculate(common,seed,a,scorer) for a in REPORT_ARMS},
                available_subsets={a:calculate(rr,seed,a,scorer) for a in REPORT_ARMS},caution='Complete-subset changes are not paired gains; historical proxy, not Kaggle public score.')
    write(OUT/'summary.json',summary);write(OUT/'kaggle_proxy_scores.json',scores)
    bykey={(r['row']['dataset'],r['row']['image_id'],r['view']):r for r in rows}
    stability=metrics.stability(ident,bykey);natural=metrics.natural(ident);write(OUT/'stability_all4.json',stability);write(OUT/'natural_pairs.json',natural);boundary=boundaries(ident)
    counts=Counter();changes=[];skips=[]
    for r in rows:
        au=r['audits']['R'];counts[au['ranking']['reason']]+=1
        if au.get('not_applied'):skips.append(dict(dataset=r['row']['dataset'],id=r['row']['image_id'],view=r['view'],reason=au['not_applied']))
        if any(r['outputs'][s]['R0']['values']!=r['outputs'][s]['B']['values'] for s in SEEDS):changes.append(dict(dataset=r['row']['dataset'],id=r['row']['image_id'],view=r['view'],B={s:r['outputs'][s]['B']['values'] for s in SEEDS},R0={s:r['outputs'][s]['R0']['values'] for s in SEEDS},reference=r['row']['reference']))
    write(OUT/'R_full_audit.json',dict(reason_counts=dict(counts),changed_views=changes,baseline_rebuild_not_applied=skips))
    requested=['ANON_68','ANON_78','ANON_118','ANON_152','image_1001.tif','image_1004.tif','image_0960.tif']
    cases=[r for r in ident if r['row']['image_id'] in requested]
    write(OUT/'requested_cases.json.gz',cases)
    cal=j(OUT/'controls/calibration.json')['summary'];confirm=j(OUT/'controls/confirmation.json')['summary'];guard=j(OUT/'controls/R_guard_recheck.json')
    s0=summary['combined112']['20261023'];s1=summary['combined112']['20261024'];checks=j(OUT/'guard_full_checks.json')
    reference78=next(r for r in ident if r['row']['image_id']=='ANON_78')['row']['reference']['FL']
    text=f'''Round80：连接/排序/FL几何分离实验（2026-11-10）

结论
本轮没有获得可升级的全量测量收益。C接续和G几何迁移未通过控制，没有进入全量组合；R共同窗口排序通过构造控制，但真实全量初版使118缺测，不能推广。
R初版在窗口未知时仍去掉原奖励并重新选层，与方案“保留歧义”的本意不一致。已增加无新参数的保留原选择回退，明确属于看过初版结果后的安全修订；原R0全量结果保留，不能把修订当新盲测。
修订R在995视图×两种子输出与当前B逐项相同，消除了新退化，也没有提升；不把回退原结果宣传为改善。

数据与控制
672份复核训练来源文件及673标签变体SHA匹配；48校准、624确认。只构造几何/概率失败场景，没有训练和新推理；上下边界、镜像都检验。
C四组稳定窗口/概率支撑组合分别有11/9/6/5个误连负例；全部正例平均覆盖增益为0。正例大多已被旧连接处理到位（控制天花板），不能由0增益推断所有真实缺口都无法修；但错误跨层接合已经违反保护门槛，因此停止扩大。
R校准48个奖励陷阱恢复48，确认624恢复624，未动正常控制；这些是构造评分控制，不是模型泛化或真实多层识别成功率。
G0有限采样点几何误差变0，但新增60次交点缺测，未通过。进一步发现旧半像素节点没有按合格邻点插值，整数点残差检查漏掉了针状尖点。修正并加入分数节点/中点检查后，几何仍精确、缺测降至51，仍不通过；没有扩大到624或全量G/组合。
原G0另做射线接缝连续性试算可补回41/60，仍19未解，属于单独反事实，不能和分数节点修正收益相加，也不能称已可靠迁移。保留旧射线截距与替换有限曲线之间的连续性、非合格片段接缝和多交点资格需要继续分开解决。
因此未完成原设想的八组全量组合；按预先门槛只完成B/R两个初始组的全量，再补R安全修订的确定性全量复核。没有隐藏失败组或换成宽松门槛。

全量
547原图＋112图四扰动=995视图，两C1种子。边界选择固定B23；不把两C1种子当两份独立边界证据。
R0只改变118原图与亮度105两个视图。原图MT19.4595→12.9543mm（参考17.9394），PA/FL由有值变缺测；亮度视图同样缺测。68、78均未解决。
111张原有FL中丢失118，剩余共同110张FL逐项相同。不能用111→110分母变化后的MAE宣称精度改善或独立数值退化；明确损失是一张测量覆盖及MT选层变差。
两条旧历史记录1009/1011的基线几何/资格重建不一致，初版R没有强行套入，保留封存值并单列；不是两次成功恢复。

G1命名病例诊断（没有按病例再调参，也不属于全量通过组）
78的旧尖折在分数节点修正后消除，固定原权重FL96.0329→99.9045mm，参考{reference78:.4f}，单例更接近；不能由此推广。
68固定原权重后反而FL缺测：种子23的一条片段权重仅2.388%，上界出现两个交点，约x319.025和348.466；种子24对应片段权重3.146%。其余片段仍可求交，但“所有原贡献均须唯一且不重分配”的本轮控制使整张缺测，未偷偷删线或归一化。
这提示边界端部小异常、交点关联与单束失败传播必须一起审计。不能把“有限曲线更忠实”直接等同为“所有肌束求交更可靠”。68原右侧片段仍未接入，两个问题并存。
1001/1004属于48校准训练源正常控制：C正常场景改动0、R正常选择改动0、G1正常新增缺测0。它们不在547推理队列中，没有混报成额外验证样本。

原图当前B与安全修订R
112张MT覆盖112，MAE{float(s0['metrics']['B']['MT']['mae']):.6f}mm；PA/FL覆盖111。
两种子FL MAE{s0['metrics']['B']['FL']['mae']:.6f}/{s1['metrics']['B']['FL']['mae']:.6f}mm，>10mm仍20/17张。
四扰动FL通过{stability['combined112']['20261023']['R']['FL']['pass_count']}/{stability['combined112']['20261024']['R']['FL']['pass_count']}，严重{stability['combined112']['20261023']['R']['FL']['severe_count']}/{stability['combined112']['20261024']['R']['FL']['severe_count']}，均与B相同。MT/PA、117真实相似对、18严格对、221源边界上下覆盖也相同。
守护修订的{checks['seed_cases']}个输出完全保留B；这说明未引入新退化，不是稳定性提高。
本地当前111完整子集代理分保持0.431245/0.425119；没有新Kaggle提交，不能把该数当公榜0.50110的替代。

下一步优先级
优先修FL几何接口中的连续性与交点资格：先消除旧分数节点残留，再保证有限曲线与端部射线接缝位置一致；非合格缺口不能拼成针状跳变，已有qualified也不能替代异常端点审计。固定肌束，单独验证端部稳健拟合、多根与原轨迹的关联以及单束失败时的显式回退；不能未经验收就丢掉小权重线或强取一个交点。
其次补足真正复现“旧机制拒接”的正例控制，检验稳定段/异常端点重建，继续保护错层负例。不能因变长或更平顺就接受连接。
共同窗口机制只保留“未知不强选”的安全约定；没有数据显示继续调此项奖励能解决68，停止盲搜权重。
68仍需同层片段和错脊识别，78仍需FL接口迁移；现有数据不支持优先重训C1，也不支持认定潜空间失败。

限制
221边界参考中78为专家源描线，辅助外延区另列；34单位沿用历史假定。所有数据已多轮使用，均为历史回归，不是新的独立盲测。
控制构造的高通过率未转化成真实选层收益，必须如实保留这项反证。G图形更好不能代替FL交点/缺测验收。
未训练、未补推理、未改默认、无子代理。旧代码受保护哈希全部不变。清单和图册已更新。
'''
    (ROOT/'thinking/ROUND80_CONNECTION_FIDELITY_RESULTS_20261110.txt').write_text(text,encoding='utf8');(OUT/'REPORT.txt').write_text(text,encoding='utf8')
    write(OUT/'report_ready.json',dict(originals=547,views=995,original_full_arms=['B','R0'],guarded='R',eight_groups_not_expanded_due_control_failures=True,training=False,inference=False,default_changed=False,
       admitted_for_promotion=[],guard_identical_to_baseline=True,not_blind=True))
    print('R80_REPORT_READY',dict(guarded_equal=checks['guarded_outputs_equal_B'],initial_R_changed=len(changes),baseline_rebuild_skipped=len(skips),FL_MAE=[s0['metrics']['B']['FL']['mae'],s1['metrics']['B']['FL']['mae']]))


if __name__=='__main__':main()
