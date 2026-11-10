"""Descriptive stability report; no automatic promotion or parameter tuning."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.round38_common import *
LABEL={'R':'冻结F0','G':'测量资格','N':'规范输入','V':'C1五视图','A':'逐轨迹稳健汇总','NV':'规范输入+C1五视图','NVA':'NV+稳健汇总','BNVA':'NVA+边界五视图'}
def f(v,n=2):return '—' if v is None else f'{v:.{n}f}'
def pct(v):return f(100*v) if v is not None else '—'

def main():
    sums=j(BASE/'evaluation/stability_summary.json');trans=j(BASE/'evaluation/transform_summary.json');acc=j(BASE/'evaluation/original_summary.json');natural=j(BASE/'evaluation/natural_summary.json')
    lines=['Round38：稳定性优先，结果','全部469原图、两份T3边界模型；C1模型共用。每图原图+六种轻微扰动；113相似组图片另加四种压力扰动。',
      '共3735图像变体×2种子×8组=59760组输出。上下边界和肌束均实际重新推理。未训练、未改默认流程、未提交Kaggle。',
      'R为Round37 F0：尺寸归一门槛；不等于历史Round36 M。T3沿用零提示输入；C1肌束像素参与后续选带。',
      '', '主验收：同图轻微扰动；每张取六个变体中最坏变化。MT<=2%，PA<=1°，FL<=5%且FL绝对差<=ROI宽3%。',
      '联合通过率固定以同种子R原图三项完整图片为分母；试验组原图缺测或任一扰动缺测都不通过。目标>=95%。',
      'P95仅描述观测到的数值变化，缺测不填0；必须结合通过率/缺测列阅读。两个种子不当作独立患者。','',
      'seed | 组 | 原图完整/469 | 三项联合通过/固定分母 | 通过% | MT最坏P95% | PA最坏P95° | FL最坏P95% | 任一缺测切换图 | 严重跳变图 | 全目标达成']
    for q in sums:
        p=q['parameters'];lines.append(f"{q['seed']} | {q['arm']} {LABEL[q['arm']]} | {q['original_complete_n']} | {q['joint_pass_count']}/{q['reference_complete_n']} | {pct(q['joint_pass_fraction'])} | {pct(p['MT']['worst_numeric_observed_only']['p95'])} | {f(p['PA']['worst_numeric_observed_only']['p95'])} | {pct(p['FL']['worst_numeric_observed_only']['p95'])} | {q['missing_flip_count']} | {q['severe_image_count']} | {q['joint_target_met']}")
    lines+=['','分数据集稳定性；不得用开发图的数量掩盖外部图失败','seed | 组 | 数据 | 原图完整 | 三项联合通过/固定分母 | 通过% | 缺测切换图']
    for q in j(BASE/'evaluation/stability_by_dataset.json'):lines.append(f"{q['seed']} | {q['arm']} | {q['dataset']} | {q['original_complete_n']} | {q['joint_pass_count']}/{q['reference_complete_n']} | {pct(q['joint_pass_fraction'])} | {q['missing_flip_count']}")
    lines+=['','分项通过率（各自采用R原图该项可测的固定分母）','seed | 组 | MT | PA | FL | MT/PA/FL在任一变体丢失图数']
    for q in sums:
        pp=q['parameters'];lines.append(f"{q['seed']} | {q['arm']} | "+' | '.join(f"{pp[p]['pass_count']}/{pp[p]['reference_n']} ({pct(pp[p]['pass_fraction'])}%)" for p in ('MT','PA','FL'))+' | '+' / '.join(str(pp[p]['lost_any']) for p in ('MT','PA','FL')))
    lines+=['','每种变换（非逐图最坏）','seed | 组 | 扰动 | n | MT相对P95% | PA绝对P95° | FL相对P95% | MT/PA/FL丢失数']
    for q in trans:
        pp=q['parameters'];lines.append(f"{q['seed']} | {q['arm']} | {q['transform']} | {q['n']} | {pct(pp['MT']['relative']['p95'])} | {f(pp['PA']['absolute']['p95'])} | {pct(pp['FL']['relative']['p95'])} | "+'/'.join(str(pp[p]['lost']) for p in ('MT','PA','FL')))
    lines+=['','真实相似图（不把实际差异强制压到0；12完全重复图对排除）','seed | 组 | 范围 | n | MT按组等权差/ROI宽% | PA按组等权差° | FL按组等权差/ROI宽% | 代表线几何匹配后切换数']
    for q in natural:lines.append(f"{q['seed']} | {q['arm']} | {q['mode']} | {q['n']} | {pct(q['group_equal']['MT'])} | {f(q['group_equal']['PA'])} | {pct(q['group_equal']['FL'])} | {q['representative_switches']}")
    lines+=['','原图准确性/覆盖作为代价监测，不据此挑稳定性赢家；外部参考单位协议仍未正式确认，分数是本地代理。','seed | 组 | 外部完整/34 | 本地分数 | MT/PA/FL MAE | 边界明确目标符合/143 | 开发完整/282 | 开发FL>2图宽']
    for seed in SEEDS:
        for arm in ARMS:
            e=next(q for q in acc if q['seed']==seed and q['arm']==arm and q['dataset']=='external34');b=next(q for q in acc if q['seed']==seed and q['arm']==arm and q['dataset']=='apo153');d=next(q for q in acc if q['seed']==seed and q['arm']==arm and q['dataset']=='dev282');pp=e['errors']['parameters']
            lines.append(f"{seed} | {arm} | {e['coverage']['complete']} | {f(e['errors']['full_cohort_score'],5)} | "+'/'.join(f(pp[p]['MAE']) for p in ('MT','PA','FL'))+f" | {b['boundary']['correct_n']} | {d['coverage']['complete']} | {d['coverage']['FL_over_2_image_width']}")
    lines+=['','解释范围：','- 所有结论只适用于此冻结数据与列明扰动，不能保证新设备/新患者。既有数据被多轮观察，不是新增盲测。',
      '- 五视图是同图独立预测证据，不跨病例平均。A/NVA/BNVA逐肌束求交后汇总，没有改成区域统一方向。',
      '- 仅对已有足够图像证据的轨迹正常测量；无法正常输出时的可计算探索估计另列，不计作完整覆盖。',
      '- 细线软概率平均可能改变连通性；候选过滤、区域缺位、边界支持仍含离散决策，可能保留不稳定。',
      '- 统一预测和测量采用变换后的图像；最终只映射既定候选/权重用于评估，没有用已知变换帮助选线。',
      '- 平移测试保持画布尺寸，边缘会有少量内容裁切；原始响应一致性只比较共同可见范围，并单列内容裁切比例。',
      '- 固定边界诊断仅覆盖113相似组图片；正式稳定性验收是全部469图、两模型都扰动的完整流程。',
      '- 代表线可以切换而数值稳定；反之选中相同轨迹也可能因方向或远处交点变化产生大FL跳变。']
    (BASE/'TABLES.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Choose only for descriptive investigation; preserve all arms and metrics.
    ranking=[]
    for arm in ARMS:
        rr=[q for q in sums if q['arm']==arm]
        ranking.append(dict(arm=arm,min_joint_pass=min(q['joint_pass_fraction'] for q in rr),max_missing=max(q['missing_flip_fraction'] for q in rr),both_seeds_pass=all(q['joint_target_met'] for q in rr)))
    ranking.sort(key=lambda q:(-q['min_joint_pass'],q['max_missing']))
    write(BASE/'evaluation/descriptive_ranking.json',ranking)
    print('REPORT TABLES DONE; descriptive ranking',ranking,flush=True)

if __name__=='__main__':main()
