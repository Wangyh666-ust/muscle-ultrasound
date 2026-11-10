"""Read-only scoring and ANON72 explanation on sealed R76 predictions."""
import ast,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scripts.round75_common import ROOT,OUT as SOURCE,j,readgz,write,sha
R76=ROOT/'runs/round76_adaptive_density_prior_v1'
OUT=ROOT/'review/round76_score_audit_20261110'
TOLS=dict(PA=6.,FL=12.,MT=3.)
COLS=dict(PA='pa_deg',FL='fl_mm',MT='mt_mm')

def official():
    nb=json.loads((ROOT/'umud-score.ipynb').read_text(encoding='utf8'))
    source=next(''.join(c['source']) for c in nb['cells'] if 'def score(' in ''.join(c.get('source',[])))
    tree=ast.parse(source);func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score')
    # Read organizer settings from the local score source. NumPy reproduces its
    # arithmetic without installing pandas or changing project dependencies.
    names=('target_cols','tolerances','weights','eval_unit','eps_secondary','eps_tertiary');settings={}
    for node in func.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:settings[node.targets[0].id]=ast.literal_eval(node.value)
    assert settings['eval_unit']=='image' and settings['tolerances']==dict(pa_deg=6.,fl_mm=12.,mt_mm=3.)
    def scorer(truth,prediction,idcol):
        by={r[idcol]:r for r in prediction};assert len(by)==len(truth)==len(prediction)
        primary=secondary=tertiary=0.;weight=sum(settings['weights'].values())
        for col in settings['target_cols']:
            e=np.asarray([by[r[idcol]][col]-r[col] for r in truth]);assert np.isfinite(e).all();w=settings['weights'][col];tau=settings['tolerances'][col]
            primary+=w*np.mean(abs(e))/tau;secondary+=w*np.median(abs(e))/tau;tertiary+=w*np.sqrt(np.mean(e**2))/tau
        return float((primary+settings['eps_secondary']*secondary+settings['eps_tertiary']*tertiary)/weight)
    return scorer

def calculate(rows,seed,arm,scorer):
    valid=[r for r in rows if all(r['outputs'][seed][arm]['values'][m] is not None for m in TOLS)]
    true=[];pred=[];metrics={}
    for r in valid:
        key=r['row']['dataset']+'|'+r['row']['image_id']
        true.append(dict(image_id=key,**{col:r['row']['reference'][m] for m,col in COLS.items()}))
        pred.append(dict(image_id=key,**{col:r['outputs'][seed][arm]['values'][m] for m,col in COLS.items()}))
    score=scorer(true,pred,'image_id') if valid else None
    for m in TOLS:
        e=np.array([r['outputs'][seed][arm]['values'][m]-r['row']['reference'][m] for r in valid])
        metrics[m]=dict(MAE=float(abs(e).mean()),score_contribution=float(abs(e).mean()/TOLS[m]/3))
    primary=sum(v['score_contribution'] for v in metrics.values())
    return dict(total=len(rows),complete=len(valid),missing_ids=[r['row']['image_id'] for r in rows if r not in valid],
        official_formula_score_on_complete_subset=float(score),primary=primary,tie_breaker=score-primary,metrics=metrics,
        full_requested_cohort_score_available=len(valid)==len(rows),no_missing_values_filled=True)

def figure72():
    sid='ANON_72';view='brightness105';seed='20261023'
    src=readgz(SOURCE/f'results/young78/{sid}/{view}.json.gz');r=readgz(R76/f'results/young78/{sid}/{view}.json.gz');t=readgz(R76/f'prior_results_v1/young78/{sid}/{view}.json.gz');row=src['row'];g=j(SOURCE/f'gallery/data/young78__{sid}__{view}.json')
    fig,axes=plt.subplots(1,2,figsize=(12,5.2));font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    for ax,arm,curve,out in [(axes[0],'B',src['baseline_curve'],r['outputs'][seed]['B']),(axes[1],'BT',t['baseline_curve'],t['outputs'][seed]['BT'])]:
        ax.imshow(Image.open(ROOT/row['image_path']).convert('L'),cmap='gray')
        for side,col in [('sup','#28cafa'),('deep','#fc6eb9')]:ax.plot(curve[side]['knots_x'],curve[side]['knots_y'],color=col,lw=2)
        for line in g['reference_lines'] or []:
            a=np.asarray(line);ax.plot(a[:,0],a[:,1],'w--',lw=1.1)
        for s in src['segments'][seed]:
            w=(out['FL_weights'] or {}).get(str(s['label_id']),0);ax.plot([s['p0'][0],s['p1'][0]],[s['p0'][1],s['p1'][1]],color='#42ee9e' if w else '#8a816c',lw=.7+4*w)
        ax.set_title(('原选择 B' if arm=='B' else '加强厚度先验 T 后')+f"\nMT {out['values']['MT']:.2f} mm；FL {out['values']['FL']:.2f} mm",fontproperties=font,fontsize=12)
        ax.axis('off')
    fig.suptitle('新增78图中的 ANON_72｜亮度×1.05｜种子20261023\n同一预测概率，只改候选评分；白虚线为人工参考（仅验收）',fontproperties=font,fontsize=13)
    fig.text(.5,.025,f"参考：MT {row['reference']['MT']:.2f} mm；PA {row['reference']['PA']:.2f}°；FL {row['reference']['FL']:.2f} mm",ha='center',fontproperties=font,fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.89));fig.savefig(OUT/'ANON_72_brightness105_B_vs_T.png',dpi=145);plt.close(fig)
    candidates=t['prior_audit']['candidates'];write(OUT/'ANON_72_evidence.json',dict(reference=row['reference'],original=r['outputs'][seed]['B']['values'],T=t['outputs'][seed]['BT']['values'],candidate_scores=candidates,
        same_probability_cache=True,view=view,seed=seed,image_id=sid,source_probability_metadata=sha(SOURCE/f'inference/young78/{sid}/{view}/boundary.json')))

def main():
    OUT.mkdir(parents=True,exist_ok=True);scorer=official()
    rows=[readgz(p) for p in sorted((R76/'results').glob('*/*/identity.json.gz'))];rows=[r for r in rows if r['row']['reference']];assert len(rows)==112
    report={}
    for ds in ('external34','young78','combined112'):
        rs=[r for r in rows if ds=='combined112' or r['row']['dataset']==ds];report[ds]={}
        for seed in ('20261023','20261024'):
            report[ds][seed]={arm:calculate(rs,seed,arm,scorer) for arm in ('B','BA','Q','QA')}
    two={}
    for seed in ('20261023','20261024'):
        rs=[r for r in rows if r['row']['dataset']=='young78' and r['outputs'][seed]['BA']['values']['FL'] is not None]
        errors={r['row']['image_id']:abs(r['outputs'][seed]['BA']['values']['FL']-r['row']['reference']['FL']) for r in rs}
        two[seed]=dict(error_118=errors['ANON_118'],error_152=errors['ANON_152'],fraction_young_FL_total=(errors['ANON_118']+errors['ANON_152'])/sum(errors.values()))
    write(OUT/'scores.json',dict(status='local_proxy_not_Kaggle_submission',unit_assumptions_inherited=True,eval_unit='image',source_notebook_sha=sha(ROOT/'umud-score.ipynb'),
        formula='mean(MAE_PA/6,MAE_FL/12,MAE_MT/3)+1e-6*mean(normalized MedAE)+1e-9*mean(normalized RMSE)',
        missing='no imputation; complete subsets only, reported denominators; no full112 score when any image incomplete',scores=report,large_error_fraction=two))
    figure72()
    text='''Round76评分与选层解释补充（2026-11-10）
从项目保存的umud-score.ipynb中score函数提取常量，用NumPy复算其公式，包含1e-6 MedAE与1e-9 RMSE平局项，未填补缺测，未上传Kaggle。
主B→BA同91张完整图：种子23 0.488810→0.479511；种子24 0.486285→0.477744。
辅助Q→QA同111张完整图：0.477354→0.470304；0.485240→0.478311。
样本数不同，不直接横向比较B和Q的优劣。完整112仍缺至少一项，不能声称已得到完整112或公榜评分。
仅原34且全34完整的Q→QA：0.444945→0.429040；0.448180→0.434689。B/BA只有25张三项齐全；其约0.38分不能冒充完整34表现。
新增78完整的Q/QA为77张，QA约0.488525/0.497572。原34好于新增来源，合并当前仍约0.47–0.48，不能判断公榜接近0.4。
公式的主项：1/3*(PA平均绝对误差/6 + FL平均绝对误差/12 + MT平均绝对误差/3)，越低越好。
QA种子23共同111张贡献：FL约0.2131、MT约0.1493、PA约0.1079。若其余不变，要从0.4703到0.4，FL MAE需再下降约2.53mm；本轮Q→QA只下降约0.254mm。

选层已经做过，不应再次笼统列作新任务：
D独立密度奖励在构造窄带保护控制失败，没有进入112正式排序，所以没有实现预期的原图118/152纠错。密度定义仍随面积缩小而变大，集中短片段可同时被错误窄带全部包住。
T较强10–31先验进入实际全量，但只改变4个扰动视图，原112图0个改变。118原图错误宽带仍靠质量/覆盖占优，罚分不足以反转；152错误宽带约28.3mm落在平坦先验内。只说“已经用了先验”并不等于修好选层。
ANON_72是新增78组，不是原34；需切到brightness105、种子20261023。原选择MT约16.73mm的候选得分0.852788；另一27.50mm候选原有0.011768罚分被10–31平坦区消除，得分升至0.853394，超过原选择，产生更宽的错误层。B FL83.99→T112.27，人工参考84.65。原始概率完全相同，是排序变化，不是网络识别突然改变。图像和数字存于ANON_72_evidence.json与B_vs_T图。
A只改实际FL贡献集合，未改PA和边界。近交点不是正确方向/正确目标层的保证。118/152两例仍占新增组FL绝对误差约三成，主问题尚在。
本轮受控集能验证评分机制行为，但构造候选/片段太理想化，T通过并不能证明真实多层目标识别可靠；后续需要先强化真实多层保护材料和同层/跨层判据，不能只继续调大先验系数或重复“加密度”。
本次只分析已封存输出，未改变Round76数据、模型、标签、默认流程；没有新增训练、推理或子代理。
'''
    (OUT/'REPORT.txt').write_text(text,encoding='utf8')
    checklist=ROOT/'thinking/things_for_considering.txt';marker='Round76评分复核【2026-11-10】'
    if marker not in checklist.read_text(encoding='utf8'):
        with checklist.open('a',encoding='utf8') as f:f.write('\n\n'+marker+'主BA完整91张代理分0.479511/0.477744；QA完整111张0.470304/0.478311，不能当完整112或公榜成绩。原34全覆盖QA为0.429040/0.434689。ANON72退化来自亮度105视图的候选排序，27.5mm层去罚后压过16.73mm候选；不是概率变化。D/T已经实验但未修好目标层：D停在控制，T原112选择0改变。需加强真实多层保护材料，避免仅凭理想化受控通过扩大。详review/round76_score_audit_20261110/REPORT.txt与scores.json。\n')
    print('R76_SCORE_AUDIT_DONE',[(s,report['combined112'][s]['BA']['official_formula_score_on_complete_subset'],report['combined112'][s]['QA']['official_formula_score_on_complete_subset']) for s in ('20261023','20261024')],two,flush=True)

if __name__=='__main__':main()
