"""Read-only follow-up: raw-response failures versus retained-geometry failures.

Expert boundaries are used only for scoring and explicit oracle diagnostics.
No model, decoder, candidate ranking, weights, or default output is changed.
"""
from copy import deepcopy
import json

from scripts.round80_recognition_triage import (
    ROOT, OUT, j, write, readgz, np, load_context, references,
)
from scripts.round78_fl_error_audit import expert_boundary
from scripts.round29_core import extended_fl, value_at


def main():
    protocol = j(ROOT / 'runs/round77_subject_layer_v1/protocol.json')
    rows = protocol['rows']
    records = [readgz(p) for p in sorted((OUT / 'records').glob('*.json.gz'))]
    assert len(records) == 221
    low = [r for r in records if any(v['flags']['response_below_low_over20pct'] for v in r['sides'].values())]
    low_young = {r['id'] for r in low if r['dataset'] == 'young78'}
    all_young = {r['image_id'] for r in rows if r['dataset'] == 'young78'}
    associations = {}
    for name, ids in [('raw_low', low_young), ('other_young', all_young-low_young), ('all_young', all_young)]:
        associations[name] = {}
        for seed in ('20261023', '20261024'):
            errors = []
            for sid in sorted(ids):
                r = readgz(ROOT / f'runs/round78_mt_geometry_adapter_v1/results/young78/{sid}/identity.json.gz')
                pred = r['outputs'][seed]['M']['values']['FL']; ref = r['row']['reference']['FL']
                if pred is not None:
                    errors.append(dict(id=sid, signed=pred-ref, absolute=abs(pred-ref)))
            associations[name][seed] = dict(images=len(ids), measured=len(errors),
                MAE=float(np.mean([e['absolute'] for e in errors])) if errors else None,
                total_abs_error=float(sum(e['absolute'] for e in errors)),
                over10=sum(e['absolute']>10 for e in errors), errors=errors)
    write(OUT / 'FL_associations.json', dict(low_response_images=[dict(dataset=r['dataset'], id=r['id'],
        sides=[s for s,v in r['sides'].items() if v['flags']['response_below_low_over20pct']]) for r in low],
        associations=associations, caveat='Observed association, NOT a causal share of FL error. Includes historical and assisted references, no independent test.'))

    row = next(r for r in rows if r['image_id']=='ANON_144' and r['dataset']=='young78')
    src = readgz(ROOT/'runs/round77_subject_layer_v1/results/young78/ANON_144/identity.json.gz')
    current = readgz(ROOT/'runs/round78_mt_geometry_adapter_v1/results/young78/ANON_144/identity.json.gz')
    geo = j(ROOT/'validation/young_expert_v03/geometry/ANON_144.json')
    ecv = expert_boundary(geo, row['annotation_offset'])
    oracle = dict(reference=row['reference'], seeds={}, warning='Diagnostic oracle, not deployable. Same C1 tracks, same actual retained weights; no missing-weight renormalization. Expert endpoints use explicit assisted completion outside source strokes.')
    for seed in ('20261023','20261024'):
        segments = {str(s['label_id']):s for s in src['segments'][seed]}
        out = current['outputs'][seed]['M']; active = [(str(sid),float(w)) for sid,w in out['FL_weights'].items() if w>0]
        assert abs(sum(w for _,w in active)-1)<1e-8
        oracle['seeds'][seed]={}
        for name, replacements in [('B',()),('upper_only',('sup',)),('lower_only',('deep',)),('both',('sup','deep'))]:
            cv=deepcopy(src['curves']['P'])
            for side in replacements:cv[side]=ecv[side]
            tracks=[]
            for sid,w in active:
                fl=extended_fl(segments[sid],cv,row['width'],row['height']); px=fl.get('fl_px_raw')
                tracks.append(dict(id=sid,weight=w,FL=None if px is None else px/row['px_per_mm'],
                    failure=None if px is not None else fl.get('fl_failure_conditions')))
            total=sum(t['weight']*t['FL'] for t in tracks) if all(t['FL'] is not None for t in tracks) else None
            oracle['seeds'][seed][name]=dict(FL=total,tracks=tracks,
                absolute_error=None if total is None else abs(total-row['reference']['FL']))
        assert abs(oracle['seeds'][seed]['B']['FL']-out['values']['FL'])<1e-7
        oracle['seeds'][seed]['retained_values']=out['values']
    write(OUT/'ANON_144_boundary_oracle.json',oracle)

    # Render an aligned figure in original input coordinates, using fixed 0..1 scale.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    c=load_context(row,'identity');gt,_=references(row)
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
    axes=axes.ravel();axes[0].imshow(Image.open(ROOT/row['image_path']).convert('L'),cmap='gray',vmin=0,vmax=255)
    axes[0].set_title('Original + source-supported expert traces + current formal boundaries')
    colors={'upper':'#00d9ff','lower':'#ff64b4'}
    for side,role in [('upper','sup'),('lower','deep')]:
        cv=src['curves']['P'][role];xx=np.linspace(240,744,600)
        axes[0].plot(xx,[value_at(cv,float(x)) for x in xx],color=colors[side],lw=1.8,label='Current '+side)
    a,b,_,_=c['frame']['input_box'];h,w=c['prob'].shape[-2:]
    for ax,head,title in zip(axes[1:],(0,1,2),('Union probability','Upper-role probability','Lower-role probability')):
        m=ax.imshow(c['prob'][head],extent=(a-.5,a+w-.5,b+h-.5,b-.5),cmap='magma',vmin=0,vmax=1)
        ax.set_title(title+' (fixed scale 0..1)')
    for k,ax in enumerate(axes):
        for side in ('upper','lower'):
            p=gt[side];ax.plot(p[:,0],p[:,1],'--',color='white',lw=1.6,label='Expert source '+side if k==0 else None)
        ax.set_xlim(230,754);ax.set_ylim(455,105);ax.set_xlabel('Original image x');ax.set_ylabel('Original image y')
    axes[0].legend(fontsize=8,loc='lower left')
    fig.colorbar(m,ax=axes[1:],fraction=.025,pad=.02)
    fig.suptitle('ANON_144 | True upper boundary is weak in BOTH union and upper-role outputs\nWhite dashed: expert source strokes only; no assisted extension shown',fontsize=14)
    fig.savefig(OUT/'ANON_144_probability_alignment.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(low_response_images=len(low),low_young=len(low_young),low_apo=sum(r['dataset']=='apo153' for r in low),
        ids=[(r['dataset'],r['id']) for r in low],associations={k:{s:{x:v[x] for x in ('images','measured','MAE','over10')} for s,v in d.items()} for k,d in associations.items()},
        oracle={s:{k:v.get('FL') for k,v in d.items() if k!='retained_values'} for s,d in oracle['seeds'].items()}),ensure_ascii=False))
    report(records, oracle, associations, low)


def report(records, oracle, associations, low):
    from scripts.round80_common import verify_default, sha
    verify_default()
    q=next(r for r in records if r['id']=='ANON_144')['sides']['upper']
    profile=q['physical_radius_profiles']['1.0']
    s23=oracle['seeds']['20261023'];s24=oracle['seeds']['20261024']
    text=f'''边界责任重新核查 | 2026-11-11 | 只读诊断

结论修正
不能把当前失败统一归到后处理。ANON_144按现有专家目标层参考，正确上边界在解码前概率图中响应极弱，候选池也没有正确上边界；属于现有模型/输入推理链的识别缺失。ANON_68同时有局部弱响应与正确右侧候选未利用；ANON_78有已确认的正式PA/FL稀疏重拟合尖折。三类必须分开处理。Round80未改动默认，不能把“查出后处理缺陷”误说成“所有失败可靠后处理修好”。

范围与绑定
在547张原图里，有221张具备可用于逐位置核查的上下边界参考：143张历史人工边界开发图、78张新增专家描线图。其余326张缺少这类参考，没有归因为正确/错误。78张仅以专家原始描线在组织ROI内的源支持部分评分，不把人工辅助延长算成网络真值；143张使用既有冻结人工参考。221个缓存文件SHA和输入框绑定检查通过。所有参考只在解码之后用于诊断，不参与候选生成或排序。
当前保留的输出是R78M/R79B；边界固定C62 C seed23。两C1种子不是两个边界模型。当前缓存概率是解码前的五平移对齐平均，不是逐次前向输出。

ANON_144：确认的证据
1. 专家上边界源支持361列，原图x240..600，y约213..214；全部在输入和测量证据区域内，并未被当前裁剪直接切掉。
2. 在专家上边界上下1mm邻域取最大响应，再跨列统计，上边界头中位数{profile['role']['median']:.6f}、最大{profile['role']['maximum']:.6f}；联合头中位数{profile['union']['median']:.8f}、最大{profile['union']['maximum']:.6f}。两头没有任何列达到现有低响应门槛0.18。0.25/0.5/1mm邻域检查一致，不是仅差一点跨过0.5阈值。
3. 正确上边界在弱候选、保留候选、选中路径、合格路径和正式曲线中的覆盖均为0。上边界池仅1条，位于更浅层。正确下边界有明显响应，不能把整张说成什么都没学到。
4. 当前MT/PA/FL：seed23为{s23['retained_values']['MT']:.4f}mm / {s23['retained_values']['PA']:.4f}度 / {s23['B']['FL']:.4f}mm；专家参考MT{oracle['reference']['MT']:.4f}mm / PA{oracle['reference']['PA']:.4f}度 / FL{oracle['reference']['FL']:.4f}mm。
5. 固定C1片段、实际参与FL的集合和每条权重，仅替换上边界为专家参考：FL seed23 {s23['B']['FL']:.4f} -> {s23['upper_only']['FL']:.4f}，seed24 {s24['B']['FL']:.4f} -> {s24['upper_only']['FL']:.4f}mm。仅替换下边界则分别{s23['lower_only']['FL']:.4f}/{s24['lower_only']['FL']:.4f}mm；上下均替换{s23['both']['FL']:.4f}/{s24['both']['FL']:.4f}mm。上边界错误是本例FL大误差的主要来源。
6. 这是使用已知答案的归因诊断，专家边界端部含显式辅助补全，不是可部署方法或独立验证收益。没有重分配缺失权重。
7. 正确带厚度约17.7mm、当前错误带约24.7mm，都处于10–31mm内；单纯加强该区间先验不能区分它们。当前候选连接/重排也不能从不存在的正确候选恢复真实边界。

221张上下边界筛查：能说什么，不能说什么
以专家线附近2个512网络格距为定位邻域，使用现有融合响应role*(0.4+0.6*union)。若一侧超过20%的源支持列低于现有0.18门槛，记为“较长区间响应不足”的诊断提醒；20%是描述筛查界限，不是已验证的训练/解码新参数。
共{len(low)}张触发：新增78图9张、人工开发143图3张；上边界8侧，下边界5侧，其中1图双侧。名单见FL_associations.json。这不等于12张都完全漏检，也不等于其余209张模型识别全部正确；宽概率带峰位置偏移、目标层混淆、局部漏段也可能不触发此界限。
其余有局部高响应但无贴合候选、候选未选、合格路径又被正式重拟合偏移等现象，逐图连续覆盖率和各阶段记录保留在records/与summary.json。高响应落在参考邻域内不代表概率峰定位正确，不直接判成解码bug。各标志可重叠，不把侧数相加当病例数，不把可见列数当独立受试者或FL误差的因果份额。
新增78中响应不足9张里8张有FL，MAE seed23/24为{associations['raw_low']['20261023']['MAE']:.3f}/{associations['raw_low']['20261024']['MAE']:.3f}mm；其余69张为{associations['other_young']['20261023']['MAE']:.3f}/{associations['other_young']['20261024']['MAE']:.3f}mm。前者被ANON_152极端错误显著拉高：去掉152后7张为8.226/8.516mm。仅为关联，不能据此声称训练问题贡献了多少总误差；118等弱响应病例的FL也可以接近参考。

当前不能进一步确定的原因
已确认的是“当前推理输出在目标边界处没有足够证据”，尚不能单凭概率图判定：训练标注的目标层/中线口径不一致、训练数据域覆盖不足、512缩放后的细节/组织尺度影响、单次输出被五视图平均削弱，哪一个是主因。没有潜空间分析证据，不能声称潜空间表示坏了，也不能声称单纯多训练就能解决。
代码事实：当前boundary推理经validation112_infer.py调用round41_native.consensus/predict，长边512，并拼接全零第4通道；C1用于后续候选选带，没有作为非零提示输入本次边界网络。此事实不证明它造成144失败。既往提示训练已有对照，不能忽略旧结果而直接重复训练。

重新排列下一步
第一条路线：对144等目标处低响应，先核同口径专家目标层与训练标签、逐次前向/平均前后概率、实际512输入中的目标保留情况。只有确定改善来自模型/训练后再设计重训。对缺少局部证据的地方，不凭加大连接阈值或亮度脑补宣称识别恢复。
第二条路线：对68等已有正确候选未利用，检查连接约束与同层排序；对78等恢复路径正确但正式拟合变形，修几何接口保真。用原有正常边界和多层负例保护，独立验收，不与识别缺失混报。
最后以固定C1下的实际FL贡献及整个测量链误差/覆盖/跳变决定优先投入，不能仅按可见边界覆盖或一张图的观感裁决。不训练、不新增推理、不改默认流程，未运行新优化轮次。

产物
ANON_144_probability_alignment.png：原图与三头概率同坐标同0–1色标；白虚线只画专家源描线部分。
ANON_144_boundary_oracle.json：固定片段和权重的上/下/双边界替换诊断。
FL_associations.json：响应不足名单与FL关联，非因果份额。
summary.json / records/：221图两侧逐阶段概率与几何统计。
脚本scripts/round80_recognition_triage.py；scripts/boundary_responsibility_review_20261111.py。
'''
    (OUT/'REPORT.txt').write_text(text,encoding='utf-8')
    (ROOT/'thinking/BOUNDARY_RESPONSIBILITY_REVIEW_20261111.txt').write_text(text,encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>边界责任重新核查</title><style>body{font:17px/1.7 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;background:#0e1925;color:#e6eff7}img{width:100%;background:white}a{color:#62caff}pre{white-space:pre-wrap;font:inherit}table{border-collapse:collapse;width:100%}td,th{border:1px solid #567;padding:10px;text-align:left}</style><h1>边界问题不能统一归为后处理</h1><p>ANON_144正确上边界在解码前没有足够响应。固定肌束和权重，只换专家上边界，FL 95.61 → 67.23 mm（参考68.43）。图中白虚线是专家源描线，蓝/粉线是当前正式测量曲线。</p><img src="ANON_144_probability_alignment.png" alt="144原图与联合、上、下概率在同一坐标系对齐"><table><tr><th>案例</th><th>已确认问题</th><th>处理范围</th></tr><tr><td>ANON_144</td><td>正确上边界概率极弱，正确候选不存在</td><td>先查模型目标、输入与原始输出；纯候选重排无解</td></tr><tr><td>ANON_68</td><td>局部响应不足，右侧已有正确候选未利用</td><td>识别与连接/排序分开</td></tr><tr><td>ANON_78</td><td>恢复路径与正式FL重拟合曲线不一致</td><td>几何接口保真；MT接口已修复不代表FL已修复</td></tr></table><p>221张有边界参考，12张触发较长区间弱响应筛查。这个计数不是全部识别失败数；低响应原因和最终FL贡献仍须分别判断。</p><p><a href="REPORT.txt">完整报告</a> · <a href="ANON_144_boundary_oracle.json">144替换诊断</a> · <a href="FL_associations.json">全量筛查名单</a> · <a href="summary.json">阶段统计</a></p></html>'''
    (OUT/'index.html').write_text(html,encoding='utf-8')
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
    write(OUT/'delivery_manifest.json',dict(read_only=True,training=False,new_inference=False,default_hashes_verified=True,
        scope=dict(images=221,sides=442,low_response_images=12),files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        script_sha={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts/round80_recognition_triage.py',ROOT/'scripts/boundary_responsibility_review_20261111.py')}))


if __name__=='__main__':main()
