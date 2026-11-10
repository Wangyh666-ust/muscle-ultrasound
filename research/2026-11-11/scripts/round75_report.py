"""All-denominator regression report, real-pair checks and lazy-loaded comparison gallery."""
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import csv,json,gzip,math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.round75_common import *
SEED_NAMES=('20261023','20261024');METRICS=('MT','PA','FL');ARMS=('B','Q')

def metric(rows,seed,arm,m):
    rr=[r for r in rows if r['outputs'][seed][arm]['values'][m] is not None]
    e=np.array([r['outputs'][seed][arm]['values'][m]-r['row']['reference'][m] for r in rr]);ae=abs(e)
    return dict(total=len(rows),available=len(rr),missing=len(rows)-len(rr),mae=float(ae.mean()) if len(ae) else None,
       median=float(np.median(ae)) if len(ae) else None,p90=float(np.quantile(ae,.9)) if len(ae) else None,
       bias=float(e.mean()) if len(e) else None,max_error=float(ae.max()) if len(ae) else None,
       over50pct=sum(abs(r['outputs'][seed][arm]['values'][m]/r['row']['reference'][m]-1)>.5 for r in rr))

def compare(rows,seed,m):
    a=[r for r in rows if all(r['outputs'][seed][arm]['values'][m] is not None for arm in ARMS)]
    recovered=[r for r in rows if r['outputs'][seed]['B']['values'][m] is None and r['outputs'][seed]['Q']['values'][m] is not None]
    lost=[r for r in rows if r['outputs'][seed]['B']['values'][m] is not None and r['outputs'][seed]['Q']['values'][m] is None]
    return dict(common=len(a),common_mae={arm:float(np.mean([abs(r['outputs'][seed][arm]['values'][m]-r['row']['reference'][m]) for r in a])) if a else None for arm in ARMS},
       recovered=len(recovered),recovered_mae=float(np.mean([abs(r['outputs'][seed]['Q']['values'][m]-r['row']['reference'][m]) for r in recovered])) if recovered else None,
       recovered_max_error=max([abs(r['outputs'][seed]['Q']['values'][m]-r['row']['reference'][m]) for r in recovered],default=None),
       lost=len(lost),better=sum(abs(r['outputs'][seed]['Q']['values'][m]-r['row']['reference'][m])<abs(r['outputs'][seed]['B']['values'][m]-r['row']['reference'][m])-1e-8 for r in a),
       worse=sum(abs(r['outputs'][seed]['Q']['values'][m]-r['row']['reference'][m])>abs(r['outputs'][seed]['B']['values'][m]-r['row']['reference'][m])+1e-8 for r in a),
       recovered_ids=[r['row']['image_id'] for r in recovered])

def stability(identities,bykey):
    targets=j(ROOT/'configs/round38_stability_first.json')['targets'];out={}
    for ds in ('external34','young78','combined112'):
        rows=[r for r in identities if r['row']['dataset'] in ('external34','young78') and (ds=='combined112' or r['row']['dataset']==ds)]
        out[ds]={}
        for seed in SEED_NAMES:
            out[ds][seed]={}
            for arm in ARMS:
                metrics={}
                for m in METRICS:
                    worst=[];missing=0;flips=0;passed=0;severe=0;comparisons=0;deltas=[];cases=[]
                    for r in rows:
                        key=(r['row']['dataset'],r['row']['image_id']);base=r['outputs'][seed][arm]['values'][m]
                        views=[bykey[(*key,v)]['outputs'][seed][arm]['values'][m] for v in VIEWS[1:]]
                        absent=base is None or any(v is None for v in views);missing+=absent
                        flips+=any((base is None)!=(v is None) for v in views)
                        dd=[abs(v-base) if m=='PA' else abs(v-base)/max(abs(base),1e-12) for v in views if base is not None and v is not None]
                        deltas+=dd;comparisons+=len(dd);w=max(dd) if dd else None
                        if w is not None:worst.append(w);severe+=w>(targets['severe_PA_deg'] if m=='PA' else targets['severe_relative'])
                        limit=targets['PA_deg'] if m=='PA' else targets[m+'_relative']
                        ok=not absent and w is not None and w<=limit
                        if m=='FL' and ok:
                            # Same inherited ROI absolute change criterion in original native coordinates.
                            scale=r['row']['px_per_mm'] or 1.;span=r['frame']['reference_width']
                            ok=max(abs(v-base)*scale/span for v in views)<=targets['FL_ROI_absolute']
                        passed+=ok
                        cases.append(dict(id=key[1],identity=base,variants=dict(zip(VIEWS[1:],views)),worst=w,missing=absent,severe=w is not None and w>(3 if m=='PA' else .2)))
                    metrics[m]=dict(total=len(rows),pass_count=passed,any_missing=missing,missing_flip=flips,severe_count=severe,
                         comparable_view_pairs=comparisons,p90_worst=float(np.quantile(worst,.9)) if worst else None,median_worst=float(np.median(worst)) if worst else None,
                         maximum=max(worst,default=None),cases=cases,threshold_source='unchanged Round38 targets; 3 registered transforms, not historical26')
                out[ds][seed][arm]=metrics
    return out

def natural(identities):
    lookup={r['row']['dataset']+'|'+r['row']['image_id']:r for r in identities};old=readgz(ROOT/'runs/round47_boundary_selection_v1/evaluation/natural_pairs.json.gz')
    pairs={tuple(sorted((r['a'],r['b']))):dict(a=r['a'],b=r['b'],strict=r['strict'],group=r['group']) for r in old}
    out={}
    for strict in (False,True):
        pp=[p for p in pairs.values() if not strict or p['strict']];out['strict' if strict else 'all']={}
        for seed in SEED_NAMES:
            out['strict' if strict else 'all'][seed]={}
            for arm in ARMS:
                mm={}
                for m in METRICS:
                    delta=[];missing=0
                    for p in pp:
                        a,b=lookup.get(p['a']),lookup.get(p['b'])
                        if a is None or b is None:continue
                        v=[r['outputs'][seed][arm]['values'][m] for r in (a,b)]
                        if any(x is None for x in v):missing+=1;continue
                        if m=='PA':d=abs(v[1]-v[0])
                        else:
                            norm=[x*(r['row']['px_per_mm'] or 1)/r['frame']['reference_width'] for x,r in zip(v,(a,b))]
                            d=abs(norm[1]-norm[0])/max((abs(norm[0])+abs(norm[1]))/2,1e-12)
                        delta.append(d)
                    mm[m]=dict(total_pairs=len(pp),available=len(delta),missing=missing,median=float(np.median(delta)) if delta else None,
                       p90=float(np.quantile(delta,.9)) if delta else None,over_severe=sum(d>(3 if m=='PA' else .2) for d in delta))
                out['strict' if strict else 'all'][seed][arm]=mm
    # New source pairs can change anatomically: compare predicted change with reference change.
    groups=defaultdict(list)
    for r in identities:
        if r['row']['dataset']=='young78':groups[r['row']['group']].append(r)
    out['young39_change_error']={}
    for seed in SEED_NAMES:
        out['young39_change_error'][seed]={}
        for arm in ARMS:
            mm={}
            for m in METRICS:
                errors=[];missing=0
                for group,rs in groups.items():
                    if len(rs)!=2:continue
                    a,b=rs;pa,pb=[r['outputs'][seed][arm]['values'][m] for r in rs]
                    if pa is None or pb is None:missing+=1;continue
                    errors.append(abs((pb-pa)-(b['row']['reference'][m]-a['row']['reference'][m])))
                mm[m]=dict(total_groups=len(groups),available=len(errors),missing=missing,mae_change_error=float(np.mean(errors)) if errors else None,p90=float(np.quantile(errors,.9)) if errors else None)
            out['young39_change_error'][seed][arm]=mm
    return out

def manual_boundary(identities):
    frozen={(r['key'],r['view']):r for r in j(ROOT/'runs/frozen_boundary_20261110/manifest.json')['rows']};rows=[]
    for r in identities:
        if r['row']['dataset']!='apo153':continue
        key='apo153|'+r['row']['image_id'];fr=frozen[(key,'identity')]
        if not fr['manual']:continue
        with np.load(ROOT/fr['input']['path']) as z:x=z['x']+fr['input']['input_box'][0];ys=z['ys']+fr['input']['input_box'][1];radius=float(z['radius'])
        result={}
        for arm,curve in [('B',r['baseline_curve']),('Q',r['qualified_curve'])]:
            result[arm]={}
            for i,side in enumerate(('sup','deep')):
                if curve is None:result[arm][side]=dict(coverage=0,mae_px=None);continue
                fit=curve[side];support=(x>=fit['knots_x'][0])&(x<=fit['knots_x'][-1]);err=abs(np.interp(x,fit['knots_x'],fit['knots_y'])-ys[i])
                result[arm][side]=dict(coverage=float(np.mean(support&(err<=radius))),support=float(support.mean()),mae_px=float(err[support].mean()) if support.any() else None)
        rows.append(dict(id=r['row']['image_id'],arms=result))
    summary={arm:{side:dict(images=len(rows),mean_correct_coverage=float(np.mean([r['arms'][arm][side]['coverage'] for r in rows])),
        mean_image_MAE_px=float(np.mean([r['arms'][arm][side]['mae_px'] for r in rows if r['arms'][arm][side]['mae_px'] is not None]))) for side in ('sup','deep')} for arm in ARMS}
    return dict(summary=summary,rows=rows)

HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Round75 · 支持资格与测量几何</title><style>body{background:#0c1522;color:#e8f0fa;max-width:1450px;margin:18px auto;padding:0 14px;font:16px system-ui}header{position:sticky;top:0;background:#15283af5;z-index:10;padding:12px;border-radius:9px}select,button{padding:7px;margin:3px;background:#263e54;border:1px solid #4e6c85;color:inherit;font:inherit;border-radius:5px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{background:#17283b;padding:13px;border-radius:10px}svg{width:100%;background:#03070d}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #385067;padding:6px;text-align:left}a{color:#8dd9ff}pre{white-space:pre-wrap;overflow-wrap:anywhere}.small{font-size:14px;color:#bdcde0}.warning{color:#ffd28b}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<header><select id="ds"><option value="external34">原34图</option><option value="young78">新增78图</option><option value="apo153">边界开发153图</option><option value="dev282">开发相似282图</option></select><button id="prev">上一张</button><select id="pick"></select><button id="next">下一张</button><span id="count"></span><br><select id="view"><option value="identity">原图</option><option value="translate2_-1">平移+2/-1像素</option><option value="brightness105">亮度×1.05</option><option value="scale1015">中心缩放×1.015</option></select><select id="seed"><option>20261023</option><option>20261024</option></select><label><input id="reference" type="checkbox">人工参考</label><label><input id="all" type="checkbox" checked>全部C1片段</label><label><input id="extend" type="checkbox">FL求交线</label><label><input id="fitroots" type="checkbox">适配全部交点</label></header>
<h2 id="title"></h2><p class="warning">S密度评分与F额外敏感性折扣未通过受控验收，未扩大全量。本图册只展示通过接口控制的Q候选；全量结果仍需验收，未升级默认流程。</p><p class="small">全部显示对齐到原图坐标。两侧画的是各自实际用于测量的曲线：B原观测点四段拟合；Q完整P几何＋独立支持资格。青/粉：上下边界；白虚线：人工参考；灰绿：C1；亮绿：实际参与FL的片段，粗细随权重变化。原34沿用历史单位假定；新增78参考含辅助补全。</p><div id="message"></div><div class="grid" id="panels"></div><details><summary>实际测量资格、权重和原始数据</summary><pre id="detail"></pre></details><p><a href="../REPORT.txt">整体报告</a> · <a href="../summary.json">统计</a> · <a href="../stability.json">扰动稳定性</a> · <a href="../natural_pairs.json">真实相似图</a> · <a href="../controls/calibration.json">未通过的受控分支</a> · <a href="../figures/S_narrow_control_failure.png">S窄带受控失败图</a> · <a href="../figures/recovery_and_regression.png">恢复与退化案例</a></p><script src="index.js"></script><script>
const $=x=>document.getElementById(x),ns='http://www.w3.org/2000/svg';let list=[],idx=0,serial=0;function el(n,a){let e=document.createElementNS(ns,n);for(let k in a)e.setAttribute(k,a[k]);return e}function line(svg,x,y,c,w=1,dash=''){svg.append(el('polyline',{points:x.map((v,i)=>v+','+y[i]).join(' '),fill:'none',stroke:c,'stroke-width':w,'stroke-dasharray':dash}))}function num(x){return x==null?'缺测':x.toFixed(2)}
async function load(){let token=++serial,row=list[idx],v=$('view').value;if(!row.views.includes(v)){$('view').value='identity';v='identity'}let r=await(await fetch(row.files[v])).json();if(token!==serial)return;let s=$('seed').value;location.replace('#'+encodeURIComponent(row.dataset+'|'+row.id+'|'+v+'|'+s));$('title').textContent=row.id;$('count').textContent=`${idx+1}/${list.length}`;$('pick').value=idx;$('panels').replaceChildren();for(let arm of ['B','Q']){let o=r.outputs[s][arm],cv=r.curves[arm],card=document.createElement('section');card.className='card';let h=document.createElement('h3');h.textContent=arm==='B'?'B · 原正式接口':'Q · 完整几何＋支持资格';card.append(h);let svg=el('svg',{viewBox:`0 0 ${r.width} ${r.height}`});svg.append(el('image',{href:r.image,width:r.width,height:r.height}));if(cv)for(let [side,c]of [['sup','#28cafa'],['deep','#fc6eb9']])line(svg,cv[side].knots_x,cv[side].knots_y,c,1.7);if($('reference').checked&&r.reference_lines){for(let a of r.reference_lines)line(svg,a.map(q=>q[0]),a.map(q=>q[1]),'#fff',1.1,'5 4')}for(let t of r.segments[s]){let w=(o.FL_weights||{})[t.id]||0;if(w>0||$('all').checked)line(svg,[t.p0[0],t.p1[0]],[t.p0[1],t.p1[1]],w>0?'#43f4a2':'#7c9069',w>0?2+5*w:.8)}let actual=(r.measurement_audit||{})[s]?.[arm],points=[];if(actual){for(let t of actual.FL_tracks){let u=t.roots.upper.point,d=t.roots.lower.point;points.push(u,d);if($('extend').checked)line(svg,[u[0],d[0]],[u[1],d[1]],'#ffbc66',.8+3*t.weight,'5 4')}for(let t of actual.MT_samples){let good=t.nearest_joint_support<=t.joint_support_threshold;for(let y of [t.upper_y,t.lower_y])svg.append(el('circle',{cx:t.x,cy:y,r:2.7,fill:good?'#8aff9d':'#ffae4b'}))}}if($('fitroots').checked&&points.length){let xs=[0,r.width,...points.map(p=>p[0])],ys=[0,r.height,...points.map(p=>p[1])],l=Math.min(...xs),t=Math.min(...ys),w=Math.max(...xs)-l,h=Math.max(...ys)-t;svg.setAttribute('viewBox',`${l-.02*w} ${t-.02*h} ${w*1.04} ${h*1.04}`)}card.append(svg);let table=document.createElement('table');table.innerHTML='<tr><th>指标</th><th>预测</th><th>参考</th><th>绝对误差</th></tr>'+['MT','PA','FL'].map(m=>`<tr><td>${m} ${m==='PA'?'°':r.unit}</td><td>${num(o.values[m])}</td><td>${r.reference?num(r.reference[m]):'无标量参考'}</td><td>${r.reference&&o.values[m]!=null?num(Math.abs(o.values[m]-r.reference[m])):'—'}</td></tr>`).join('');card.append(table);let p=document.createElement('p');p.className='small';p.textContent='MT原因：'+(o.MT_reasons||[]).join('/')+'；FL实际贡献 '+Object.values(o.FL_weights||{}).filter(w=>w>0).length+'段';card.append(p);$('panels').append(card)}$('detail').textContent=JSON.stringify({Q_adapter:r.Q_adapter,outputs:r.outputs,actual_measurement:r.measurement_audit},null,2);$('message').textContent=r.reference?'本图参考仅用于验收，不参与预测或规则选择。':'本图用于边界/相似图诊断；没有完整三项标量参考。'}
function rebuild(){list=INDEX.filter(r=>r.dataset===$('ds').value);idx=0;$('pick').innerHTML=list.map((r,i)=>`<option value="${i}">${r.id}</option>`).join('');load()}$('ds').onchange=rebuild;$('pick').onchange=()=>{idx=+$('pick').value;load()};$('prev').onclick=()=>{idx=(idx+list.length-1)%list.length;load()};$('next').onclick=()=>{idx=(idx+1)%list.length;load()};for(let k of ['view','seed','reference','all','extend','fitroots'])$(k).onchange=load;let parts=decodeURIComponent(location.hash.slice(1)).split('|');if(['external34','young78','apo153','dev282'].includes(parts[0]))$('ds').value=parts[0];if(['identity','translate2_-1','brightness105','scale1015'].includes(parts[2]))$('view').value=parts[2];if(['20261023','20261024'].includes(parts[3]))$('seed').value=parts[3];list=INDEX.filter(r=>r.dataset===$('ds').value);$('pick').innerHTML=list.map((r,i)=>`<option value="${i}">${r.id}</option>`).join('');idx=Math.max(0,list.findIndex(r=>r.id===parts[1]));load();</script></html>'''

def figures(identities):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    referred=[r for r in identities if r['row']['reference']]
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for i,ds in enumerate(('external34','young78')):
        rs=[r for r in referred if r['row']['dataset']==ds]
        for ax,m in zip(axes[i],METRICS):
            for arm,c in [('B','#3d93c0'),('Q','#e38c34')]:
                rr=[r for r in rs if r['outputs']['20261023'][arm]['values'][m] is not None]
                ax.scatter([r['row']['reference'][m] for r in rr],[r['outputs']['20261023'][arm]['values'][m] for r in rr],s=15,alpha=.7,color=c,label=arm)
            lo,hi=ax.get_xlim();yl,yh=ax.get_ylim();a,b=min(lo,yl),max(hi,yh);ax.plot([a,b],[a,b],'k--',lw=.7)
            ax.set_title(f'{ds} | {m}');ax.set_xlabel('Reference');ax.set_ylabel('Measured');ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(dest/'accuracy_B_Q.png',dpi=150);plt.close(fig)
    # A real recovery and the largest FL degradation; both selected after all predictions, no tuning.
    recovery=next(r for r in referred if r['row']['image_id']=='ANON_10')
    usable=[r for r in referred if all(r['outputs']['20261023'][a]['values']['FL'] is not None for a in ARMS)]
    worse=max(usable,key=lambda r:abs(r['outputs']['20261023']['Q']['values']['FL']-r['row']['reference']['FL'])-abs(r['outputs']['20261023']['B']['values']['FL']-r['row']['reference']['FL']))
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for i,r in enumerate((recovery,worse)):
        for j,arm in enumerate(ARMS):
            ax=axes[i,j];ax.imshow(Image.open(ROOT/r['row']['image_path']).convert('L'),cmap='gray');cv=r['baseline_curve'] if arm=='B' else r['qualified_curve'];o=r['outputs']['20261023'][arm]
            if cv:
                for s,c in [('sup','#23cafa'),('deep','#ff6cbb')]:ax.plot(cv[s]['knots_x'],cv[s]['knots_y'],color=c,lw=1.5)
            for s in r['segments']['20261023']:
                weight=(o['FL_weights'] or {}).get(str(s['label_id']),0);ax.plot([s['p0'][0],s['p1'][0]],[s['p0'][1],s['p1'][1]],color='#4ada98' if weight else '#7b8865',lw=.8+3*weight)
            mt=o['values']['MT'];fl=o['values']['FL'];ax.set_title(f"{r['row']['image_id']} | {arm} | MT {mt if mt is None else round(mt,2)} | FL {fl if fl is None else round(fl,2)}");ax.axis('off')
    fig.tight_layout();fig.savefig(dest/'recovery_and_regression.png',dpi=140);plt.close(fig)
    write(OUT/'selected_cases.json',dict(recovery=recovery['row']['image_id'],FL_regression=worse['row']['image_id']))

def gallery(rows):
    dest=OUT/'gallery';(dest/'data').mkdir(parents=True,exist_ok=True);index={}
    oldfrozen={(r['key'],r['view']):r for r in j(ROOT/'runs/frozen_boundary_20261110/manifest.json')['rows']}
    for r in rows:
        row=r['row'];ds,sid=row['dataset'],row['image_id'];key=(ds,sid);view=r['view'];filename=f'{ds}__{sid}__{view}.json'
        cv={}
        for arm,curve in [('B',r['baseline_curve']),('Q',r['qualified_curve'])]:
            cv[arm]={s:{k:curve[s][k] for k in ('knots_x','knots_y','left_ray','right_ray','support')} for s in ('sup','deep')} if curve else None
        lines=None
        if ds=='young78':
            g=j(ROOT/f'validation/young_expert_v03/geometry/{sid}.json');off=row['annotation_offset'];lines=[(np.array(g[s]['source_dense'])-[off,0]).tolist() for s in ('upper','lower')]
        elif ds=='apo153':
            fr=oldfrozen[(ds+'|'+sid,'identity')]
            if fr['manual']:
                with np.load(ROOT/fr['input']['path']) as z:lines=[np.c_[z['x']+fr['input']['input_box'][0],y+fr['input']['input_box'][1]].tolist() for y in z['ys']]
        obj=dict(id=sid,dataset=ds,view=view,image='../../../'+row['image_path'],width=row['width'],height=row['height'],unit='mm' if row['px_per_mm'] else 'px',
           reference=row['reference'],reference_lines=lines,curves=cv,outputs=r['outputs'],Q_adapter=r['Q_adapter'],
           segments={s:[dict(id=t['label_id'],p0=t['p0'],p1=t['p1']) for t in segs] for s,segs in r['segments'].items()})
        audit=OUT/'measurement_audit'/ds/sid/(view+'.json')
        if audit.exists():obj['measurement_audit']=j(audit)
        write(dest/'data'/filename,obj)
        if key not in index:index[key]=dict(dataset=ds,id=sid,views=[],files={})
        index[key]['views'].append(view);index[key]['files'][view]='data/'+filename
    (dest/'index.js').write_text('const INDEX='+json.dumps(list(index.values()),ensure_ascii=False)+';',encoding='utf8');(dest/'index.html').write_text(HTML,encoding='utf8')

def main():
    paths=sorted((OUT/'results').rglob('*.json.gz'));rows=[readgz(p) for p in paths];identity=[r for r in rows if r['view']=='identity']
    assert len(identity)==547 and len(rows)==883,(len(identity),len(rows))
    bykey={(r['row']['dataset'],r['row']['image_id'],r['view']):r for r in rows}
    summary={};comparison={}
    for ds in ('external34','young78','combined112'):
        rr=[r for r in identity if r['row']['reference'] and (ds=='combined112' or r['row']['dataset']==ds)]
        summary[ds]={s:{a:{m:metric(rr,s,a,m) for m in METRICS} for a in ARMS} for s in SEED_NAMES}
        comparison[ds]={s:{m:compare(rr,s,m) for m in METRICS} for s in SEED_NAMES}
    availability={ds:{s:{a:{m:sum(r['outputs'][s][a]['values'][m] is not None for r in identity if r['row']['dataset']==ds) for m in METRICS} for a in ARMS} for s in SEED_NAMES} for ds in ('external34','young78','apo153','dev282')}
    write(OUT/'summary.json',summary);write(OUT/'common_and_recovered.json',comparison);write(OUT/'availability.json',availability)
    write(OUT/'stability.json',stability(identity,bykey));write(OUT/'natural_pairs.json',natural(identity));write(OUT/'manual_boundary.json',manual_boundary(identity))
    with (OUT/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['dataset','id','view','seed','arm','MT','PA','FL'];w=csv.DictWriter(f,fields);w.writeheader()
        for r in rows:
            for s in SEED_NAMES:
                for a in ARMS:w.writerow(dict(dataset=r['row']['dataset'],id=r['row']['image_id'],view=r['view'],seed=s,arm=a,**r['outputs'][s][a]['values']))
    figures(identity);gallery(rows)
    write(OUT/'report_data_ready.json',dict(originals=547,views=883,model_seeds=2,measurement_outputs=883*4,
         failed_branches_not_expanded=['S','F'],baseline112_exact_replays=sum(len(r['baseline112_parity']) for r in identity)))
    print('R75_REPORT_DATA_READY',len(rows),flush=True)

if __name__=='__main__':main()
