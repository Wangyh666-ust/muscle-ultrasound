"""Offline per-image lazy-loaded gallery; includes every similarity group."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from scripts.round38_common import *
from scripts.round38_report import LABEL
from scripts.round29_core import extended_fl

def thin(a,row):
    curve=a['curve'];cv={s:{k:curve[s][k] for k in ('knots_x','knots_y','left_ray','right_ray')} for s in ('sup','deep')} if curve else None
    segments=a.get('segments',[c['segment'] for c in a.get('candidates',[])]);ids=set(a['selection']['selected_ids']);cs=[]
    for s in segments:
        q={k:s[k] for k in ('label_id','p0','p1','centroid','direction_deg')}
        if s['label_id'] in ids and curve:
            f=extended_fl(s,curve,row['width'],row['height']);q.update(sup=f['intersection_sup'],deep=f['intersection_deep'])
        cs.append(q)
    return dict(values=a['values'],stage=a['stage'],curve=cv,segments=cs,selection=a['selection'],PA_ids=a['PA_ids'],
        FL_weights=a['FL_weights'],PA_weights=a['PA_weights'],roi=a['roi'],exploratory_values=a['exploratory_values'],
        contributors={p:len(a[p+'_weights']) for p in ('PA','FL')},aggregation={p:a['aggregation']['metrics'][p]['dispersion_original_units'] for p in ('PA','FL')} if a.get('aggregation') else None)

def main():
    assert (BASE/'evaluation_success.json').exists();dest=BASE/'gallery';dest.mkdir(exist_ok=True);(dest/'data').mkdir(exist_ok=True);(dest/'images').mkdir(exist_ok=True)
    groupdoc=j(S01/'groups.json');member=members();extreme=gzread(BASE/'evaluation/extreme_cases.json.gz')
    top_extreme=sorted([e for e in extreme if e['gentle'] and e['metric']=='FL' and e['relative'] is not None],key=lambda e:-e['relative'])[:50]
    extra={e['key'] for e in top_extreme};rr=[r for r in ordered_inputs() if r['dataset']!='dev282' or key(r) in member or key(r) in extra]
    worst=gzread(BASE/'evaluation/worst_by_image.json.gz');wi={(r['key'],r['seed'],r['arm']):r for r in worst};catalog=[];views=0
    for i,r in enumerate(rr):
        ident=r['dataset']+'_'+Path(r['image_id']).stem;im=Image.open(ROOT/r['image_path']).convert('RGB');im.thumbnail((1100,900));im.save(dest/'images'/f'{ident}.jpg',quality=88)
        row=dict(key=key(r),id=ident,dataset=r['dataset'],image_id=r['image_id'],width=r['width'],height=r['height'],image='images/'+ident+'.jpg',transforms=list(transforms(r)),seeds={})
        for seed in SEEDS:
            row['seeds'][seed]={}
            for tr in transforms(r):
                q=gzread(predpath(r,tr,seed));row['seeds'][seed][tr]={a:thin(v,q) for a,v in q['arms'].items()};views+=8
            row['seeds'][seed]['worst']={a:wi[key(r),seed,a] for a in ARMS}
            row['unit']='mm' if q.get('px_per_mm') else 'px'
        (dest/'data'/f'{ident}.js').write_text('window.receiveImage('+json.dumps(clean(row),ensure_ascii=False,allow_nan=False,separators=(',',':'))+');',encoding='utf8')
        catalog.append({k:row[k] for k in ('key','id','dataset','image_id','image','transforms')})
        if (i+1)%30==0:print('GALLERY',i+1,len(rr),flush=True)
    summary=j(BASE/'evaluation/stability_summary.json')
    index=dict(catalog=catalog,groups=groupdoc['groups'],arms=LABEL,seeds=SEEDS,summary=summary,
        extremes=top_extreme)
    (dest/'index_data.js').write_text('window.INDEX='+json.dumps(index,ensure_ascii=False,allow_nan=False,separators=(',',':'))+';',encoding='utf8')
    (dest/'index.html').write_text(HTML,encoding='utf8');(dest/'app.js').write_text(JS,encoding='utf8')
    write(BASE/'gallery_success.json',dict(images=len(rr),views=views,groups=len(groupdoc['groups']),offline=True,not_browser_render_verified=True))
    print('GALLERY DONE',len(rr),views,flush=True)

HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Round38 · 稳定性诊断</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1420;color:#e4eef8;font:15px/1.55 system-ui,sans-serif}header{padding:18px 24px;background:#152437}h1{font-size:23px;margin:0 0 8px}p{margin:8px 0}.controls{position:sticky;top:0;background:#152437ee;padding:12px 24px;z-index:5;display:flex;gap:8px;flex-wrap:wrap}select,button{font:inherit;background:#243b54;color:#fff;border:1px solid #48617c;border-radius:6px;padding:6px}main{padding:18px 24px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px;border-bottom:1px solid #345;text-align:left}.scroll{overflow:auto}.panels{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.panel{background:#152437;border:1px solid #314b66;border-radius:9px;padding:12px;min-width:0}canvas{display:block;width:100%;background:#05080d}h2{font-size:17px;margin:4px 0 10px}.value{font-size:19px;font-weight:650}.muted{color:#9eb4cb}.warn{color:#ffcb71}a{color:#83c9ff}#status{min-height:1.5em}.controls label{white-space:nowrap}details{margin:12px 0}.badge{padding:4px 8px;background:#243b54;border-radius:4px;margin-right:6px}@media(max-width:900px){.panels{grid-template-columns:1fr}main,header,.controls{padding:12px}}
</style><header><h1>Round38 · 同一图像变化后，测量还能稳定吗？</h1><p>两种子 · 全469图 · 八组 · 边界与肌束均重新预测。先看最坏波动、缺测和极端跳变。</p><p class="muted">全部数据为内部诊断；没有训练、没有更改默认流程。MT/FL单位随图片显示；开发图像的px不是mm。代表线不等于汇总所用的全部肌束。</p></header>
<div class="controls"><select id="dataset"><option value="external34">外部34图</option><option value="apo153">边界开发153图</option><option value="groups">全部44相似组</option><option value="dev282">相似组/极端开发病例</option></select><select id="group"></select><button id="prev">上一张</button><select id="image"></select><button id="next">下一张</button><select id="seed"></select><select id="arm"></select><select id="transform"></select><select id="mode"><option value="within">同组：原图 vs 扰动</option><option value="arms">同扰动：R vs 所选组</option></select><label><input id="all" type="checkbox" checked>全部候选</label><label><input id="rays" type="checkbox">交点延长线</label></div>
<main><div id="status" class="muted"></div><details><summary>全量稳定性结果与验收口径</summary><p>联合通过率分母固定为同种子R原图三项可测图片。任一变体缺测即不通过；P95只描述能观察到的数值，必须同时看缺测。目标：MT≤2%、PA≤1°、FL≤5%且≤ROI宽3%；至少95%图片全部通过。</p><div class="scroll" id="summary"></div></details><div id="worst"></div><div class="panels"><section class="panel"><h2 id="title0"></h2><canvas id="c0" width="1000" height="720"></canvas><div id="text0"></div></section><section class="panel"><h2 id="title1"></h2><canvas id="c1" width="1000" height="720"></canvas><div id="text1"></div></section></div><p class="muted">蓝：边界观测拟合；黄：候选片段；绿：FL代表线；粉：参与PA计算；橙：代表线交点延长。扰动结果已映回原图坐标以便对照，预测时没有借用原图结果。</p><details><summary>严重FL跳变入口（全量检索，不限成功图）</summary><div id="cases"></div></details></main><script src="index_data.js"></script><script src="app.js"></script></html>'''

JS=r'''
const $=id=>document.getElementById(id),I=window.INDEX;let current=null,requestedTransform=null,img=new Image(),cache={};const fmt=(v,n=2)=>v==null?'—':Number(v).toFixed(n),pct=v=>v==null?'—':fmt(v*100)+'%';
const T={identity:'原图',translate2_3:'右移2px、下移3px','translate-3_-2':'左移3px、上移2px',scale095:'缩放至95%',scale105:'缩放至105%',pad10:'四周各加10%黑边',brightness095:'亮度降低5%',scale075:'缩放至75%',scale125:'缩放至125%',pad20:'四周各加20%黑边','translate1_-1':'右移1px、上移1px'};
const options=(el,rows)=>{el.replaceChildren(...rows.map(([v,t])=>{let o=document.createElement('option');o.value=v;o.textContent=t;return o}))};
options($('seed'),I.seeds.map(x=>[x,x]));options($('arm'),Object.entries(I.arms).map(([k,v])=>[k,k+' · '+v]));options($('group'),I.groups.map(g=>[g.id,g.id+' · '+g.members.length+'图']));
function subset(){return $('dataset').value==='groups'?I.catalog.filter(r=>I.groups.find(g=>g.id===$('group').value).members.includes(r.key)):I.catalog.filter(r=>r.dataset===$('dataset').value)}
function refreshList(key){$('group').style.display=$('dataset').value==='groups'?'':'none';options($('image'),subset().map(r=>[r.key,r.image_id]));if(key&&subset().some(r=>r.key===key))$('image').value=key;load()}
window.receiveImage=data=>{cache[data.key]=data;if(data.key===$('image').value){current=data;ready()}};
function load(){const key=$('image').value;current=null;$('status').textContent='载入 '+key;let r=I.catalog.find(r=>r.key===key);if(!r)return;if(cache[key]){current=cache[key];ready();return}let s=document.createElement('script');s.src='data/'+r.id+'.js';s.onerror=()=>$('status').textContent='无法载入图像数据';document.body.append(s)}
function ready(){let old=requestedTransform||$('transform').value;requestedTransform=null;options($('transform'),current.transforms.map(x=>[x,T[x]||x]));if(current.transforms.includes(old))$('transform').value=old;else $('transform').value='translate2_3';img.onload=render;img.src=current.image;render()}
function render(){if(!current)return;let seed=$('seed').value,arm=$('arm').value,tr=$('transform').value,mode=$('mode').value,aa=mode==='within'?arm:'R',ta=mode==='within'?'identity':tr;const s=current.seeds[seed];if(!s)return;let a=s[ta][aa],b=s[tr][arm];$('title0').textContent=aa+' · '+I.arms[aa]+' · '+ta;$('title1').textContent=arm+' · '+I.arms[arm]+' · '+tr;draw($('c0'),a);draw($('c1'),b);textPanel($('text0'),a);textPanel($('text1'),b);
let w=s.worst[arm];let ranges=['MT','PA','FL'].map(p=>{let vv=current.transforms.slice(0,7).map(t=>s[t][arm].values[p]).filter(v=>v!=null);return p+' '+(vv.length?fmt(Math.min(...vv))+'–'+fmt(Math.max(...vv)):'—')+' '+(p==='PA'?'°':current.unit)+'（'+vv.length+'/7有值）'}).join('； ');$('worst').innerHTML='<p><b>该图六种轻微扰动中的最坏变化：</b> MT '+pct(w.parameters.MT.worst)+'； PA '+fmt(w.parameters.PA.worst)+'°； FL '+pct(w.parameters.FL.worst)+'。 <span class="warn">'+(w.joint_pass?'本图通过':'本图未通过')+(w.any_missing_flip?' · 出现缺测切换':'')+'</span></p><p class="muted">原图及轻微扰动的实际数值范围：'+ranges+'。这是本次扰动范围，不是置信区间。</p>';
$('status').textContent=current.key+' · '+($('image').selectedIndex+1)+'/'+$('image').options.length+' · 像素坐标映回原图比较';location.hash=encodeURIComponent([current.key,seed,arm,tr,mode].join('|'))}
function draw(canvas,a){let ctx=canvas.getContext('2d'),cw=canvas.width,ch=canvas.height;ctx.fillStyle='#05080d';ctx.fillRect(0,0,cw,ch);let scale=Math.min(cw/current.width,ch/current.height),ox=(cw-current.width*scale)/2,oy=(ch-current.height*scale)/2;ctx.save();ctx.translate(ox,oy);ctx.scale(scale,scale);ctx.beginPath();ctx.rect(0,0,current.width,current.height);ctx.clip();if(img.complete&&img.naturalWidth)ctx.drawImage(img,0,0,current.width,current.height);let line=(points,color,width=2,dash=[])=>{if(points.some(p=>!p||p.some(x=>!Number.isFinite(x))))return;ctx.strokeStyle=color;ctx.lineWidth=width/scale;ctx.setLineDash(dash.map(x=>x/scale));ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke()};
if(a.curve)for(let side of ['sup','deep']){let c=a.curve[side];line(c.knots_x.map((x,i)=>[x,c.knots_y[i]]),'#63beff',3)}let ids=new Set(a.selection.selected_ids),pa=new Set(a.PA_ids);for(let s of a.segments){let selected=ids.has(s.label_id);if($('all').checked||selected||pa.has(s.label_id))line([s.p0,s.p1],selected?'#55f6bd':pa.has(s.label_id)?'#ff80b6':'#d8c879',selected?3:1.5);if(selected){if($('rays').checked&&s.sup&&s.deep)line([s.sup,s.deep],'#ffb347',1.5,[6,4]);ctx.fillStyle='#fff';ctx.font=(13/scale)+'px sans-serif';ctx.fillText('#'+s.label_id,s.centroid[0],s.centroid[1]-5/scale)}}let [x0,y0,x1,y1]=a.roi.box;for(let u of [1/3,2/3])line([[x0+(x1-x0)*u,y0],[x0+(x1-x0)*u,y1]],'#91bd9b',1,[4,5]);ctx.restore()}
function textPanel(el,a){let missing=a.selection.missing_regions.join(' / ')||'无';let exp=Object.entries(a.exploratory_values).filter(([k,v])=>v!=null).map(([k,v])=>k+' '+fmt(v)).join(' / ');el.innerHTML='<p class="value">MT '+fmt(a.values.MT)+' '+current.unit+'　PA '+fmt(a.values.PA)+'°　FL '+fmt(a.values.FL)+' '+current.unit+'</p><p>代表线 '+(a.selection.selected_ids.join(' / ')||'无')+'；FL参与 '+a.contributors.FL+' 条，PA参与 '+a.contributors.PA+' 条。<br>区域缺位：'+missing+'；状态：'+a.stage+'</p>'+(exp?'<p class="warn">探索估计（不计正常覆盖）：'+exp+'</p>':'')+(a.aggregation?'<p class="muted">轨迹离散度：FL '+fmt(a.aggregation.FL)+' '+current.unit+' / PA '+fmt(a.aggregation.PA)+'°。不是医学置信区间。</p>':'')}
let th='<table><thead><tr><th>种子</th><th>组</th><th>原图完整</th><th>联合通过</th><th>MT P95</th><th>PA P95</th><th>FL P95</th><th>缺测切换</th></tr></thead><tbody>';
for(let q of I.summary){let p=q.parameters;th+='<tr><td>'+q.seed+'</td><td>'+q.arm+'</td><td>'+q.original_complete_n+'/469</td><td>'+q.joint_pass_count+'/'+q.reference_complete_n+' ('+pct(q.joint_pass_fraction)+')</td><td>'+pct(p.MT.worst_numeric_observed_only.p95)+'</td><td>'+fmt(p.PA.worst_numeric_observed_only.p95)+'°</td><td>'+pct(p.FL.worst_numeric_observed_only.p95)+'</td><td>'+q.missing_flip_count+'</td></tr>'}$('summary').innerHTML=th+'</tbody></table>';
$('cases').replaceChildren(...I.extremes.map(e=>{let b=document.createElement('button');b.textContent=e.key+' '+e.arm+' '+e.transform+' FL '+pct(e.relative);b.onclick=()=>{let r=I.catalog.find(r=>r.key===e.key);if(!r)return;$('dataset').value=r.dataset;$('seed').value=e.seed;$('arm').value=e.arm;$('mode').value='within';requestedTransform=e.transform;refreshList(r.key)};return b}));
for(let id of ['seed','arm','transform','mode','all','rays'])$(id).onchange=render;for(let id of ['dataset','group'])$(id).onchange=()=>refreshList();$('image').onchange=load;$('prev').onclick=()=>{let e=$('image');e.selectedIndex=(e.selectedIndex-1+e.options.length)%e.options.length;load()};$('next').onclick=()=>{let e=$('image');e.selectedIndex=(e.selectedIndex+1)%e.options.length;load()};
let hash=decodeURIComponent(location.hash.slice(1)).split('|');if(hash.length>=6){let key=hash[0]+'|'+hash[1],r=I.catalog.find(r=>r.key===key);if(r){$('dataset').value=r.dataset;$('seed').value=hash[2];$('arm').value=hash[3];$('mode').value=hash[5];requestedTransform=hash[4];refreshList(key)}else refreshList()}else refreshList();
'''

if __name__=='__main__':main()
