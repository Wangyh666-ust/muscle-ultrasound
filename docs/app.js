'use strict';
const data = window.SHOWCASE_DATA;
const records = new Map(data.records.map(r => [r.id, r]));
const colors = {apo:'#38bdf8',mt:'#5ee7b5',pa:'#f29bc7',fl:'#ffb454'};
const el = id => document.getElementById(id);
let current = data.featured[0], mode = 'measure';
const fmt = (v,n=2) => Number.isFinite(v) ? v.toFixed(n) : '未输出';
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const value = (p,k) => k==='MT' ? p.value_mm : k==='PA' ? p.value_deg : p.selected_mean_mm;
const line = (a,b,color,width=2,dash=false) => `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="${width}" ${dash?'stroke-dasharray="8 6"':''} vector-effect="non-scaling-stroke"/>`;
const failText = r => r.apo.status!=='candidate' ? '上下边界支持不足，当前规则拒绝测量。' : current==='im_06_arch' ? '已知边界选择偏差案例：有厚度输出并不意味着选对了边界。' : current==='im_20_arch' ? '已知偏薄案例：PA 选线还包含结构异常诊断，未自动过滤。' : current==='im_29_arch' ? '已知 FL 偏高案例：多条线的一致输出也可能存在共同偏差。' : 'G1 允许交点在画面外，仍限制外推比例 ≤ 80%。输出不等于测量已验证正确。';

function choose(id){if(!records.has(id)) return;current=id;el('imageSelect').value=id;render();}
function draw(r,k){
 const p=r.parameters[k], fits=r.apo.fits, chosen=p.selected_details||[];
 let xmin=0,ymin=0,xmax=r.width,ymax=r.height, svg='';
 if(mode==='measure') for(const d of chosen) for(const pt of [d.intersection_sup,d.intersection_deep]) if(pt&&pt.every(Number.isFinite)){xmin=Math.min(xmin,pt[0]);xmax=Math.max(xmax,pt[0]);ymin=Math.min(ymin,pt[1]);ymax=Math.max(ymax,pt[1]);}
 const pad=Math.max(r.width,r.height)*.025;
 svg+=`<image href="${r.image}" width="${r.width}" height="${r.height}"/>`;
 if(mode==='mask') for(const task of ['apo','fasc']) svg+=`<image href="assets/images/${r.id}_${task}.png" width="${r.width}" height="${r.height}"/>`;
 if(mode==='measure' && fits && r.apo.status==='candidate'){
   const [lo,hi]=r.apo.support;
   const y=(key,x)=>fits[key][0]*x+fits[key][1];
   for(const key of ['sup','deep']){
     svg+=line([lo,y(key,lo)],[hi,y(key,hi)],colors.apo,2.4);
     if(xmin<lo)svg+=line([xmin,y(key,xmin)],[lo,y(key,lo)],colors.apo,1,true);
     if(xmax>hi)svg+=line([hi,y(key,hi)],[xmax,y(key,xmax)],colors.apo,1,true);
   }
   for(const x of r.parameters.MT.selected_positions_px||[])svg+=line([x,y('sup',x)],[x,y('deep',x)],colors.mt,1.8);
   for(const d of r.parameters.PA.selected_details||[])svg+=line(d.p0,d.p1,colors.pa,3);
   for(const d of chosen){
     svg+=line(d.intersection_sup,d.intersection_deep,colors.fl,2.1,true);
     svg+=line(d.p0,d.p1,colors.fl,4);
     for(const a of [d.intersection_sup,d.intersection_deep]) svg+=`<circle cx="${a[0]}" cy="${a[1]}" r="${Math.max(r.width,r.height)*.004}" fill="${colors.fl}"/>`;
   }
 }
 el('canvasArea').innerHTML=`<svg role="img" aria-label="${r.id} ${mode} ${k}" viewBox="${xmin-pad} ${ymin-pad} ${xmax-xmin+2*pad} ${ymax-ymin+2*pad}" xmlns="http://www.w3.org/2000/svg">${svg}</svg>`;
 el('legend').innerHTML=mode==='original'?'原始图像 · 未叠加标注':mode==='mask'?`<span style="--c:${colors.apo}">边界预测 mask</span><span style="--c:${colors.fl}">肌束预测 mask</span>`:`<span style="--c:${colors.apo}">上下边界</span><span style="--c:${colors.mt}">三处厚度</span><span style="--c:${colors.pa}">角度选线</span><span style="--c:${colors.fl}">长度选线与延长线</span>`;
}
function render(){
 const r=records.get(current),policy=el('policy').value;
 el('currentId').textContent=current;
 el('caseType').textContent=r.featured?'精选示例':'完整集合';
 el('imageMeta').textContent=`${r.width} × ${r.height} px · 提供尺度 ${fmt(r.scale,3)} px/mm · 原图与预测坐标一致`;
 draw(r,policy);
 const parts=[['MT','厚度 · 两边界间距','mm',colors.mt],['PA','角度 · 相对深边界','°',colors.pa],[policy,`长度 · ${policy==='FL_G1'?'G1 实验性外推':'G0 严格规则'}`,'mm',colors.fl]];
 el('measurements').innerHTML=parts.map(([k,label,unit,color])=>{
  const p=r.parameters[k],v=value(p,k),ref=r.reference[k],count=p.selected_count||0;
  let detail=`实际选中 ${count} ${k==='MT'?'个位置':'条候选'}`;
  if(k.startsWith('FL')&&count)detail+=` · 最大外推 ${fmt(Math.max(...p.extrapolation_ratios)*100,1)}%`;
  const comparison=el('showReference').checked?`<div class="metric-reference">专家均值 ${fmt(ref.mean)} ${unit} · ${ref.n} 位<br>${Number.isFinite(v)?'探索性差值 '+(v-ref.mean>=0?'+':'')+fmt(v-ref.mean)+' '+unit:'无输出，不填零、不计算误差'}</div>`:'';
  return `<div class="measurement"><div class="metric-label"><span>${label}</span><span>${k.replace('FL_','')}</span></div><div class="metric-value" style="color:${color}">${fmt(v)}${Number.isFinite(v)?`<small>${unit}</small>`:''}</div><div class="metric-detail">${detail}</div>${comparison}</div>`;
 }).join('');
 el('caseNote').textContent=failText(r);
 document.querySelectorAll('#featured button').forEach(b=>b.classList.toggle('active',b.dataset.image===current));
 document.querySelectorAll('#viewMode button').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));
}
el('featured').innerHTML=data.featured.map((id,i)=>`<button data-image="${id}">精选 ${i+1} · ${id.slice(3,5)}</button>`).join('');
el('imageSelect').innerHTML=data.records.map(r=>`<option value="${esc(r.id)}">${esc(r.id)}${r.featured?' · 精选':''}</option>`).join('');
document.querySelectorAll('[data-image]').forEach(b=>b.addEventListener('click',()=>{choose(b.dataset.image);if(b.closest('.failure-links'))el('explorer').scrollIntoView({behavior:'smooth'});}));
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;render();}));
el('imageSelect').addEventListener('change',e=>choose(e.target.value));
el('policy').addEventListener('change',render);el('showReference').addEventListener('change',render);
el('metricsBody').innerHTML=[['MT','厚度 MT','mm'],['PA','角度 PA','°'],['FL_G0','长度 G0','mm'],['FL_G1','长度 G1（实验）','mm']].map(([k,label,u])=>`<tr><td>${label}</td><td>${data.counts[k]} / 34</td><td>${fmt(data.metrics[k].mae)} ${u} <small>n=${data.metrics[k].n}</small></td></tr>`).join('');
choose(current);
