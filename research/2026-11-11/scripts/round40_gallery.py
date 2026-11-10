"""Offline full-cohort gallery, original/new/stress inputs and eight methods."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from scripts.round40_common import *
from scripts.round38_gallery import HTML,JS
LABEL=dict(J='冻结Round39基线',Qoff='取消C1硬拒绝',Qobs='只比较有边界支持的肌束',Qsoft='连续肌束证据选带',T='平滑端部外延',U='边界敏感性FL权重',TU='平滑外延＋敏感性',K='预定完整组合')

def display(a):
    cv=a['curve'];curve={s:{k:cv[s][k] for k in ('knots_x','knots_y','left_ray','right_ray')} for s in ('sup','deep')} if cv else None
    cs=[]
    for c in a['candidates']:
        s=c['segment'];q={k:s[k] for k in ('label_id','p0','p1','centroid','direction_deg')}
        q.update(sup=c['intersection_sup'],deep=c['intersection_deep']);cs.append(q)
    agg=a['aggregation'];risk=a.get('boundary_sensitivity');decision=[];gates=a.get('pair_eligibility') or []
    if cv is None:
        rs=sorted({x for g in gates for x in g['reasons']});decision.append('没有正常边界输出；'+(' / '.join(rs) if rs else '未生成合适候选'))
    else:
        g=next((x for x in gates if x['pair_id']==a['selected_pair_id']),{})
        if g.get('low_evidence') or not g.get('C1_gate'):decision.append('肌束资格证据不足；保留边界输出不等于可靠')
        if g.get('C1_gate') and g.get('C1_containment') is not None and g['C1_containment']<.6:decision.append('仍有肌束分布冲突；放开资格或软排序保留了这对边界，不代表恢复正确')
    if risk and risk['all_tracks_unreliable']:decision.append('所有FL贡献轨迹均未通过敏感性可靠条件；归一化权重不能代表可靠')
    return dict(values=a['values'],stage=a['stage'],curve=curve,segments=cs,selection=a['selection'],PA_ids=a['PA_ids'],FL_weights=a['FL_weights'],PA_weights=a['PA_weights'],roi=a['roi'],exploratory_values=a['exploratory_values'],
        contributors={m:len(a[m+'_weights']) for m in ('FL','PA')},aggregation={m:agg['metrics'][m].get('dispersion_original_units') for m in ('FL','PA')} if agg else None,decision='；'.join(decision),risk=risk,selected_pair_id=a['selected_pair_id'])

def main():
    assert j(DEST/'evaluation_success.json')['measurements']==97280;dest=DEST/'gallery'
    for name in ('data','images'):(dest/name).mkdir(parents=True,exist_ok=True)
    wi={(r['key'],r['seed'],r['arm']):r for r in gzread(DEST/'evaluation/worst_by_image.json.gz') if r['scope']=='combined' and r['arm'] in ARMS}
    ext=gzread(DEST/'evaluation/extreme_cases.json.gz');top=sorted([e for e in ext if e['metric']=='FL' and e['arm'] in ('J','K','T','U') and e['relative'] is not None],key=lambda e:-e['relative'])[:70]
    catalog=[];views=0
    for idx,r in enumerate(ordered_inputs()):
        ident=r['dataset']+'_'+Path(r['image_id']).stem;im=Image.open(ROOT/r['image_path']).convert('RGB');im.thumbnail((1100,900));im.save(dest/'images'/f'{ident}.jpg',quality=88)
        row=dict(key=key(r),id=ident,dataset=r['dataset'],image_id=r['image_id'],width=r['width'],height=r['height'],image='images/'+ident+'.jpg',transforms=list(transforms(r)),seeds={})
        for seed in SEEDS:
            row['seeds'][seed]={}
            for tr in transforms(r):
                q=gzread(outpath(r,tr,seed));row['seeds'][seed][tr]={a:display(v) for a,v in q['arms'].items()};views+=8
            row['seeds'][seed]['worst']={a:wi[key(r),seed,a] for a in ARMS};row['unit']='mm' if q.get('px_per_mm') else 'px'
        (dest/'data'/f'{ident}.js').write_text('window.receiveImage('+json.dumps(clean(row),ensure_ascii=False,allow_nan=False,separators=(',',':'))+');',encoding='utf8')
        catalog.append({k:row[k] for k in ('key','id','dataset','image_id','image','transforms')})
        if (idx+1)%40==0:print('GALLERY',idx+1,views,flush=True)
    index=dict(catalog=catalog,groups=j(S01/'groups.json')['groups'],arms=LABEL,seeds=SEEDS,summary=[x for x in j(DEST/'evaluation/stability_summary.json') if x['scope']=='combined' and x['partition']=='all' and x['arm'] in ARMS],extremes=top)
    html=HTML.replace('Round38','Round40').replace('边界与肌束均重新预测','旧缓存完整复用；新增五种扰动对边界、肌束都补推理').replace('同扰动：R vs 所选组','同扰动：J vs 所选组').replace('相似组/极端开发病例','开发282图（全量）')
    html=html.replace('没有训练、没有更改默认流程。','没有训练、没有更改默认流程。138张开发锁参，331张本轮复核；仍属已观察过的数据，不是新患者盲测。')
    html=html.replace('<input id="rays" type="checkbox">','<input id="rays" type="checkbox" checked>')
    html=html.replace('预测时没有借用原图结果。','预测时没有借用原图结果。固定显示原图范围，远交点可能在画外；请同时查看长度数值和敏感性记录。')
    js=JS.replace("aa=mode==='within'?arm:'R'","aa=mode==='within'?arm:'J'").replace('六种轻微扰动','十一种轻微扰动').replace('current.transforms.slice(0,7)','current.transforms.filter(t=>![\'scale075\',\'scale125\',\'pad20\',\'translate1_-1\'].includes(t))').replace("+'/7有值）'","+'/12有值）'")
    js=js.replace("const options=", "Object.assign(T,{'translate-2_1':'左移2px、下移1px（新增）',scale0975:'缩放至97.5%（新增）',scale1025:'缩放至102.5%（新增）',pad15:'四周各加15%黑边（新增）',brightness105:'亮度增加5%（新增）'});\nconst options=")
    js+='\nconst baseTextPanel=textPanel;textPanel=function(el,a){baseTextPanel(el,a);if(a.decision)el.innerHTML+="<p class=warn>"+a.decision+"</p>";if(a.risk)el.innerHTML+="<details><summary>FL敏感性与实际贡献权重</summary><pre style=white-space:pre-wrap>"+JSON.stringify({weights:a.FL_weights,sensitivity:a.risk},null,2)+"</pre></details>"};if(current)render();\n'
    (dest/'index.html').write_text(html,encoding='utf8');(dest/'app.js').write_text(js,encoding='utf8');(dest/'index_data.js').write_text('window.INDEX='+json.dumps(index,ensure_ascii=False,allow_nan=False,separators=(',',':'))+';',encoding='utf8')
    write(DEST/'gallery_success.json',dict(images=len(catalog),views=views,groups=44,offline=True,browser_render_verified=False));print('GALLERY DONE',len(catalog),views,flush=True)
if __name__=='__main__':main()
