"""Isolated S01 diagnosis of frozen Round36. No training/default mutation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, gzip, hashlib, itertools, json, math, sys, time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scripts.fascicle_geometry import extract_segments
from scripts.round28_core import rejection_details, region, select
from scripts.round29_core import value_at
from scripts.round32_common import map_curve, map_segment, inverse_spec, spec, warp
from scripts.round36_core import measure, axis_delta
from src.umud_baseline.data import restore_original_coords

BASE = ROOT/'runs/s01_round36_stability_v1'
R36 = ROOT/'runs/round36_candidate_consensus_v1'
SEEDS = (20261023, 20261024)
DS = ('dev282', 'apo153', 'external34')
cv2.setNumThreads(2)

def j(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def key(r): return r['dataset']+'|'+r['image_id']
def ungzip(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def stats(x):
    a=np.asarray([v for v in x if v is not None],float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(n=0)
def binary(p):return np.asarray(Image.open(p).convert('L'))>127
def dice(a,b):
    den=a.sum()+b.sum()
    return float(2*np.logical_and(a,b).sum()/den) if den else 1.
def ssim(a,b):
    a=a.astype(np.float32);b=b.astype(np.float32);f=lambda x:cv2.GaussianBlur(x,(11,11),1.5)
    u,v=f(a),f(b);uu,vv,uv=u*u,v*v,u*v
    q=((2*uv+6.5025)*(2*(f(a*b)-uv)+58.5225))/((uu+vv+6.5025)*(f(a*a)-uu+f(b*b)-vv+58.5225))
    return float(q[5:-5,5:-5].mean())
def ncc(a,b):
    a=a.astype(float).ravel();b=b.astype(float).ravel();a-=a.mean();b-=b.mean()
    return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
def source_inputs():
    return [r for r in j(ROOT/'runs/round34_boundary_guidance_v1/c1_guidance/manifest.json')['rows'] if r['dataset']!='train772']
def pred_path(seed,ds):return R36/f'predictions/{seed}/{ds}.json.gz'
def load_rows(seed):
    out={}
    for ds in DS:
        for r in ungzip(pred_path(seed,ds)):
            r['arms']={a:r['arms'][a] for a in ('D0','D1','M')}
            r.pop('pool',None);r.pop('scores',None);out[key(r)]=r
    return out

def discover():
    """Image-only screening of all pairs; no model output or reference selects groups."""
    BASE.mkdir(exist_ok=True)
    cfg={'cohorts':{'dev282':282,'apo153':153,'external34':34},'seeds':SEEDS,
         'primary_arm':'M (P has identical measurements in Round36)',
         'crop_normalized':[.2,.15,.8,.85],'screen_size':128,'candidate_ncc':.70,
         'aspect_ratio_tolerance':.03,'max_translation_crop_pixels':5.,
         'similar_ssim':.80,'similar_ncc':.85,'strict_ssim':.90,'strict_ncc':.95,
         'registration':'translation only; select higher SSIM of zero/estimated shift; overlap >=90%',
         'component_matching':{'angle_degrees':15,'centroid_distance_image_diagonal':.06},
         'new_inference':'Frozen weights only: C1 identity all469, representative repeat/brightness, group-member soft transforms; T3 representative identity repeats for both seeds. No optimizer.',
         'causal_controls':'3-factor symmetric Shapley: boundary geometry, support metadata, filtered C1 segments; diagnostic hybrid inputs only',
         'training':False,'default_changed':False,'subagents':False}
    write(BASE/'config.json',cfg)
    rows=source_inputs();assert len(rows)==469
    rows=sorted(rows,key=lambda r:(DS.index(r['dataset']),r['image_id']))
    thumbs=[]
    for r in rows:
        im=np.asarray(Image.open(ROOT/r['image_path']).convert('L'));h,w=im.shape
        assert [w,h]==[r['width'],r['height']]
        assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256']
        thumbs.append(cv2.resize(im[int(h*.15):int(h*.85),int(w*.2):int(w*.8)],(128,128),interpolation=cv2.INTER_AREA))
    f=np.stack(thumbs).reshape(469,-1).astype(np.float32);f-=f.mean(1)[:,None];f/=np.maximum(np.linalg.norm(f,axis=1)[:,None],1e-8)
    corr=f@f.T;edges=[];tested=0;parent=list(range(len(rows)))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for i in range(len(rows)):
        for k in np.where(corr[i,i+1:]>=cfg['candidate_ncc'])[0]+i+1:
            ratio=(rows[i]['width']/rows[i]['height'])/(rows[k]['width']/rows[k]['height'])
            if abs(ratio-1)>.03:continue
            tested+=1;a,b=thumbs[i],thumbs[k];best=(ssim(a,b),float(corr[i,k]),0.,0.)
            try:
                _,w=cv2.findTransformECC(a.astype(np.float32)/255,b.astype(np.float32)/255,np.eye(2,3,dtype=np.float32),cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-5))
                dx,dy=map(float,w[:,2])
                if max(abs(dx),abs(dy))<=5:
                    bb=cv2.warpAffine(b,w,(128,128),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
                    good=cv2.warpAffine(np.ones_like(b),w,(128,128),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)>0
                    ys,xs=np.nonzero(good);sl=np.s_[ys.min():ys.max()+1,xs.min():xs.max()+1]
                    score=ssim(a[sl],bb[sl]);cc=ncc(a[sl],bb[sl])
                    if good.mean()>=.9 and score>best[0]:best=(score,cc,dx,dy)
            except cv2.error:pass
            score,cc,dx,dy=best
            if score<.8 or cc<.85:continue
            # ECC maps template A to input B. B points map to A by subtracting shift.
            e={'a':key(rows[i]),'b':key(rows[k]),'ssim':score,'ncc':cc,'translation_normalized':[dx/128*.6,dy/128*.7],
               'strict':bool(score>=.9 and cc>=.95),'pixel_identical':rows[i]['pixel_sha256']==rows[k]['pixel_sha256'],
               'same_existing_group':rows[i]['group_id']==rows[k]['group_id']}
            edges.append(e);parent[find(k)]=find(i)
    groups=defaultdict(list);members={v for e in edges for v in (e['a'],e['b'])}
    for i,r in enumerate(rows):
        if key(r) in members:groups[find(i)].append(key(r))
    gs=[{'id':f'SG{i+1:03d}','members':sorted(v),'representative':sorted(v)[0]} for i,v in enumerate(groups.values())]
    groupof={m:g['id'] for g in gs for m in g['members']}
    for e in edges:e['group']=groupof[e['a']]
    write(BASE/'groups.json',{'screened_images':469,'all_possible_pairs':469*468//2,'registration_candidates':tested,'rows':rows,'groups':gs,'pairs':edges,
       'limitations':['Appearance groups are not confirmed subject/video identities.','Only qualifying edges are compared; transitive group membership does not imply every pair is similar.','Central image crop is image-only; visually review representative and worst pairs.','Pairs failing fixed appearance/aspect screening are excluded, not claimed dissimilar anatomically.']})
    files=[ROOT/'configs/round36_candidate_consensus.json',ROOT/'runs/round34_boundary_guidance_v1/c1_guidance/manifest.json',R36/'binding.json']+[pred_path(s,d) for s in SEEDS for d in DS]
    files+=list((ROOT/'src').rglob('*.py'))+[p for p in (ROOT/'scripts').glob('*.py') if not p.name.startswith('s01_')]
    write(BASE/'protected_sources.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    print('DISCOVER',len(gs),'groups',len(members),'images',len(edges),'pairs',sum(e['strict'] for e in edges),'strict',flush=True)

def infer_c1():
    import torch
    from scripts.round32_infer import infer_array
    from src.umud_baseline.direction_model import DirectionAuxUNet
    from src.umud_baseline.data import read_image_gray3
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cp=ROOT/'runs/round26_1_direction_aux_corrected_v1/training/C1_corrected/checkpoints/epoch_017.pt'
    ck=torch.load(cp,map_location='cpu',weights_only=False);model=DirectionAuxUNet(pretrained=False).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    g=j(BASE/'groups.json');members={m for x in g['groups'] for m in x['members']};representatives={x['representative'] for x in g['groups']}
    out=[];start=time.time();forwards=0
    for i,r in enumerate(g['rows']):
        image=read_image_gray3(ROOT/r['image_path']);prob,meta,masks=infer_array(model,image,1024,False);forwards+=1
        old=binary(ROOT/r['mask_path']);assert sha(ROOT/r['mask_path'])==r['mask_sha256']
        rec={'key':key(r),'mask_changed_pixels':int((masks[0]!=old).sum())}
        if key(r) in members:
            p=BASE/f"c1_probability/{r['dataset']}/{Path(r['image_id']).stem}.npz";p.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(p,prob=prob,meta=json.dumps(meta));rec['prob_path']=p.relative_to(BASE).as_posix();rec['prob_sha256']=sha(p)
        if key(r) in representatives:
            repeat,_,_=infer_array(model,image,1024,False);forwards+=1
            rec['repeat_max_abs_probability_delta']=float(abs(prob-repeat).max())
            bright=np.rint(image.astype(float)*.95).clip(0,255).astype(np.uint8)
            bp,bm,bmask=infer_array(model,bright,1024,False);forwards+=1
            a=extract_segments(old);b=extract_segments(bmask[0]);p=BASE/f"brightness/{r['dataset']}/{Path(r['image_id']).stem}.npz";p.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(p,mask=bmask[0])
            rec['brightness']={'factor':.95,'mask_dice':dice(old,bmask[0]),'probability_mae':float(abs(bp-prob).mean()),'path':p.relative_to(BASE).as_posix(),
                               'before_evaluable':len(a['segments']),'after_evaluable':len(b['segments'])}
        out.append(rec)
        if (i+1)%25==0:print('C1',i+1,469,'seconds',round(time.time()-start),flush=True)
    write(BASE/'c1_inference.json',{'rows':out,'checkpoint_sha256':sha(cp),'forwards':forwards,'seconds':time.time()-start,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
          'changed_images':sum(r['mask_changed_pixels']>0 for r in out),'changed_pixels':sum(r['mask_changed_pixels'] for r in out),'training':False})
    assert not any(r['mask_changed_pixels'] for r in out),'C1 cache parity failed; do not interpret re-inferred probability as frozen source'
    print('C1 PARITY PASS',forwards,flush=True)

def mapping(a,b,e):
    dx,dy=e['translation_normalized']
    return dict(sx=a['width']/b['width'],sy=a['height']/b['height'],tx=-dx*a['width'],ty=-dy*a['height'])
def match_components(a,b,w,h):
    if not a or not b:return []
    diag=math.hypot(w,h);cost=np.full((len(a),len(b)),1e4)
    for i,s in enumerate(a):
        for k,t in enumerate(b):
            dist=float(np.linalg.norm(np.subtract(s['centroid'],t['centroid'])))/diag;ang=abs(float(axis_delta(s['direction_deg'],t['direction_deg'])))
            if dist<=.06 and ang<=15:cost[i,k]=dist/.06+ang/15+.15*abs(math.log(max(s['length_px'],1)/max(t['length_px'],1)))
    ii,kk=linear_sum_assignment(cost)
    return [(a[i],b[k]) for i,k in zip(ii,kk) if cost[i,k]<1e3]
def mix_curve(geom,sup):
    if geom is None or sup is None:return None
    c=copy.deepcopy(geom)
    for part in (None,'sup','deep'):
        x=c if part is None else c[part];y=sup if part is None else sup[part]
        for k in ('support','paired_x','max_sample_gap'):x[k]=copy.deepcopy(y[k])
    return c
def shapley(ca,cb,sa,sb,row):
    if ca is None or cb is None:return None
    vals={}
    for bits in itertools.product((0,1),repeat=3):
        curve=mix_curve((ca,cb)[bits[0]],(ca,cb)[bits[1]])
        vals[bits]=measure(curve,(sa,sb)[bits[2]],row,None)[0]['values']
    out={}
    for metric in ('MT','PA','FL'):
        if any(v[metric] is None for v in vals.values()):out[metric]=None;continue
        parts=[]
        for factor in range(3):
            total=0.
            for bits in itertools.product((0,1),repeat=3):
                if bits[factor]:continue
                n=sum(bits);weight=math.factorial(n)*math.factorial(2-n)/6;other=list(bits);other[factor]=1
                total+=weight*(vals[tuple(other)][metric]-vals[bits][metric])
            parts.append(total)
        delta=vals[(1,1,1)][metric]-vals[(0,0,0)][metric]
        assert abs(sum(parts)-delta)<1e-6
        out[metric]={'geometry':parts[0],'support':parts[1],'C1_segments':parts[2],'delta':delta}
    return out

@lru_cache(maxsize=8)
def original_probability(path):
    with np.load(path,allow_pickle=False) as z:
        p=z['prob'];m=json.loads(str(z['meta']))
    return np.stack([restore_original_coords(x,m['original_wh'],m['scale'],m['pad'],'nearest') for x in p])
def compare_prob(pa,pb,t,w,h):
    mat=np.array([[t['sx'],0,t['tx']],[0,t['sy'],t['ty']]],np.float32)
    b=np.stack([cv2.warpAffine(x,mat,(w,h),flags=cv2.INTER_NEAREST) for x in pb])
    valid=cv2.warpAffine(np.ones(pb.shape[1:],np.uint8),mat,(w,h),flags=cv2.INTER_NEAREST)>0
    roi=np.zeros_like(valid);roi[int(.15*h):int(.85*h),int(.2*w):int(.8*w)]=True;valid &= roi
    out=[]
    for x,y in zip(pa,b):
        foreground=((x>.1)|(y>.1))&valid
        out.append({'foreground_probability_mae':float(abs(x-y)[foreground].mean()) if foreground.any() else 0.,
                    'mask_dice':dice((x>.5)&valid,(y>.5)&valid),'foreground_pixels':int(foreground.sum())})
    return out

def pair_audit():
    g=j(BASE/'groups.json');ins={key(r):r for r in g['rows']};inf=j(BASE/'c1_inference.json');assert inf['changed_pixels']==0
    ip={r['key']:r for r in inf['rows']};ex={}
    for k in {m for x in g['groups'] for m in x['members']}:
        r=ins[k];q=extract_segments(binary(ROOT/r['mask_path']));ex[k]=q
    rows=[];start=time.time();replay_count=0
    for seed in SEEDS:
        rr=load_rows(seed)
        pm=j(ROOT/f'runs/round35_boundary_continuity_v1/predictions/{seed}/T3/manifest.json')
        pp={key(r):ROOT/f'runs/round35_boundary_continuity_v1/predictions/{seed}/T3'/r['path'] for r in pm['rows'] if r['transform']['name']=='identity'}
        for i,e in enumerate(g['pairs']):
            ka,kb=e['a'],e['b'];a,b=rr[ka],rr[kb];w,h=a['width'],a['height'];t=mapping(a,b,e)
            sa=ex[ka]['segments'];sb=[map_segment(s,t) for s in ex[kb]['segments']]
            aa=sa+ex[ka]['rejected'];bb=[map_segment(s,t) for s in ex[kb]['segments']+ex[kb]['rejected']]
            matches=match_components(aa,bb,w,h);idmap={s['label_id']:v['label_id'] for s,v in matches}
            flips=[{'a_id':s['label_id'],'b_id':v['label_id'],'a_category':s['category'],'b_category':v['category'],'a_flags':rejection_details(s),'b_flags_original_pixels':rejection_details(next(x for x in ex[kb]['segments']+ex[kb]['rejected'] if x['label_id']==v['label_id']))} for s,v in matches if (s['category']=='evaluable')!=(v['category']=='evaluable')]
            rec={**e,'seed':seed,'component_counts':[a['component_counts'],b['component_counts']], 'matched_components':len(matches),'filter_status_changes':flips,
                 'C1_probability':compare_prob(original_probability(BASE/ip[ka]['prob_path']),original_probability(BASE/ip[kb]['prob_path']),t,w,h)[0],
                 'boundary_probability':compare_prob(original_probability(pp[ka]),original_probability(pp[kb]),t,w,h),'arms':{}}
            for arm in ('D0','D1','M'):
                av,bv=a['arms'][arm],b['arms'][arm];ca=av['curve'];cb=map_curve(bv['curve'],t)
                # Recompute in A's pixel coordinates. Native mm values are not compared across images.
                context={'width':w,'height':h,'px_per_mm':None}
                ma,csa=measure(ca,sa,context,None);mb,csb=measure(cb,sb,context,None)
                factor=a.get('px_per_mm') or 1
                for p in ('MT','PA','FL'):
                    expected=av['values'][p];expected=expected if p=='PA' or expected is None else expected*factor
                    assert (expected is None)==(ma['values'][p] is None)
                    if expected is not None:assert abs(expected-ma['values'][p])<1e-6
                replay_count+=1
                den={'MT':h,'PA':1.,'FL':w};dif={p:(mb['values'][p]-ma['values'][p])/den[p] if ma['values'][p] is not None and mb['values'][p] is not None else None for p in den}
                ar={'A_values_px':ma['values'],'B_registered_values_px':mb['values'],'delta_normalized':dif,
                    'coverage_changed':{p:(ma['values'][p] is None)!=(mb['values'][p] is None) for p in den},'boundary_missing_changed':(ca is None)!=(cb is None)}
                if ca and cb:
                    lo=max(ca['support'][0],cb['support'][0]);hi=min(ca['support'][1],cb['support'][1])
                    ar['support_endpoint_delta_width']=max(abs(x-y) for x,y in zip(ca['support'],cb['support']))/w
                    ar['support_A']=ca['support'];ar['support_B_registered']=cb['support']
                    ar['curve_mean_delta_height']=float(np.mean([abs(value_at(ca[s],x)-value_at(cb[s],x))/h for s in ('sup','deep') for x in np.linspace(lo,hi,33)])) if hi>lo else None
                    ar['extension_slope_max_delta_degrees']=max(abs(math.degrees(math.atan(ca[s][r][0])-math.atan(cb[s][r][0]))) for s in ('sup','deep') for r in ('left_ray','right_ray'))
                a_ids=ma['selection']['selected_ids'];b_ids=mb['selection']['selected_ids'];mapped=[idmap.get(x) for x in a_ids]
                ar['selected_A']=a_ids;ar['selected_B']=b_ids;ar['matched_A_to_B']=mapped
                ar['selection_changed']=set(mapped)!=set(b_ids) or None in mapped
                ar['selected_tracks_unmatched']=sum(x is None for x in mapped)
                lookup={c['label_id']:c for c in csb};held=None
                if mapped and None not in mapped and all(x in lookup and lookup[x]['F0']['fl_px_raw'] is not None for x in mapped):
                    held=sum(wt*lookup[x]['F0']['fl_px_raw'] for wt,x in zip(ma['selection']['weights'],mapped))
                ar['B_FL_with_matched_A_selection']=held
                ar['selection_effect_FL_width']=(mb['values']['FL']-held)/w if held is not None and mb['values']['FL'] is not None else None
                if arm=='M':ar['factor_decomposition']=shapley(ca,cb,sa,sb,context)
                rec['arms'][arm]=ar
            rows.append(rec)
            if (i+1)%15==0:print('PAIRS',seed,i+1,len(g['pairs']),'seconds',round(time.time()-start),flush=True)
        del rr
    write(BASE/'pairs.json',{'rows':rows,'measurement_replay_checks':replay_count,'groups_sha256':sha(BASE/'groups.json'),
          'normalization':'MT/image height; FL/image width; PA degrees. B registered into A pixel coordinates.',
          'factor_order':['geometry','support','C1_segments'],'decomposition_is_diagnostic_not_ground_truth_causality':True})
    print('PAIR AUDIT COMPLETE',len(rows),flush=True)

def controls():
    """Reuse all frozen C1 transform masks; diagnose original-pixel filtering separately."""
    g=j(BASE/'groups.json');ins={key(r):r for r in g['rows']};members={m for x in g['groups'] for m in x['members']}
    src=ROOT/'runs/round32_chain_foundation_v1/inference/fasc_transformed';manifest=j(src/'manifest.json')
    old={};original_ex={};rec=[];states={};start=time.time()
    for k,r in ins.items():old[k]=binary(ROOT/r['mask_path']);original_ex[k]=extract_segments(old[k])
    for i,p in enumerate(manifest['rows']):
        k=key(p);r=ins[k];t=p['transform'];path=src/p['path'];assert sha(path)==p['sha256']
        with np.load(path,allow_pickle=False) as z:m=z['mask']
        frozen=warp(old[k],t,True);a=original_ex[k];b=extract_segments(m);f=extract_segments(frozen)
        inv=inverse_spec(t);mat=np.array([[inv['sx'],0,inv['tx']],[0,inv['sy'],inv['ty']]],np.float32)
        back=cv2.warpAffine(m.astype(np.uint8),mat,(r['width'],r['height']),flags=cv2.INTER_NEAREST)>0
        c=extract_segments(back)
        d={'key':k,'transform':t['name'],'original_evaluable':len(a['segments']),'network_evaluable_native':len(b['segments']),
           'frozen_mask_evaluable_native':len(f['segments']),'network_evaluable_after_map_to_original':len(c['segments']),
           'network_vs_frozen_mask_dice_transformed':dice(m,frozen),'network_mask_dice_original':dice(back,old[k]),
           'network_reject_flags':dict(Counter(x for s in b['rejected'] for x in rejection_details(s))),
           'frozen_reject_flags':dict(Counter(x for s in f['rejected'] for x in rejection_details(s)))}
        rec.append(d)
        if k in members:states[(k,t['name'])]={'network_native':[map_segment(s,inv) for s in b['segments']],
            'frozen_native':[map_segment(s,inv) for s in f['segments']],'network_original_grid':c['segments']}
        if (i+1)%150==0:print('C1 CONTROLS',i+1,1407,'seconds',round(time.time()-start),flush=True)
    measures=[]
    for seed in SEEDS:
        rr=load_rows(seed)
        for i,((k,tr),ss) in enumerate(states.items()):
            r=rr[k];cv=r['arms']['M']['curve'];context={'width':r['width'],'height':r['height'],'px_per_mm':None}
            before=measure(cv,original_ex[k]['segments'],context,None)[0]
            for method,segs in ss.items():
                after=measure(cv,segs,context,None)[0]
                factors={'MT':r['height'],'PA':1.,'FL':r['width']}
                measures.append({'key':k,'seed':seed,'transform':tr,'method':method,'before':before['values'],'after':after['values'],
                   'delta_normalized':{p:(after['values'][p]-before['values'][p])/factors[p] if after['values'][p] is not None and before['values'][p] is not None else None for p in factors},
                   'lost_FL':before['values']['FL'] is not None and after['values']['FL'] is None,'new_FL':before['values']['FL'] is None and after['values']['FL'] is not None})
            if (i+1)%70==0:print('MEASURE C1 CONTROL',seed,i+1,len(states),flush=True)
        del rr
    write(BASE/'c1_controls.json',{'component_rows':rec,'measurement_rows':measures,'source_manifest_sha256':sha(src/'manifest.json'),
          'scope':'All469 images x3 transforms for masks/filtering; all113 similarity-group members x2 seeds for fixed-M-boundary measurements.',
          'frozen_native':'No C1 forward; resize/pad original mask then apply existing native-pixel component gates, inverse-map segments.',
          'network_native':'Previously cached same C1 forward; extract at transformed resolution, inverse-map segments.',
          'network_original_grid':'Same transformed network mask mapped to original grid before component extraction. Diagnostic only, not promoted.',
          'boundary_fixed':True,'source_current_C1_parity_required':True})
    # Full-cohort boundary-network versus frozen-probability comparisons already use
    # exactly Round36 D0/D1 baselines, as verified by Round36's 1876 exact replays.
    src=ROOT/'runs/round35_boundary_continuity_v1/stability/summary.json';d=j(src);out=[]
    for r in d['rows']:
        if r['training']!='T3' or r['decoder'] not in ('D0','D1'):continue
        v={k:r[k] for k in ('seed','dataset','mode','transform','decoder','n','FL_selected_set_changed')}
        v['measurement']={p:{k:z for k,z in q.items() if k not in ('common_ids','new_ids','lost_ids','both_missing_ids')}|{'new_n':len(q['new_ids']),'lost_n':len(q['lost_ids'])} for p,q in r['measurement'].items()}
        out.append(v)
    write(BASE/'boundary_controls.json',{'rows':out,'source_sha256':sha(src),'C1_fixed':True,'source_round':'Round35 T3, numerically identical to Round36 D0/D1 baselines',
       'M_not_claimed_replayed_under_transforms':True,'raw_transformed_probability_archives_were_cleaned_but_signed_measurement_summaries_retained':True})
    print('CONTROLS COMPLETE',len(rec),len(measures),flush=True)

def repeats():
    import torch
    from scripts.round35_infer import infer
    from scripts.round34_model import GuidedBoundaryUNet
    from src.umud_baseline.data import read_image_gray3
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    g=j(BASE/'groups.json');ins={key(r):r for r in g['rows']};reps=[ins[x['representative']] for x in g['groups']];out=[]
    for seed in SEEDS:
        cp=ROOT/f'runs/round35_boundary_continuity_v1/training/{seed}/T3/checkpoints/epoch_017_model.pt'
        src=ROOT/f'runs/round35_boundary_continuity_v1/predictions/{seed}/T3';pm=j(src/'manifest.json');assert sha(cp)==pm['binding']['checkpoint_sha256']
        pp={key(r):r for r in pm['rows'] if r['transform']['name']=='identity'}
        ck=torch.load(cp,map_location='cpu',weights_only=False);model=GuidedBoundaryUNet().cuda().eval();model.load_state_dict(ck['model'],strict=True)
        for r in reps:
            im=read_image_gray3(ROOT/r['image_path']);hint=binary(ROOT/r['guide_path'])
            a,_,am=infer(model,im,hint,False);b,_,_=infer(model,im,hint,False)
            p=src/pp[key(r)]['path'];assert sha(p)==pp[key(r)]['sha256']
            with np.load(p,allow_pickle=False) as z:old=z['prob'];oldm=z['masks']
            out.append({'key':key(r),'seed':seed,'repeat_max_abs_delta':float(abs(a-b).max()),'cache_max_abs_delta':float(abs(a-old).max()),'cache_mask_changed_pixels':int((am!=oldm).sum())})
        del model;torch.cuda.empty_cache();print('T3 REPEAT',seed,len(reps),flush=True)
    write(BASE/'boundary_repeat.json',{'rows':out,'actual_forwards':len(out)*2,'training':False,'hint_used':False,
         'reason_no_hint':'Exact Round35 T3 source inference explicitly passes use_hint=False; C1 affects downstream selection, not this forward.'})

def seed_audit():
    aa,bb=[load_rows(seed) for seed in SEEDS];out=[]
    for i,k in enumerate(aa):
        a,b=aa[k],bb[k];assert a['input_sha']['C1']==b['input_sha']['C1']
        sa=[c['segment'] for c in a['arms']['M']['candidates']];sb=[c['segment'] for c in b['arms']['M']['candidates']];assert sa==sb
        ca,cb=a['arms']['M']['curve'],b['arms']['M']['curve'];r={'width':a['width'],'height':a['height'],'px_per_mm':None}
        vals={}
        for geom,sup in itertools.product((0,1),repeat=2):vals[(geom,sup)]=measure(mix_curve((ca,cb)[geom],(ca,cb)[sup]),sa,r,None)[0]['values']
        parts={}
        for metric,den in [('MT',a['height']),('PA',1),('FL',a['width'])]:
            if any(q[metric] is None for q in vals.values()):parts[metric]=None;continue
            v00,v10,v01,v11=[vals[t][metric] for t in [(0,0),(1,0),(0,1),(1,1)]]
            parts[metric]={'geometry':(.5*(v10-v00+v11-v01))/den,'support':(.5*(v01-v00+v11-v10))/den,'delta':(v11-v00)/den}
        av,bv=a['arms']['M'],b['arms']['M'];lookup={s['segment']['label_id']:s['raw_FL'] for s in bv['candidates']};held=None
        ids=av['selection']['selected_ids'];weights=av['selection']['weights']
        if ids and all(lookup.get(x) is not None for x in ids):held=sum(w*lookup[x] for w,x in zip(weights,ids))
        bfl=vals[(1,1)]['FL']
        out.append({'key':k,'seed_A':SEEDS[0],'seed_B':SEEDS[1],'C1_exactly_fixed':True,'factors':parts,
           'selected_ids_changed':av['selection']['selected_ids']!=bv['selection']['selected_ids'],
           'selection_effect_FL_width':(bfl-held)/a['width'] if bfl is not None and held is not None else None,
           'coverage_changed':{p:(av['values'][p] is None)!=(bv['values'][p] is None) for p in ('MT','PA','FL')}})
        if (i+1)%100==0:print('SEED SENSITIVITY',i+1,469,flush=True)
    write(BASE/'seed_sensitivity.json',{'rows':out,'all_469':True,'interpretation':'Boundary-model seed sensitivity on identical inputs/C1. Not stochastic repeat inference.'})

def gate_audit():
    g=j(BASE/'groups.json');ins={key(r):r for r in g['rows']};members={m for x in g['groups'] for m in x['members']}
    ex={k:extract_segments(binary(ROOT/ins[k]['mask_path'])) for k in members};out=[];bright=[]
    inference={r['key']:r for r in j(BASE/'c1_inference.json')['rows']}
    for seed in SEEDS:
        rr=load_rows(seed)
        for e in g['pairs']:
            a,b=rr[e['a']],rr[e['b']];t=mapping(a,b,e);ae=ex[e['a']];be=ex[e['b']]
            allb=[map_segment(s,t) for s in be['segments']+be['rejected']]
            matches=match_components(ae['segments']+ae['rejected'],allb,a['width'],a['height'])
            actual={s['label_id']:s for s in allb if s['category']=='evaluable'};counter=dict(actual);changes=[]
            for s,v in matches:
                if (s['category']=='evaluable')==(v['category']=='evaluable'):continue
                changes.append([s['label_id'],v['label_id'],s['category'],v['category']])
                if s['category']=='evaluable':counter[v['label_id']]=v
                else:counter.pop(v['label_id'],None)
            context={'width':a['width'],'height':a['height'],'px_per_mm':None};cv=map_curve(b['arms']['M']['curve'],t)
            actual_measure,_=measure(cv,list(actual.values()),context,None);cf,_=measure(cv,list(counter.values()),context,None)
            v0,v1=actual_measure['values'],cf['values']
            out.append({'a':e['a'],'b':e['b'],'seed':seed,'group':e['group'],'strict':e['strict'],'pixel_identical':e['pixel_identical'],
                'changed_filter_status':changes,'actual_B':v0,'B_with_matched_A_acceptance':v1,
                'filter_switch_effect_FL_width':(v0['FL']-v1['FL'])/a['width'] if v0['FL'] is not None and v1['FL'] is not None else None,
                'filter_switch_effect_PA':v0['PA']-v1['PA'] if v0['PA'] is not None and v1['PA'] is not None else None})
        for group in g['groups']:
            k=group['representative'];r=rr[k];context={'width':r['width'],'height':r['height'],'px_per_mm':None};cv=r['arms']['M']['curve']
            meta=inference[k]['brightness']
            with np.load(BASE/meta['path'],allow_pickle=False) as z:q=extract_segments(z['mask'])
            ma,_=measure(cv,ex[k]['segments'],context,None);mb,_=measure(cv,q['segments'],context,None)
            bright.append({'key':k,'seed':seed,'group':group['id'],'mask_dice':meta['mask_dice'],'before':ma['values'],'after':mb['values'],
                'before_count':len(ex[k]['segments']),'after_count':len(q['segments']),
                'FL_delta_width':(mb['values']['FL']-ma['values']['FL'])/r['width'] if ma['values']['FL'] is not None and mb['values']['FL'] is not None else None,
                'PA_delta':mb['values']['PA']-ma['values']['PA'] if ma['values']['PA'] is not None and mb['values']['PA'] is not None else None,
                'lost_FL':ma['values']['FL'] is not None and mb['values']['FL'] is None})
        del rr;print('GATE AUDIT',seed,flush=True)
    write(BASE/'filter_intervention.json',{'rows':out,'caveat':'Matched-component acceptance status only is transplanted A to B; B component geometry and boundary stay fixed. Diagnostic, not a proposed prediction or proof of anatomical correctness. Includes downstream reselection; not additive to the three-factor decomposition.'})
    write(BASE/'brightness_measurements.json',{'rows':bright,'C1_input_brightness_factor':.95,'boundary_fixed':True,'group_representatives':44})

def soft_probes():
    """Separate bitmap resampling from pure pixel-gate scaling; inspect soft responses."""
    import torch
    from scripts.round32_infer import infer_array
    from src.umud_baseline.direction_model import DirectionAuxUNet
    from src.umud_baseline.data import read_image_gray3
    g=j(BASE/'groups.json');members={m for x in g['groups'] for m in x['members']};pure=[];originals={}
    for r in g['rows']:
        ex=extract_segments(binary(ROOT/r['mask_path']));originals[key(r)]=ex
        for name in ('scale075','scale125','pad10'):
            t=spec(r['width'],r['height'],name);after=[]
            for s in ex['segments']+ex['rejected']:
                v=map_segment(s,t);v['elongation']=v['length_px']/max(v['width_px'],1.)
                after.append((s,v,not rejection_details(v)))
            changes=[{'id':s['label_id'],'before':s['category']=='evaluable','after':keep,'before_flags':rejection_details(s),'after_flags':rejection_details(v)} for s,v,keep in after if (s['category']=='evaluable')!=keep]
            pure.append({'key':key(r),'transform':name,'before':len(ex['segments']),'after':sum(keep for _,_,keep in after),'changes':changes})
    write(BASE/'analytic_gate_scaling.json',{'rows':pure,'no_network_no_bitmap_resampling':True,'purpose':'Scale the same component geometry and apply existing absolute pixel thresholds; isolates hard gates from raster topology changes.'})
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cp=ROOT/'runs/round26_1_direction_aux_corrected_v1/training/C1_corrected/checkpoints/epoch_017.pt'
    ck=torch.load(cp,map_location='cpu',weights_only=False);model=DirectionAuxUNet(pretrained=False).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    src=ROOT/'runs/round32_chain_foundation_v1/inference/fasc_transformed';old={ (key(p),p['transform']['name']):p for p in j(src/'manifest.json')['rows']}
    rows=[];translated={};forwards=0
    def restore_prob(prob,meta):return restore_original_coords(prob[0],meta['original_wh'],meta['scale'],meta['pad'],'nearest')
    selected=[r for r in g['rows'] if key(r) in members]
    for i,r in enumerate(selected):
        k=key(r);w,h=r['width'],r['height'];im=read_image_gray3(ROOT/r['image_path']);p,m,mask=infer_array(model,im,1024,False);forwards+=1
        assert np.array_equal(mask[0],binary(ROOT/r['mask_path']));baseline=restore_prob(p,m)
        for name in ('scale075','scale125','pad10','translate2_3'):
            if name=='translate2_3':
                t=dict(sx=1.,sy=1.,tx=2.,ty=3.,width=w,height=h,name=name)
                image=cv2.warpAffine(im,np.array([[1,0,2],[0,1,3]],np.float32),(w,h),flags=cv2.INTER_NEAREST)
            else:t=spec(w,h,name);image=warp(im,t)
            pp,mm,ms=infer_array(model,image,1024,False);forwards+=1
            cached_delta=None
            if name!='translate2_3':
                with np.load(src/old[(k,name)]['path'],allow_pickle=False) as z:cached_delta=int((ms[0]!=z['mask']).sum())
                assert cached_delta==0,'C1 transform cache mismatch'
            transformed=restore_prob(pp,mm);inv=inverse_spec(t);mat=np.array([[inv['sx'],0,inv['tx']],[0,inv['sy'],inv['ty']]],np.float32)
            back=cv2.warpAffine(transformed,mat,(w,h),flags=cv2.INTER_LINEAR)
            valid=cv2.warpAffine(np.ones(transformed.shape,np.uint8),mat,(w,h),flags=cv2.INTER_NEAREST)>0
            # Exclude the boundary of the inverse-warped field; keep all tissue areas.
            valid=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))>0
            fg=((baseline>.1)|(back>.1))&valid
            rows.append({'key':k,'transform':name,'foreground_probability_mae':float(abs(baseline-back)[fg].mean()) if fg.any() else 0.,
              'mask_dice':dice((baseline>.5)&valid,(back>.5)&valid),'cached_mask_changed_pixels':cached_delta,
              'foreground_pixels':int(fg.sum()),'near_threshold_fraction':float((((back>.4)&(back<.6))&fg).sum()/max(fg.sum(),1))})
            if name=='translate2_3':
                original_mask=cv2.warpAffine(ms[0].astype(np.uint8),mat,(w,h),flags=cv2.INTER_NEAREST)>0
                translated[k]=extract_segments(original_mask)['segments']
        if (i+1)%20==0:print('SOFT PROBES',i+1,len(selected),flush=True)
    measures=[]
    del model;torch.cuda.empty_cache()
    for seed in SEEDS:
        rr=load_rows(seed)
        for k,segs in translated.items():
            r=rr[k];context={'width':r['width'],'height':r['height'],'px_per_mm':None};cv=r['arms']['M']['curve']
            ma,_=measure(cv,originals[k]['segments'],context,None);mb,_=measure(cv,segs,context,None)
            measures.append({'key':k,'seed':seed,'before':ma['values'],'after':mb['values'],
                'FL_delta_width':(mb['values']['FL']-ma['values']['FL'])/r['width'] if ma['values']['FL'] is not None and mb['values']['FL'] is not None else None,
                'PA_delta':mb['values']['PA']-ma['values']['PA'] if ma['values']['PA'] is not None and mb['values']['PA'] is not None else None,
                'lost_FL':ma['values']['FL'] is not None and mb['values']['FL'] is None})
        del rr
    write(BASE/'soft_probes.json',{'rows':rows,'translation_measurements':measures,'actual_new_forwards':forwards,'cache_parity_all_exact':True,
           'scope':'All113 similarity-group images; exact same C1 weights; probabilities inverse-mapped before comparison, common valid field only.',
           'translation':'2px right,3px down; inverse-map to original; M boundary fixed for measurement.',
           'training':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['discover','infer','pairs','controls','repeats','seeds','gates','soft']);a=p.parse_args()
    {'discover':discover,'infer':infer_c1,'pairs':pair_audit,'controls':controls,'repeats':repeats,'seeds':seed_audit,'gates':gate_audit,'soft':soft_probes}[a.stage]()
