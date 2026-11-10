"""Isolated, resumable Round38 stability experiment. No training/default writes."""
import copy, gzip, hashlib, json, os, sys
from functools import lru_cache
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from PIL import Image
from scripts.round37_io import ROOT, S01, CP, j, sha, clean, binary, inputs, key, gzread
from scripts.round32_common import inverse_spec, map_point, map_segment
BASE=ROOT/'runs/round38_stability_first_v1'
R37=ROOT/'runs/round37_stability_repair_v1'
R36=ROOT/'runs/round36_candidate_consensus_v1'
R35=ROOT/'runs/round35_boundary_continuity_v1'
CFG=ROOT/'configs/round38_stability_first.json'
SEEDS=(20261023,20261024)
ARMS=('R','G','N','V','A','NV','NVA','BNVA')
DS=('external34','apo153','dev282')
GENTLE=('translate2_3','translate-3_-2','scale095','scale105','pad10','brightness095')
STRESS=('scale075','scale125','pad20','translate1_-1')
VIEWS=((0,0),(2,0),(-2,0),(0,2),(0,-2))

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)

def gzwrite(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf8',compresslevel=3) as f:json.dump(clean(v),f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    os.replace(tmp,p)

def spec(w,h,name):
    sx=sy=1.;tx=ty=0.;nw,nh=w,h
    if name.startswith('scale'):
        scale=int(name[5:])/100;nw,nh=round(w*scale),round(h*scale);sx,sy=nw/w,nh/h;tx,ty=(sx-1)/2,(sy-1)/2
    elif name.startswith('pad'):
        d=int(name[3:])/100;tx,ty=round(w*d),round(h*d);nw,nh=w+2*tx,h+2*ty
    elif name.startswith('translate'):tx,ty=map(int,name[9:].split('_'))
    return dict(sx=sx,sy=sy,tx=float(tx),ty=float(ty),width=nw,height=nh,name=name)

def warp(im,t):
    name=t['name']
    if name=='identity':return im.copy()
    if name.startswith('brightness'):return np.rint(im.astype(float)*.95).clip(0,255).astype(np.uint8)
    if name.startswith('scale'):return np.asarray(Image.fromarray(im).resize((t['width'],t['height']),Image.Resampling.BILINEAR))
    return cv2.warpAffine(im,np.array([[1,0,t['tx']],[0,1,t['ty']]],np.float32),(t['width'],t['height']),flags=cv2.INTER_NEAREST)

def back(a,t,w,h,interpolation=cv2.INTER_NEAREST):
    iv=inverse_spec(t);return cv2.warpAffine(a,np.array([[iv['sx'],0,iv['tx']],[0,iv['sy'],iv['ty']]],np.float32),(w,h),flags=interpolation)

@lru_cache(maxsize=1)
def members():return {m for g in j(S01/'groups.json')['groups'] for m in g['members']}
def ordered_inputs():return sorted(inputs(),key=lambda r:(DS.index(r['dataset']),r['image_id']))
def transforms(r):return ('identity',)+GENTLE+(STRESS if key(r) in members() else ())
def cachepath(r,tr,kind):return BASE/f"inference/{kind}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz"
def predpath(r,tr,seed):return BASE/f"predictions/{seed}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
def checkpoint(seed):return R35/f'training/{seed}/T3/checkpoints/epoch_017_model.pt'

def packsave(path,masks,preview,record):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with open(tmp,'wb') as f:np.savez_compressed(f,packed=np.packbits(masks.ravel()),shape=masks.shape,preview=preview.astype(np.float16))
    os.replace(tmp,path);write(path.with_suffix('.json'),{**record,'sha256':sha(path)})

def unpack(path):
    with np.load(path) as z:return np.unpackbits(z['packed'],count=int(np.prod(z['shape']))).reshape(z['shape']).astype(bool)

def verify_code(stage):
    b=j(BASE/'binding.json');assert sha(CFG)==b['config_sha256']
    for p,h in b['protected'].items():assert sha(ROOT/p)==h,'protected source changed: '+p
    return b

def freeze():
    if (BASE/'binding.json').exists():return verify_code('resume')
    BASE.mkdir(parents=True,exist_ok=True)
    cfg=dict(version='round38_stability_first_v1',seeds=SEEDS,arms=ARMS,cohort=469,gentle=GENTLE,stress=STRESS,views=VIEWS,
        reference_filter=j(ROOT/'configs/round37_stability_repair.json'),
        eligibility=dict(inside_fraction=.5,support_fraction=.5,min_linearity=.9),
        aggregate=dict(region_weights=[.25,.5,.25],anchors=[1/6,.5,5/6],spatial_sigma=.22,
          length_cap_roi=.15,weight_floor=.05,huber_delta=1.5,scale_floor_FL_roi=.01,scale_floor_PA_deg=.5,
          quality_sensitivity_weight=2.,min_rep_spacing_roi=.1,iterations=30),
        targets=dict(MT_relative=.02,PA_deg=1.,FL_relative=.05,FL_ROI_absolute=.03,pass_fraction=.95,missing_flip_fraction=.01,severe_relative=.2,severe_PA_deg=3.),
        training=False,default_changed=False,subagents=False,kaggle_upload=False,
        boundary_hint='zero as frozen T3 training/inference; C1 selects band downstream',
        view_alignment='integer native prepared-image pixels; nearest frozen restore of soft probabilities; inverse-align then equal mean over valid views',
        aggregation='individual-track measurements, never average angles before intersection',
        measurement_coordinates='all gates, decoder and selection in perturbed image; selected measurements inverse-mapped for evaluation without reselecting',
        uncertainty='diagnostic dispersion, not calibrated confidence interval')
    write(CFG,cfg)
    paths=list((ROOT/'src').rglob('*.py'))+[p for p in (ROOT/'scripts').glob('*.py') if not p.name.startswith('round38_')]
    paths += [p for p in (ROOT/'configs').glob('*.json') if p!=CFG]
    paths += [S01/'groups.json',CP,*[checkpoint(s) for s in SEEDS],R37/'roi.json']
    paths += [R37/f'predictions/{s}/{d}.json.gz' for s in SEEDS for d in DS]
    write(BASE/'binding.json',dict(config_sha256=sha(CFG),protected={p.relative_to(ROOT).as_posix():sha(p) for p in paths},prediction_uses_references=False))
    print('FROZEN',len(inputs()),'variants',sum(len(transforms(r)) for r in inputs()),flush=True)

if __name__=='__main__':freeze()
