"""Frozen epoch017 prediction, original/scale/padding; no reference reads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.round35_common import *
from scripts.round34_model import GuidedBoundaryUNet,guide_tensor
from scripts.round35_common import transform as spec,warp
from src.umud_baseline.data import read_image_gray3,resize_long_side,restore_original_coords,IMAGENET_MEAN,IMAGENET_STD

@torch.no_grad()
def infer(model,image,guide,use_hint=True,device='cuda'):
    h,w=image.shape[:2];small,scale=resize_long_side(image,512,False);nh,nw=small.shape[:2]
    x=np.pad(small,((0,512-nh),(0,512-nw),(0,0)))
    x=torch.from_numpy(x.copy()).permute(2,0,1).float()/255
    x=(x-torch.tensor(IMAGENET_MEAN)[:,None,None])/torch.tensor(IMAGENET_STD)[:,None,None]
    hint=guide_tensor(guide,512) if use_hint else torch.zeros((1,512,512))
    with torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'):u,i=model(torch.cat([x,hint],0)[None].to(device))
    prob=torch.cat([u,i],1).float().sigmoid()[0].cpu().numpy().astype(np.float32)
    meta={'original_wh':[w,h],'scale':scale,'pad':[512-nw,512-nh],'storage':'float32','hint_area_interpolation':'cv2.INTER_AREA'}
    masks=np.stack([restore_original_coords(p,(w,h),scale,meta['pad'],'nearest')>.5 for p in prob])
    return prob,meta,masks

def main(seed,arm,zero=False):
    code_manifest=BASE/"inference_code_manifest.json"
    verify(j(code_manifest)["files"])
    name=arm;out=BASE/f'predictions/{seed}/{name}';out.mkdir(parents=True,exist_ok=True)
    (out/'success.json').unlink(missing_ok=True)
    training=R34/f'training/{seed}/A' if arm=='T0' else BASE/f'training/{seed}/{arm}'
    cp=training/'checkpoints/epoch_017_model.pt';success=j(training/'success.json')
    require(sha(cp)==success['model_sha256'],'checkpoint binding');gs=j(R34/'c1_guidance/success.json')
    require(sha(R34/'c1_guidance/manifest.json')==gs['manifest_sha256'],'guide manifest')
    rows=[r for r in j(R34/'c1_guidance/manifest.json')['rows'] if r['dataset']!='train772'];unique(rows,['dataset','image_id'])
    binding={'seed':seed,'arm':arm,'zero_hint':zero,'checkpoint_sha256':sha(cp),'config_sha256':sha(CFG),
        'inference_source_manifest_sha256':sha(code_manifest),'guidance_sha256':gs['manifest_sha256'],'code':{str(p.relative_to(ROOT).as_posix()):sha(p) for p in [Path(__file__),ROOT/'scripts/round34_model.py',ROOT/'src/umud_baseline/data.py']}}
    if (out/'binding.json').exists():require(j(out/'binding.json')==binding,'prediction resume binding')
    else:write(out/'binding.json',binding)
    ck=torch.load(cp,map_location='cpu',weights_only=False);require(ck['epoch']==17 and ck['arm']==('A' if arm=='T0' else arm) and ck['seed']==seed,'checkpoint role')
    expected_cfg=ROOT/'configs/round34_boundary_guidance.json' if arm=='T0' else CFG
    expected_package=R34/'package_manifest.sha256' if arm=='T0' else BASE/'package_manifest.sha256'
    require(ck['config_sha256']==sha(expected_cfg) and ck['package_sha256']==expected_package.read_text().strip(),'checkpoint training configuration')
    require(ck['shared_init_sha256']==sha(R30/f'training/{seed}/shared_init.pt'),'checkpoint initialization binding')
    require(torch.cuda.is_available(),'CUDA prediction');torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=GuidedBoundaryUNet().cuda().eval();model.load_state_dict(ck['model'],strict=True)
    transforms=j(code_manifest)['transforms_by_arm'][arm];recs=[];tic=time.time()
    for transform in transforms:
        for n,r in enumerate(rows):
            ph,wh=image_identity(ROOT/r['image_path']);require(ph==r['pixel_sha256'] and wh==[r['width'],r['height']],'image changed')
            require(sha(ROOT/r['guide_path'])==r['guide_sha256'],'guide changed')
            folder=out/r['dataset']/transform;folder.mkdir(parents=True,exist_ok=True);dest=folder/(Path(r['image_id']).stem+'.npz');sidecar=dest.with_suffix('.json')
            if sidecar.exists():
                rec=j(sidecar);require(rec['binding']==binding and sha(dest)==rec['sha256'],'partial prediction changed');recs.append(rec);continue
            image=read_image_gray3(ROOT/r['image_path']);hint=binary(ROOT/r['guide_path']);t=spec(r['width'],r['height'],transform)
            image=warp(image,t,False);hint=warp(hint,t,True)
            prob,meta,masks=infer(model,image,hint,False)
            np.savez_compressed(dest,prob=prob,meta=json.dumps(meta),masks=masks)
            rec={k:r[k] for k in ('dataset','image_id','width','height','image_path','pixel_sha256','group_id')}
            rec.update(transform=t,path=dest.relative_to(out).as_posix(),sha256=sha(dest),binding=binding)
            write(sidecar,rec);recs.append(rec)
            if (n+1)%75==0:print('predict',seed,name,transform,n+1,'seconds',round(time.time()-tic),flush=True)
    require(len(recs)==469*len(transforms),'full prediction denominator');unique(recs,['dataset','image_id','path'])
    write(out/'manifest.json',{'rows':recs,'binding':binding,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'probability_dtype':'float32','fixed_C1_transformed_without_inference':True,'expert_values_read':False,'seconds':time.time()-tic})
    write(out/'success.json',{'manifest_sha256':sha(out/'manifest.json'),'count':len(recs)})
    print('PREDICTION COMPLETE',seed,name,len(recs),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=SEEDS,required=True);ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--zero-hint',action='store_true');a=ap.parse_args()
    try:main(a.seed,a.arm,a.zero_hint)
    except Exception as e:fail(BASE/f'predictions/{a.seed}/{a.arm+"_zero_hint" if a.zero_hint else a.arm}',e);raise
