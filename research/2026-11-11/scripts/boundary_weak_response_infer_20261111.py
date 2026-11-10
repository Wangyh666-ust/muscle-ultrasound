"""Inference-only diagnostic replay. This file imports real torch, never CPU stubs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json,hashlib,time
import numpy as np
import cv2,torch
from scipy.ndimage import map_coordinates
from scripts.round34_model import GuidedBoundaryUNet
from scripts.round41_native import predict,VIEWS,spec,warp
from src.umud_baseline.data import resize_long_side,restore_original_coords,IMAGENET_MEAN,IMAGENET_STD

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/boundary_weak_response_investigation_20261111'


def j(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


@torch.inference_mode()
def at_resolution(model,image,size):
    h,w=image.shape[:2];im,scale=resize_long_side(image,size,False);nh,nw=im.shape[:2]
    im=np.pad(im,((0,size-nh),(0,size-nw),(0,0)));x=torch.from_numpy(im.copy()).permute(2,0,1).float()/255
    x=(x-torch.tensor(IMAGENET_MEAN)[:,None,None])/torch.tensor(IMAGENET_STD)[:,None,None]
    x=torch.cat([x,torch.zeros((1,size,size))],0)
    with torch.autocast('cuda',dtype=torch.float16):a,b=model(x[None].cuda())
    ps=torch.cat([a,b],1).float().sigmoid()[0].cpu().numpy()
    return np.stack([restore_original_coords(p,(w,h),scale,(size-nw,size-nh),'nearest') for p in ps])


def near(field,points,radius):
    n=int(np.ceil(radius/.5));offset=np.arange(-n,n+1,dtype=float)*.5
    offset=np.unique(np.r_[offset[abs(offset)<=radius+1e-8],-radius,radius])
    return np.max([map_coordinates(field,[points[:,1]+dy,points[:,0]],order=1,mode='constant',cval=0.) for dy in offset],axis=0)


def stats(v):
    return dict(median=float(np.median(v)),mean=float(np.mean(v)),maximum=float(np.max(v)),over018=float(np.mean(v>=.18)),over05=float(np.mean(v>=.5)))


def measure(prob,gt,radius):
    out={}
    for i,side in enumerate(('upper','lower')):
        fields=dict(role=prob[i+1],union=prob[0],fused=prob[i+1]*(.4+.6*prob[0]))
        profiles={k:near(v,gt[side],radius) for k,v in fields.items()}
        out[side]=dict(metrics={k:stats(v) for k,v in profiles.items()},profiles={k:v.astype(float).tolist() for k,v in profiles.items()})
    return out


def main():
    proto=j(OUT/'protocol.json');assert sha(ROOT/proto['model'])==proto['model_sha256']
    for rel,h in proto['protected'].items():assert sha(ROOT/rel)==h,rel
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    gpu=torch.cuda.get_device_name(0);model=GuidedBoundaryUNet()
    model.load_state_dict(torch.load(ROOT/proto['model'],map_location='cpu',weights_only=True)['model'],strict=True)
    model.cuda().eval().requires_grad_(False)
    t0=time.monotonic();done=0;retain={'ANON_144','ANON_46','ANON_152','image_0642.tif'}
    write(OUT/'inference_binding.json',dict(protocol_sha256=sha(OUT/'protocol.json'),model_sha256=proto['model_sha256'],gpu=gpu,torch=str(torch.__version__),cuda=torch.version.cuda,opencv=cv2.__version__,numpy=np.__version__,training=False,script_sha256=sha(__file__)))
    for row in proto['rows']:
        dest=OUT/f"results/{row['dataset']}__{row['id']}.json"
        if dest.exists():continue
        assert sha(ROOT/row['input'])==row['input_sha256'];assert sha(ROOT/row['cached_probability'])==row['cached_sha256']
        with np.load(ROOT/row['input']) as z:image=z['image'].copy();gt={s:z[s].copy() for s in ('upper','lower')}
        rgb=np.repeat(image[:,:,None],3,2);h,w=image.shape;total=np.zeros((3,h,w),np.float32);count=np.zeros((h,w),np.float32);view_metrics=[]
        for dx,dy in VIEWS:
            view=rgb if (dx,dy)==(0,0) else warp(rgb,spec(w,h,'translate'+str(dx)+'_'+str(dy)))
            pr=predict(model,view,True);inverse=np.array([[1,0,-dx],[0,1,-dy]],np.float32)
            valid=cv2.warpAffine(np.ones((h,w),np.uint8),inverse,(w,h),flags=cv2.INTER_NEAREST)>0
            aligned=np.stack([cv2.warpAffine(p,inverse,(w,h),flags=cv2.INTER_NEAREST) for p in pr])
            total+=aligned*valid[None];count+=valid
            if (dx,dy)==(0,0):identity=pr.copy()
            view_metrics.append(dict(view=[dx,dy],sides=measure(aligned,gt,row['baseline_radius_px'])))
        consensus=total/np.maximum(count,1)[None]
        with np.load(ROOT/row['cached_probability']) as z:cached=z['probability'].copy()
        assert cached.shape==consensus.shape
        replay=dict(max_abs_error=float(abs(consensus-cached).max()),mean_abs_error=float(abs(consensus-cached).mean()))
        assert replay['max_abs_error']<1e-5,(row['id'],replay)
        probabilities=dict(F512_single=identity,F512_TTA5=consensus,F768_single=at_resolution(model,rgb,768))
        l,t,r,b=row['crop_box'];masked=np.zeros_like(rgb);masked[t:b,l:r]=rgb[t:b,l:r]
        probabilities['M512_single']=predict(model,masked,True)
        pc=predict(model,rgb[t:b,l:r],True);canvas=np.zeros_like(consensus);canvas[:,t:b,l:r]=pc
        probabilities['C512_single']=canvas
        result=dict(id=row['id'],dataset=row['dataset'],cohort=row['cohort'],binding=sha(OUT/'protocol.json'),input_sha256=row['input_sha256'],
            cached_replay=replay,crop_reference_retention=row['crop_reference_retention'],views=view_metrics,
            arms={k:measure(p,gt,row['baseline_radius_px']) for k,p in probabilities.items()})
        write(dest,result)
        if row['id'] in retain:
            path=OUT/f"probability_examples/{row['dataset']}__{row['id']}.npz";path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**{k:p.astype(np.float16) for k,p in probabilities.items()})
        done+=1
        write(OUT/'progress.json',dict(completed=done,total=len(proto['rows']),last=row['id'],seconds=round(time.monotonic()-t0,2)))
        print('WEAK_RESPONSE_DIAGNOSTIC',done,len(proto['rows']),row['id'],round(time.monotonic()-t0,2),replay,flush=True)
    results=[j(p) for p in sorted((OUT/'results').glob('*.json'))];assert len(results)==24
    write(OUT/'inference_done.json',dict(completed=len(results),new_completed=done,seconds=time.monotonic()-t0,
        replay_max_error=max(r['cached_replay']['max_abs_error'] for r in results),training=False))
    print('DIAGNOSTIC_INFERENCE_DONE',len(results),flush=True)


if __name__=='__main__':main()
