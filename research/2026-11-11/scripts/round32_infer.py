"""Frozen model probability replay and fixed image transforms. No expert/GT reads."""
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from scripts.round32_common import *
from src.umud_baseline.data import read_image_gray3,resize_long_side,restore_original_coords,IMAGENET_MEAN,IMAGENET_STD
from src.umud_baseline.boundary_identity import BoundaryIdentityUNet
from src.umud_baseline.direction_model import DirectionAuxUNet

@torch.no_grad()
def infer_array(model,image,long_side,boundary=True):
    h,w=image.shape[:2];im,scale=resize_long_side(image,long_side,False);nh,nw=im.shape[:2]
    im=np.pad(im,((0,long_side-nh),(0,long_side-nw),(0,0)))
    x=torch.from_numpy(im.copy()).permute(2,0,1).float()/255
    x=(x-torch.tensor(IMAGENET_MEAN)[:,None,None])/torch.tensor(IMAGENET_STD)[:,None,None]
    with torch.autocast('cuda',dtype=torch.float16):a,b=model(x[None].cuda())
    probs=(torch.cat([a,b],1) if boundary else a).float().sigmoid()[0].cpu().numpy()
    meta={'original_wh':[w,h],'scale':scale,'pad':[long_side-nw,long_side-nh]}
    masks=np.stack([restore_original_coords(p,(w,h),scale,meta['pad'],'nearest')>.5 for p in probs])
    return probs,meta,masks

def restore(path):
    z=np.load(path,allow_pickle=False);meta=json.loads(str(z['meta']))
    return np.stack([restore_original_coords(p,meta['original_wh'],meta['scale'],meta['pad'],'nearest') for p in z['prob']])

def model_for(seed=None):
    if seed is None:
        cp=ROOT/'runs/round26_1_direction_aux_corrected_v1/training/C1_corrected/checkpoints/epoch_017.pt'
        expected=j(OLD/'c1/manifest.json')['checkpoint_sha256'];model=DirectionAuxUNet(pretrained=False)
    else:
        cp=BND/f'training/{seed}/T1/checkpoints/epoch_017_model.pt'
        expected=j(BND/f'predictions/{seed}/T1/manifest.json')['checkpoint_sha256'];model=BoundaryIdentityUNet()
    if sha(cp)!=expected:raise ValueError('checkpoint changed')
    ck=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
    return model.cuda().eval()

def main(stage):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    execution=freeze_execution();all_rows=inputs();lookup=indexed(all_rows)
    dest=BASE/'inference'/stage;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'success.json').exists():
        verify_files(dest,j(dest/'success.json')['files']);print('ALREADY COMPLETE',stage);return
    start=time.time();outrows=[];forwards=0
    selected=[lookup[tuple(k)] for k in execution['pilot_ids']] if stage=='pilot' else all_rows
    transforms=('identity',) if stage in ('pilot','probability') else TRANSFORMS[1:]
    if stage=='probability' and not (BASE/'inference/pilot/success.json').exists():raise ValueError('pilot not complete')
    if stage=='transformed' and not (BASE/'inference/probability/success.json').exists():raise ValueError('probability replay not complete')
    if stage in ('pilot','probability','transformed'):
        for seed in SEEDS:
            model=model_for(seed);old=indexed(j(BND/f'predictions/{seed}/T1/manifest.json')['rows'])
            for i,r in enumerate(selected):
                image=read_image_gray3(ROOT/r['image_path'])
                if hashlib.sha256(image[:,:,0].tobytes()).hexdigest()!=r['pixel_sha256']:raise ValueError('image identity changed')
                for name in transforms:
                    t=spec(r['width'],r['height'],name);im=warp(image,t)
                    probs,meta,masks=infer_array(model,im,512);forwards+=1
                    path=dest/f"{seed}/{name}/{r['dataset']}/{Path(r['image_id']).stem}.npz";path.parent.mkdir(parents=True,exist_ok=True)
                    if stage=='transformed':np.savez_compressed(path,masks=masks)
                    else:np.savez_compressed(path,prob=probs.astype(np.float32),meta=json.dumps(meta))
                    rec={'seed':seed,'dataset':r['dataset'],'image_id':r['image_id'],'transform':t,'path':path.relative_to(dest).as_posix(),'sha256':sha(path)}
                    if name=='identity':
                        ref=old[key(r)];changes={}
                        for c,side in enumerate(('union','sup','deep')):
                            original=binary(BND/ref['mask_paths'][side]);delta=masks[c]!=original
                            changes[side]={'changed_pixels':int(delta.sum()),'total_pixels':int(delta.size),
                                          'max_probability_distance_from_half_on_changed':float(abs(restore_original_coords(probs[c],meta['original_wh'],meta['scale'],meta['pad'],'nearest')[delta]-.5).max()) if delta.any() else 0.}
                        rec['parity']=changes
                    outrows.append(rec)
                if (i+1)%30==0 or stage=='pilot':print(stage,seed,i+1,len(selected),flush=True)
            del model;torch.cuda.empty_cache()
    elif stage=='fasc_transformed':
        model=model_for();parity=[]
        for k in execution['pilot_ids']:
            r=lookup[tuple(k)];_,_,m=infer_array(model,read_image_gray3(ROOT/r['image_path']),1024,False);forwards+=1
            parity.append({'identity':k,'changed_pixels':int(np.sum(m[0]!=binary(ROOT/r['mask_path'])))})
        if any(r['changed_pixels'] for r in parity):
            write(dest/'parity_failed.json',parity);raise ValueError('C1 replay mismatch; investigate before transforms')
        write(dest/'pilot_parity.json',parity)
        for i,r in enumerate(all_rows):
            image=read_image_gray3(ROOT/r['image_path'])
            for name in TRANSFORMS[1:]:
                t=spec(r['width'],r['height'],name);_,_,m=infer_array(model,warp(image,t),1024,False);forwards+=1
                path=dest/f"{name}/{r['dataset']}/{Path(r['image_id']).stem}.npz";path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,mask=m[0])
                outrows.append({'dataset':r['dataset'],'image_id':r['image_id'],'transform':t,'path':path.relative_to(dest).as_posix(),'sha256':sha(path)})
            if (i+1)%30==0:print(stage,i+1,len(all_rows),flush=True)
    else:raise ValueError(stage)
    doc={'rows':outrows,'actual_forwards':forwards,'seconds':time.time()-start,'torch':torch.__version__,
         'gpu':torch.cuda.get_device_name(0),'execution_sources_sha256':sha(BASE/'execution_sources.json'),'references_read':False,'training':False}
    if stage in ('pilot','probability'):
        doc['parity_totals']={s:sum(r['parity'][s]['changed_pixels'] for r in outrows) for s in ('union','sup','deep')}
        doc['union_changed_images']=sum(bool(r['parity']['union']['changed_pixels']) for r in outrows)
    write(dest/'manifest.json',doc)
    files={r['path']:r['sha256'] for r in outrows};files['manifest.json']=sha(dest/'manifest.json')
    if stage=='fasc_transformed':files['pilot_parity.json']=sha(dest/'pilot_parity.json')
    write(dest/'success.json',{'files':files,'scope':stage,'probability_parity_exact':not any(doc.get('parity_totals',{}).values())})
    print('COMPLETE',stage,forwards,doc.get('parity_totals'),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','probability','transformed','fasc_transformed']);a=ap.parse_args()
    try:main(a.stage)
    except Exception as e:write(BASE/f'inference/{a.stage}/failure.json',{'error':str(e)});raise
