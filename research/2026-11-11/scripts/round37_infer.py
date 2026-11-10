import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,torch
from scripts.round37_io import *
from scripts.round37_core import image_roi,normalized_crop
from scripts.round32_common import spec,warp,inverse_spec
from scripts.round32_infer import infer_array
from src.umud_baseline.data import read_image_gray3,restore_original_coords
from src.umud_baseline.direction_model import DirectionAuxUNet

def transform(r,name):
    w,h=r['width'],r['height']
    if name in ('translate2_3','brightness095'):return dict(sx=1.,sy=1.,tx=2. if name=='translate2_3' else 0.,ty=3. if name=='translate2_3' else 0.,width=w,height=h,name=name)
    return spec(w,h,name)
def image_transform(im,t):
    if t['name']=='brightness095':return np.rint(im.astype(float)*.95).clip(0,255).astype(np.uint8)
    if t['name']=='translate2_3':return cv2.warpAffine(im,np.array([[1,0,2],[0,1,3]],np.float32),(t['width'],t['height']),flags=cv2.INTER_NEAREST)
    return warp(im,t)
def back(a,t,w,h,interp=cv2.INTER_LINEAR):
    it=inverse_spec(t);return cv2.warpAffine(a,np.array([[it['sx'],0,it['tx']],[0,it['sy'],it['ty']]],np.float32),(w,h),flags=interp)
def normalized_predict(model,im,roi):
    crop,box=normalized_crop(im,roi);p,meta,_=infer_array(model,crop,1024,False)
    cp=restore_original_coords(p[0],meta['original_wh'],meta['scale'],meta['pad'],'nearest')
    a,b,c,d=box;h,w=im.shape[:2];out=np.zeros((h,w),np.float32);l,t,r,bt=max(0,a),max(0,b),min(w,c),min(h,d)
    out[t:bt,l:r]=cp[t-b:bt-b,l-a:r-a]
    return out,p,meta,box
def dice(a,b):return float(2*(a&b).sum()/max(a.sum()+b.sum(),1)) if a.any() or b.any() else 1.
def main():
    verify_binding();cfg=j(CFG);rois=j(BASE/'roi.json');g=j(S01/'groups.json');members={m for v in g['groups'] for m in v['members']};reps={v['representative'] for v in g['groups']}
    if (BASE/'inference_success.json').exists():print('INFERENCE ALREADY COMPLETE');return
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=DirectionAuxUNet(pretrained=False).cuda().eval();model.load_state_dict(torch.load(CP,map_location='cpu',weights_only=False)['model'],strict=True)
    records=[];metrics=[];forwards=0;start=time.time()
    oldsrc=ROOT/'runs/round32_chain_foundation_v1/inference/fasc_transformed';old={(key(p),p['transform']['name']):p for p in j(oldsrc/'manifest.json')['rows']}
    for i,r in enumerate(inputs()):
        k=key(r);w,h=r['width'],r['height'];im=read_image_gray3(ROOT/r['image_path']);assert sha(ROOT/r['mask_path'])==r['mask_sha256']
        baseline,p,meta,box=normalized_predict(model,im,rois[k]);forwards+=1
        dest=maskpath('normalized','identity',r);dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,mask=baseline>.5)
        records.append(dict(key=k,network='normalized',transform=transform(r,'identity'),path=dest.relative_to(BASE).as_posix(),sha256=sha(dest),roi=rois[k],crop_box=box))
        if k in reps:
            repeat,_,_,_=normalized_predict(model,im,rois[k]);forwards+=1
            assert np.array_equal(repeat,baseline),'normalized repeat nondeterminism'
        if k in members:
            pp=BASE/f"probability/normalized/{r['dataset']}/{Path(r['image_id']).stem}.npz";pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,prob=p,meta=json.dumps(meta),crop_box=box)
            for name in cfg['transforms']:
                if name=='brightness095' and k not in reps:continue
                t=transform(r,name);img=image_transform(im,t);roi=image_roi(img[:,:,0]);q,_,_,box2=normalized_predict(model,img,roi);forwards+=1
                dest=maskpath('normalized',name,r);dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,mask=q>.5)
                records.append(dict(key=k,network='normalized',transform=t,path=dest.relative_to(BASE).as_posix(),sha256=sha(dest),roi=roi,crop_box=box2))
                qb=back(q,t,w,h);valid=back(np.ones(q.shape,np.uint8),t,w,h,cv2.INTER_NEAREST)>0;valid=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))>0
                fg=((baseline>.1)|(qb>.1))&valid
                metrics.append(dict(key=k,transform=name,mask_dice=dice((baseline>.5)&valid,(qb>.5)&valid),foreground_probability_mae=float(abs(baseline-qb)[fg].mean()) if fg.any() else 0.,roi_fallback=roi['fallback']))
                # Existing old masks can be reused exactly; only translation cache was not retained.
                if name=='translate2_3':
                    _,_,ms=infer_array(model,img,1024,False);forwards+=1;path=maskpath('old',name,r);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,mask=ms[0]);pathrel=path.relative_to(ROOT).as_posix();sh=sha(path)
                elif name=='brightness095':
                    path=S01/f"brightness/{r['dataset']}/{Path(r['image_id']).stem}.npz";pathrel=path.relative_to(ROOT).as_posix();sh=sha(path)
                else:
                    info=old[k,name];path=oldsrc/info['path'];assert sha(path)==info['sha256'];pathrel=path.relative_to(ROOT).as_posix();sh=info['sha256']
                records.append(dict(key=k,network='old',transform=t,root_path=pathrel,sha256=sh,roi=roi))
        if (i+1)%20==0:print('INFER',i+1,469,'forwards',forwards,'seconds',round(time.time()-start),flush=True)
    write(BASE/'inference.json',{'rows':records,'normalized_probability_metrics':metrics,'new_forwards':forwards,'device':torch.cuda.get_device_name(0),'seconds':time.time()-start,'repeat_images':len(reps),'checkpoint_sha256':sha(CP)})
    status('inference',{'manifest_sha256':sha(BASE/'inference.json'),'new_forwards':forwards,'identity_images':469,'transformed_images':len(metrics),'training':False})
    print('INFERENCE COMPLETE',forwards,flush=True)
if __name__=='__main__':main()
