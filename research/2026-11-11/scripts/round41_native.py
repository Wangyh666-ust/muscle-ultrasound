"""Standalone frozen input and five-view inference for Colab evaluation.

Only consumed consensus outputs are calculated. Full-canvas unused C1 branches
are omitted; this is verified against the original consumed mask/probability.
"""
import math
import numpy as np,cv2,torch
from PIL import Image
from src.umud_baseline.data import resize_long_side,restore_original_coords,IMAGENET_MEAN,IMAGENET_STD
VIEWS=((0,0),(2,0),(-2,0),(0,2),(0,-2))

def image_roi(gray):
    h,w=gray.shape;yy,xx=np.nonzero(gray>8)
    if len(xx)<.01*gray.size:return dict(box=[0,0,w,h],fallback=True,reason='insufficient_image_content')
    x0,x1,y0,y1=int(xx.min()),int(xx.max()+1),int(yy.min()),int(yy.max()+1)
    crop=gray[y0:y1,x0:x1];ch,cw=crop.shape;scale=512/max(ch,cw)
    a=cv2.resize(crop,(max(1,round(cw*scale)),max(1,round(ch*scale))),interpolation=cv2.INTER_AREA).astype(np.float32)
    mean=cv2.blur(a,(11,11));std=np.sqrt(np.maximum(0,cv2.blur(a*a,(11,11))-mean*mean))
    mask=cv2.morphologyEx(((mean>12)&(std>6)).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n<2:return dict(box=[0,0,w,h],fallback=True,reason='no_texture_component')
    q=stats[1+np.argmax(stats[1:,cv2.CC_STAT_AREA])];x,y,rw,rh,area=map(int,q)
    if area<.05*a.size or rw<.2*a.shape[1] or rh<.2*a.shape[0]:return dict(box=[0,0,w,h],fallback=True,reason='small_or_narrow_texture')
    box=[max(0,x0+math.floor(x*cw/a.shape[1])),max(0,y0+math.floor(y*ch/a.shape[0])),min(w,x0+math.ceil((x+rw)*cw/a.shape[1])),min(h,y0+math.ceil((y+rh)*ch/a.shape[0]))]
    return dict(box=box,fallback=False,reason='largest_textured_field',content_box=[x0,y0,x1,y1])

def normalized_crop(image,roi,margin=.03):
    h,w=image.shape[:2]
    if roi['fallback']:return image.copy(),[0,0,w,h]
    x0,y0,x1,y1=roi['box'];mx=round((x1-x0)*margin);my=round((y1-y0)*margin);box=[x0-mx,y0-my,x1+mx,y1+my];a,b,c,d=box
    out=np.zeros((d-b,c-a,3),np.uint8);l,t,r,bt=max(0,a),max(0,b),min(w,c),min(h,d);out[t-b:bt-b,l-a:r-a]=image[t:bt,l:r]
    return out,box

def spec(w,h,name):
    old=name.removeprefix('r41_');sx=sy=1.;tx=ty=0.;nw,nh=w,h
    scales={'scale075':.75,'scale125':1.25,'scale095':.95,'scale105':1.05,'scale0975':.975,'scale1025':1.025,'scale0985':.985,'scale1015':1.015}
    brightness={'brightness095':.95,'brightness105':1.05,'brightness0975':.975}
    if old in scales:
        s=scales[old];nw,nh=round(w*s),round(h*s);sx,sy=nw/w,nh/h;tx,ty=(sx-1)/2,(sy-1)/2
    elif old.startswith('pad'):
        p=int(old[3:])/100;tx,ty=round(w*p),round(h*p);nw,nh=w+2*tx,h+2*ty
    elif old.startswith('translate'):tx,ty=map(int,old[9:].split('_'))
    elif old!='identity' and old not in brightness:raise ValueError(name)
    return dict(sx=sx,sy=sy,tx=float(tx),ty=float(ty),width=nw,height=nh,name=name,brightness=brightness.get(old,1.))

def warp(im,t):
    name=t['name'].removeprefix('r41_')
    if name=='identity':return im.copy()
    if name.startswith('brightness'):return np.rint(im.astype(float)*t['brightness']).clip(0,255).astype(np.uint8)
    if name.startswith('scale'):return np.asarray(Image.fromarray(im).resize((t['width'],t['height']),Image.Resampling.BILINEAR))
    return cv2.warpAffine(im,np.array([[1,0,t['tx']],[0,1,t['ty']]],np.float32),(t['width'],t['height']),flags=cv2.INTER_NEAREST)

@torch.no_grad()
def predict(model,image,boundary=False):
    h,w=image.shape[:2];long_side=512 if boundary else 1024;im,scale=resize_long_side(image,long_side,False);nh,nw=im.shape[:2]
    im=np.pad(im,((0,long_side-nh),(0,long_side-nw),(0,0)));x=torch.from_numpy(im.copy()).permute(2,0,1).float()/255
    x=(x-torch.tensor(IMAGENET_MEAN)[:,None,None])/torch.tensor(IMAGENET_STD)[:,None,None]
    if boundary:x=torch.cat([x,torch.zeros((1,long_side,long_side))],0)
    with torch.autocast('cuda',dtype=torch.float16):a,b=model(x[None].cuda())
    ps=(torch.cat([a,b],1) if boundary else a).float().sigmoid()[0].cpu().numpy()
    return np.stack([restore_original_coords(p,(w,h),scale,(long_side-nw,long_side-nh),'nearest') for p in ps])

def consensus(model,image,boundary=False):
    h,w=image.shape[:2];total=np.zeros((3 if boundary else 1,h,w),np.float32);count=np.zeros((h,w),np.float32)
    for dx,dy in VIEWS:
        t=spec(w,h,'translate'+str(dx)+'_'+str(dy));view=image if (dx,dy)==(0,0) else warp(image,t);p=predict(model,view,boundary)
        iv=np.array([[1,0,-dx],[0,1,-dy]],np.float32);valid=cv2.warpAffine(np.ones((h,w),np.uint8),iv,(w,h),flags=cv2.INTER_NEAREST)>0
        for ch in range(len(p)):total[ch]+=cv2.warpAffine(p[ch],iv,(w,h),flags=cv2.INTER_NEAREST)*valid
        count+=valid
    return total/np.maximum(count,1)[None]

def used_probability(model,im,boundary=False):
    roi=image_roi(im[:,:,0]);h,w=im.shape[:2]
    if boundary:return consensus(model,im,True),roi
    crop,box=normalized_crop(im,roi);p=consensus(model,crop,False);a,b,c,d=box;out=np.zeros((1,h,w),np.float32);l,t,r,bt=max(0,a),max(0,b),min(w,c),min(h,d)
    out[:,t:bt,l:r]=p[:,t-b:bt-b,l-a:r-a];return out,roi
