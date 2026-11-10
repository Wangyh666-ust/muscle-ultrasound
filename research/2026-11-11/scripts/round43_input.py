"""Image-only conservative field; fixed native five-view protocol."""
from scripts.round43_common import *
from scripts.round41_native import spec as oldspec,warp as oldwarp,normalized_crop,consensus,image_roi

def spec(w,h,name):
    if name not in NEW43:return oldspec(w,h,name)
    l,r,t,b=(.12,.04,.08,.02) if name=='r43_asymmetric' else (.08,)*4
    l,r,t,b=round(w*l),round(w*r),round(h*t),round(h*b)
    return dict(sx=1.,sy=1.,tx=float(l),ty=float(t),width=w+l+r,height=h+t+b,name=name,brightness=1.,padding=[l,r,t,b],border_value=16 if name=='r43_gray08' else 0)

def warp(im,t):
    if t['name'] not in NEW43:return oldwarp(im,t)
    l,r,top,b=t['padding'];out=cv2.copyMakeBorder(im,top,b,l,r,cv2.BORDER_CONSTANT,value=(t['border_value'],)*3)
    assert np.array_equal(out[top:top+im.shape[0],l:l+im.shape[1]],im)
    return out

def field(gray,margin):
    h,w=gray.shape;roi=image_roi(gray)
    if roi['fallback']:return np.ones((h,w),bool),dict(**roi,input_box=[0,0,w,h],valid_width=w,margin=margin)
    yy,xx=np.nonzero(gray>8);x0,x1,y0,y1=int(xx.min()),int(xx.max()+1),int(yy.min()),int(yy.max()+1)
    ch,cw=y1-y0,x1-x0;scale=512/max(ch,cw)
    a=cv2.resize(gray[y0:y1,x0:x1],(max(1,round(cw*scale)),max(1,round(ch*scale))),interpolation=cv2.INTER_AREA).astype(np.float32)
    mean=cv2.blur(a,(11,11));std=np.sqrt(np.maximum(0,cv2.blur(a*a,(11,11))-mean*mean))
    m=cv2.morphologyEx(((mean>12)&(std>6)).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    _,labs,ss,_=cv2.connectedComponentsWithStats(m,8);m=(labs==1+np.argmax(ss[1:,cv2.CC_STAT_AREA])).astype(np.uint8)
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);filled=np.zeros_like(m);cv2.drawContours(filled,cs,-1,1,cv2.FILLED)
    native=np.zeros((h,w),np.uint8);native[y0:y1,x0:x1]=cv2.resize(filled,(cw,ch),interpolation=cv2.INTER_NEAREST)
    bx0,by0,bx1,by1=roi['box'];mx,my=round((bx1-bx0)*margin),round((by1-by0)*margin)
    native=cv2.dilate(native,np.ones((2*my+1,2*mx+1),np.uint8))>0
    box=[bx0-mx,by0-my,bx1+mx,by1+my];cols=np.flatnonzero(native.any(0))
    return native,dict(**roi,input_box=box,valid_width=int(len(cols)),margin=margin,valid_pixels=int(native.sum()))

def crop_box(im,box):
    a,b,c,d=box;h,w=im.shape[:2];out=np.zeros((d-b,c-a,3),np.uint8);l,t,r,bt=max(0,a),max(0,b),min(w,c),min(h,d)
    out[t-b:bt-b,l-a:r-a]=im[t:bt,l:r];return out

def input_probability(model,im,meta):
    a,b,c,d=meta['input_box'];h,w=im.shape[:2];crop=crop_box(im,meta['input_box']);p=consensus(model,crop,True)
    out=np.full((3,h,w),np.nan,np.float32);l,t,r,bt=max(0,a),max(0,b),min(w,c),min(h,d)
    out[:,t:bt,l:r]=p[:,t-b:bt-b,l-a:r-a];return out

def loadfield(r,tr,margin):
    p=fieldpath(r,tr);side=p.with_suffix('.json')
    if p.exists():
        m=j(side);assert m['margin']==margin and m['source']==sha(Path(__file__)) and sha(p)==m['sha256']
        with np.load(p) as z:v=np.unpackbits(z['bits'])[:np.prod(z['shape'])].reshape(z['shape']).astype(bool)
        return v,m
    from src.umud_baseline.data import read_image_gray3
    t=spec(r['width'],r['height'],tr);im=warp(read_image_gray3(ROOT/r['image_path']),t);v,m=field(im[:,:,0],margin)
    p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,bits=np.packbits(v),shape=v.shape)
    m.update(key=key(r),transform=t,source=sha(Path(__file__)),sha256=sha(p));write(side,m);return v,m
