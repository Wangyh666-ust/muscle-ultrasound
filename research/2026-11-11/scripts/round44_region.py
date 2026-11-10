"""Image-only peripheral evidence exclusion; frozen learned models untouched."""
import cv2,numpy as np
from scripts.round44_common import *
from scripts.round41_native import image_roi
from scripts.round43_input import field as oldfield,crop_box,spec as oldspec,warp as oldwarp

def stripbox(gray,tol):
    h,w=gray.shape;a,b,c,d=0,0,w,h
    # Keep every nonflat line, including thin annotations or boundary echoes.
    for _ in range(4):
        prev=(a,b,c,d)
        while b<d-2 and np.ptp(gray[b,a:c].astype(np.int16))<=tol:b+=1
        while d>b+2 and np.ptp(gray[d-1,a:c].astype(np.int16))<=tol:d-=1
        while a<c-2 and np.ptp(gray[b:d,a].astype(np.int16))<=tol:a+=1
        while c>a+2 and np.ptp(gray[b:d,c-1].astype(np.int16))<=tol:c-=1
        if prev==(a,b,c,d):break
    return [a,b,c,d]

def locate(gray,cfg):
    h,w=gray.shape;box=stripbox(gray,cfg['flat_tolerance']);a,b,c,d=box;crop=gray[b:d,a:c]
    if min(c-a,d-b)<8 or (c-a)*(d-b)<.05*w*h:
        return dict(input_box=oldfield(gray,.03)[1]['input_box'],evidence_box=[0,0,w,h],fallback=True,reason='ambiguous_tiny_nonflat_field',scan_box=[0,0,w,h])
    detector=crop
    if cfg['locator']=='L1':
        low,high=np.percentile(crop,[1,99])
        if high-low<8:
            return dict(input_box=oldfield(gray,.03)[1]['input_box'],evidence_box=box,fallback=True,reason='weak_contrast',scan_box=box)
        detector=np.rint(np.clip((crop.astype(float)-low)*255/(high-low),0,255)).astype(np.uint8)
    roi=image_roi(detector)
    if roi['fallback']:
        return dict(input_box=oldfield(gray,.03)[1]['input_box'],evidence_box=box,fallback=True,reason='texture_uncertain',scan_box=box)
    x0,y0,x1,y1=roi['box'];x0+=a;x1+=a;y0+=b;y1+=b;mx=round((x1-x0)*.03);my=round((y1-y0)*.03)
    return dict(input_box=[x0-mx,y0-my,x1+mx,y1+my],evidence_box=box,scan_box=[x0,y0,x1,y1],fallback=False,reason='peripheral_flat_bands_'+cfg['locator'])

def input_crop(view,meta):
    # Confirmed display padding is represented by the same zero context as
    # out-of-canvas padding. Actual tissue intensities are never normalized.
    a,b,c,d=meta['evidence_box'];v=np.zeros_like(view);v[b:d,a:c]=view[b:d,a:c]
    return crop_box(v,meta['input_box'])

def spec(w,h,name):
    if name not in NEW44:return oldspec(w,h,name)
    scale=1.02 if name=='r44_scale_gray' else 1.;sw,sh=round(w*scale),round(h*scale)
    ll,rr,tt,bb=(.05,.11,.03,.07) if name=='r44_gray32_asym' else (.06,)*4
    l,r,t,b=round(sw*ll),round(sw*rr),round(sh*tt),round(sh*bb)
    return dict(name=name,sx=sw/w,sy=sh/h,tx=(sw/w-1)/2+l,ty=(sh/h-1)/2+t,width=sw+l+r,height=sh+t+b,
        padding=[l,r,t,b],base_width=sw,base_height=sh,brightness=1.03 if name=='r44_bright_gray' else 1.,border_value=4 if name=='r44_gray04' else 32 if name=='r44_gray32_asym' else 16)

def warp(im,t):
    if t['name'] not in NEW44:return oldwarp(im,t)
    x=np.clip(np.rint(im.astype(float)*t['brightness']),0,255).astype(np.uint8)
    if x.shape[1]!=t['base_width'] or x.shape[0]!=t['base_height']:x=cv2.resize(x,(t['base_width'],t['base_height']),interpolation=cv2.INTER_LINEAR)
    l,r,top,b=t['padding'];return cv2.copyMakeBorder(x,top,b,l,r,cv2.BORDER_CONSTANT,value=(t['border_value'],)*3)
