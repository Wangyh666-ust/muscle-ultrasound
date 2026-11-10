"""Current-image-only envelopes; separate context and measurement evidence."""
from scripts.round46_common import *
from scripts.round41_native import image_roi
from scripts.round43_input import crop_box
from scripts.round44_region import spec as oldspec,warp as oldwarp

def exterior(gray):
    h,w=gray.shape;a,b,c,d=0,0,w,h;bands=[]
    if int(np.ptp(gray))<8:return [a,b,c,d],bands,True
    # Only exactly constant, dark, complete exterior bands are certified.
    # Thin nonzero echoes (1..4 pixels) and bright rims are retained.
    for _ in range(8):
        previous=(a,b,c,d)
        for side in ('top','bottom','left','right'):
            rows=side in ('top','bottom');forward=side in ('top','left')
            n=(d-b) if rows else (c-a)
            if n<3:continue
            def line(i):
                if rows:return gray[(b+i if forward else d-1-i),a:c]
                return gray[b:d,(a+i if forward else c-1-i)]
            z=line(0);v=int(z[0])
            if v>64 or not np.all(z==v):continue
            run=1
            while run<n-2 and np.all(line(run)==v):run+=1
            if v and run<5:continue
            if side=='top':b+=run
            elif side=='bottom':d-=run
            elif side=='left':a+=run
            else:c-=run
            bands.append(dict(side=side,value=v,width=run))
        if previous==(a,b,c,d):break
    return [a,b,c,d],bands,False

def locate(gray,method):
    h,w=gray.shape;box,bands,ambiguous=exterior(gray);a,b,c,d=box
    core=gray[b:d,a:c];roi=None;fallback=ambiguous
    if method!='C':
        detector=core
        if method=='T1':
            lo,hi=np.percentile(core,[1,99])
            if hi-lo>=8:detector=np.rint(np.clip((core.astype(float)-lo)*255/(hi-lo),0,255)).astype(np.uint8)
            else:fallback=True
        roi=image_roi(detector)
        if roi['fallback']:fallback=True
    if method=='C' or fallback:fieldbox=list(box)
    else:
        x,y,z,t=roi['box'];fieldbox=[a+x,b+y,a+z,b+t]
    x,y,z,t=fieldbox;mx=round(.03*(z-x));my=round(.03*(t-y))
    return dict(method=method,scan_box=fieldbox,input_box=[x-mx,y-my,z+mx,t+my],evidence_box=box,reference_width=z-x,fallback=fallback,bands=bands,raw_locator=roi,reason='certified_dark_exterior_envelope' if method=='C' else 'texture_after_certified_exterior',semantic_scan_segmentation=False)

def input_crop(im,fm):
    a,b,c,d=fm['evidence_box'];v=np.zeros_like(im);v[b:d,a:c]=im[b:d,a:c]
    return crop_box(v,fm['input_box'])

def spec(w,h,name):
    if name not in NEW46:return oldspec(w,h,name)
    sc=.985 if name=='r46_bright_scale_gray' else 1.;sw,sh=round(w*sc),round(h*sc)
    fractions=(.09,0,0,0) if name=='r46_gray_left' else (.035,.115,.065,.025) if name=='r46_gray_unequal' else (.07,.025,.035,.095)
    l,r,t,b=[round(v*dim) for v,dim in zip(fractions,(sw,sw,sh,sh))]
    return dict(name=name,sx=sw/w,sy=sh/h,tx=(sw/w-1)/2+l,ty=(sh/h-1)/2+t,width=sw+l+r,height=sh+t+b,padding=[l,r,t,b],base_width=sw,base_height=sh,brightness=1.025 if name=='r46_bright_scale_gray' else 1.,border_value=24)

def warp(im,t):
    if t['name'] not in NEW46:return oldwarp(im,t)
    v=np.rint(im.astype(float)*t['brightness']).clip(0,255).astype(np.uint8)
    if v.shape[:2]!=(t['base_height'],t['base_width']):v=cv2.resize(v,(t['base_width'],t['base_height']),interpolation=cv2.INTER_LINEAR)
    l,r,a,b=t['padding'];return cv2.copyMakeBorder(v,a,b,l,r,cv2.BORDER_CONSTANT,value=(t['border_value'],)*3)
