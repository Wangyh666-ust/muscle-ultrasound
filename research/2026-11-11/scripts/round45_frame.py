"""Conservative image evidence and boundary-only reference width. No GT reads."""
from scripts.round45_common import *
from scripts.round41_native import image_roi

def annulus(gray):
    h,w=gray.shape;a,b,c,d=0,0,w,h;peak=int(gray.max())
    if int(gray.min())==peak:return [a,b,c,d]
    while c-a>2 and d-b>2:
        v=int(gray[b,a])
        if v>=peak:break
        if not (np.all(gray[b,a:c]==v) and np.all(gray[d-1,a:c]==v) and np.all(gray[b:d,a]==v) and np.all(gray[b:d,c-1]==v)):break
        a+=1;b+=1;c-=1;d-=1
    return [a,b,c,d]

def frame(gray,legacy_roi,method):
    box=annulus(gray);a,b,c,d=box;crop=gray[b:d,a:c];first=image_roi(crop);second=None
    low,high=np.percentile(crop,[1,99])
    if high-low>=8:
        norm=np.rint(np.clip((crop.astype(float)-low)*255/(high-low),0,255)).astype(np.uint8);second=image_roi(norm)
    candidates=[x for x in (first,second if method=='F1' else None) if x is not None and not x['fallback']]
    fallback=False;reason='texture_reference_'+method
    if candidates:
        l=max(x['box'][0] for x in candidates);rr=min(x['box'][2] for x in candidates)
        if rr-l<max(8,.2*(c-a)):fallback=True;reason='narrow_or_empty_intersection'
    else:fallback=True;reason='no_valid_texture_locator'
    if fallback:l,rr=legacy_roi['box'][0],legacy_roi['box'][2]
    else:l+=a;rr+=a
    return dict(evidence_box=box,reference_interval=[int(l),int(rr)],reference_width=float(rr-l),fallback=fallback,reason=reason,raw_locator=first,normalized_locator=second,method=method)
