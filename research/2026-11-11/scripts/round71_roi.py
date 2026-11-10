"""Image-only target area, independent of predictions and annotation extents."""
import cv2,numpy as np
from scipy.ndimage import uniform_filter,binary_fill_holes

def locate(image):
    im=np.asarray(image,np.float32);h,w=im.shape;k=max(5,round(min(h,w)*.018)//2*2+1)
    mean=uniform_filter(im,k);sd=np.sqrt(np.maximum(0,uniform_filter(im*im,k)-mean*mean))
    mask=((sd>=4)&(mean>=8)&(mean<=245)).astype(np.uint8)
    opening=max(3,round(min(h,w)*.008)//2*2+1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((opening,opening),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n<=1:return np.ones(im.shape,bool),dict(reason='ambiguous_no_texture',ambiguous=True,box=[0,0,w,h])
    label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));area=stats[label,cv2.CC_STAT_AREA]
    component=(labels==label).astype(np.uint8);points=cv2.findNonZero(component);hull=cv2.convexHull(points)
    result=np.zeros_like(component);cv2.fillConvexPoly(result,hull,1)
    radius=max(2,round(min(h,w)*.006));result=cv2.dilate(result,np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
    result=binary_fill_holes(result);yy,xx=np.where(result);box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    ambiguous=bool(area<.08*h*w or box[2]-box[0]<.25*w)
    if ambiguous:result=np.ones(im.shape,bool);box=[0,0,w,h]
    return result,dict(reason='texture_component_convex_envelope' if not ambiguous else 'ambiguous_texture_extent',ambiguous=ambiguous,
         box=box,component_area=int(area),texture_threshold=4,semantic_scan_segmentation=False)
