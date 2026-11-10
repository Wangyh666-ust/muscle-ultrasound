"""Round35 paths and shared serialization; old rounds remain immutable."""
from scripts.round34_common import j,sha,clean,write,require,unique,verify,binary,image_identity,stats,fail,ROOT,R30,R31,R32,R33,C1
from pathlib import Path
import numpy as np
import json,hashlib
R34=ROOT/'runs/round34_boundary_guidance_v1'
BASE=ROOT/'runs/round35_boundary_continuity_v1'
CFG=ROOT/'configs/round35_boundary_continuity.json'
SEEDS=(20261023,20261024)
ARMS=('T0','T1','T2','T3')
DATASETS={'apo153':153,'dev282':282,'external34':34}
def labels():return j(R34/'labels/guided_index.json')
def inputs():return [r for r in j(R34/'c1_guidance/manifest.json')['rows'] if r['dataset']!='train772']
def source_prediction(seed,arm):return R34/f'predictions/{seed}/A' if arm=='T0' else BASE/f'predictions/{seed}/{arm}'
def transform(w,h,name):
    from scripts.round32_common import spec
    if name in ('identity','scale075','scale125','pad10'):return spec(w,h,name)
    if name.startswith('pad'):
        d=int(name[3:])/100;dx,dy=round(w*d),round(h*d)
        return dict(sx=1.,sy=1.,tx=float(dx),ty=float(dy),width=w+2*dx,height=h+2*dy,name=name)
    s=int(name[5:])/100;nw,nh=round(w*s),round(h*s);sx,sy=nw/w,nh/h
    return dict(sx=sx,sy=sy,tx=(sx-1)/2,ty=(sy-1)/2,width=nw,height=nh,name=name)
def warp(a,t,mask=False):
    from PIL import Image
    if t['name']=='identity':return a.copy()
    if t['name'].startswith('pad'):
        pads=((int(t['ty']),int(t['ty'])),(int(t['tx']),int(t['tx'])))+(((0,0),) if a.ndim==3 else ())
        return np.pad(a,pads)
    v=np.asarray(Image.fromarray(a.astype(np.uint8)*255 if mask else a).resize((t['width'],t['height']),Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR))
    return v>127 if mask else v
