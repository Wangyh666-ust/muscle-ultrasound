"""Round40 isolated protocol, cache routing and immutable implementation lock."""
from scripts.round38_common import *
from scripts.round39_common import DEST as R39, verify_lock as verify39
DEST=ROOT/'runs/round40_intersection_qualification_v1'
NEW=('translate-2_1','scale0975','scale1025','pad15','brightness105')
ARMS=('J','Qoff','Qobs','Qsoft','T','U','TU','K')
OLD_CACHE=cachepath
OLD_SPEC=spec
OLD_WARP=warp
OLD_TRANSFORMS=transforms

def cachepath(r,tr,kind):
    return DEST/f"inference/{kind}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz" if tr in NEW else OLD_CACHE(r,tr,kind)

def outpath(r,tr,seed):return DEST/f"predictions/{seed}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
def priorpath(r,tr,seed):return R39/f"predictions/{seed}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
def transforms(r):return OLD_TRANSFORMS(r)+NEW
def spec(w,h,name):
    if name in ('scale0975','scale1025'):
        s={'scale0975':.975,'scale1025':1.025}[name];nw,nh=round(w*s),round(h*s);sx,sy=nw/w,nh/h
        return dict(sx=sx,sy=sy,tx=(sx-1)/2,ty=(sy-1)/2,width=nw,height=nh,name=name)
    return OLD_SPEC(w,h,name)
def warp(im,t):
    if t['name']=='brightness105':return np.rint(im.astype(float)*1.05).clip(0,255).astype(np.uint8)
    return OLD_WARP(im,t)
def verify_lock():
    verify39();q=j(DEST/'implementation_lock.json')
    for p,h in q['files'].items():assert sha(ROOT/p)==h,('Round40 locked file changed',p)
    return q

def get_state(r,tr,cfg):
    if tr not in NEW:
        from scripts.round38_extract import get_states
        return get_states(r,tr,cfg)[3]
    from scripts.round38_core import state
    src=cachepath(r,tr,'c1');rec=j(src.with_suffix('.json'));assert sha(src)==rec['sha256']
    # No cross-worker writes; the two seeds independently reproduce the same C1 extraction.
    return state(unpack(src)[3],rec['roi'],cfg)
