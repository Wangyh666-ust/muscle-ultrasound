"""Round32 isolated diagnostic utilities. No reference access in inference."""
import copy, hashlib, json, math
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/round32_chain_foundation_v1'
OLD=ROOT/'runs/round31_fascicle_guided_pair_v1'
BND=ROOT/'runs/round30_boundary_identity_v1'
CFG=ROOT/'configs/round32_chain_foundation.json'
SEEDS=(20261023,20261024)
TRANSFORMS=('identity','scale075','scale125','pad10')

def j(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def binary(p):
    with Image.open(p) as im:return np.asarray(im.convert('L'))>127
def verify_files(base,files):
    for rel,h in files.items():
        p=(base/rel).resolve()
        if not p.is_relative_to(base.resolve()) or sha(p)!=h:raise ValueError('source mismatch '+rel)
def inputs():return j(OLD/'c1/manifest.json')['rows']
def key(r):return (r['dataset'],r['image_id'])
def indexed(rows):
    d={key(r):r for r in rows}
    if len(d)!=len(rows):raise ValueError('duplicate inputs')
    return d
def spec(w,h,name):
    if name=='pad10':
        dx,dy=round(w*.1),round(h*.1);return dict(sx=1.,sy=1.,tx=float(dx),ty=float(dy),width=w+2*dx,height=h+2*dy,name=name)
    s={'identity':1.,'scale075':.75,'scale125':1.25}[name]
    nw,nh=round(w*s),round(h*s);sx,sy=nw/w,nh/h
    return dict(sx=sx,sy=sy,tx=(sx-1)/2,ty=(sy-1)/2,width=nw,height=nh,name=name)
def warp(a,t,mask=False):
    if t['name']=='identity':return a.copy()
    if t['name']=='pad10':
        pads=((int(t['ty']),int(t['ty'])),(int(t['tx']),int(t['tx'])))+(((0,0),) if a.ndim==3 else ())
        return np.pad(a,pads)
    im=Image.fromarray(a.astype(np.uint8)*255 if mask else a)
    v=np.asarray(im.resize((t['width'],t['height']),Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR))
    return v>127 if mask else v
def map_point(p,t,inverse=False):
    if inverse:return [(p[0]-t['tx'])/t['sx'],(p[1]-t['ty'])/t['sy']]
    return [p[0]*t['sx']+t['tx'],p[1]*t['sy']+t['ty']]
def inverse_spec(t):
    return {**t,'sx':1/t['sx'],'sy':1/t['sy'],'tx':-t['tx']/t['sx'],'ty':-t['ty']/t['sy']}
def map_curve(cv,t):
    if cv is None:return None
    c=copy.deepcopy(cv);sx,sy,tx,ty=[t[k] for k in ('sx','sy','tx','ty')]
    for part in [c,c['sup'],c['deep']]:
        part['support']=[x*sx+tx for x in part['support']]
        part['paired_x']=[x*sx+tx for x in part['paired_x']]
        part['max_sample_gap']*=sx
    for side in ('sup','deep'):
        f=c[side];f['knots_x']=[x*sx+tx for x in f['knots_x']];f['knots_y']=[y*sy+ty for y in f['knots_y']]
        for ray in ('left_ray','right_ray'):
            m,b=f[ray];f[ray]=[sy*m/sx,sy*b+ty-sy*m*tx/sx]
    return c
def map_segment(seg,t):
    s=copy.deepcopy(seg);sx,sy=t['sx'],t['sy']
    for k in ('centroid','p0','p1'):s[k]=map_point(s[k],t)
    th=math.radians(s['direction_deg']);factor=math.hypot(sx*math.cos(th),sy*math.sin(th))
    s['direction_deg']=math.degrees(math.atan2(sy*math.sin(th),sx*math.cos(th)))%180
    s['length_px']*=factor;s['width_px']*=sx*sy/factor;s['area_px']*=sx*sy
    return s
def stats(values):
    a=np.asarray([x for x in values if x is not None],float)
    if not len(a):return {'n':0,'mean':None,'median':None,'p90':None,'max':None}
    if not np.isfinite(a).all():raise ValueError('nonfinite stats')
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))

def freeze_execution():
    p=BASE/'execution_sources.json'
    code=['scripts/round32_common.py','scripts/round32_infer.py','configs/round32_chain_foundation.json',
          'scripts/round30_predict.py','scripts/round26_1_predict.py','src/umud_baseline/data.py',
          'src/umud_baseline/boundary_identity.py','src/umud_baseline/direction_model.py']
    if p.exists():
        verify_files(ROOT,j(p)['code']);return j(p)
    verify_files(BASE,j(BASE/'preparation_success.json')['files'])
    bindings=j(BASE/'source_bindings.json');verify_files(ROOT,bindings['consumed_verified_sources'])
    for rel,h in j(OLD/'sources.json')['code'].items():
        if sha(ROOT/rel)!=h:raise ValueError('frozen measurement code changed: '+rel)
    chosen=[]
    for ds in ('apo153','dev282','external34'):
        rows=sorted([r for r in inputs() if r['dataset']==ds],key=lambda r:(r['width']*r['height'],r['image_id']))
        chosen.extend([list(key(rows[i])) for i in (0,len(rows)//2,len(rows)-1)])
    d={'code':{x:sha(ROOT/x) for x in code},'preparation_sha256':sha(BASE/'preparation_success.json'),
       'pilot_ids':chosen,'pilot_selection':'smallest/median/largest area within each fixed dataset; ID breaks ties',
       'references_used_for_selection':False,'transforms':TRANSFORMS,
       'interpolation':'PIL bilinear image / nearest mask; pixel-centre affine uses actual rounded sizes',
       'probabilities':'native padded canvas float32; exact original-space nearest restore on load',
       'measurements':'decode in transformed coordinates, inverse-map curves and segments, measure in original coordinates and original supplied scale'}
    write(p,d);return d
