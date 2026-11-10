"""Round34 isolated paths, binding and serialization. No default changes."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/round34_boundary_guidance_v1'
CFG=ROOT/'configs/round34_boundary_guidance.json'
R30=ROOT/'runs/round30_boundary_identity_v1'
R31=ROOT/'runs/round31_fascicle_guided_pair_v1'
R32=ROOT/'runs/round32_chain_foundation_v1'
R33=ROOT/'runs/round33_boundary_recognition_v1'
C1=ROOT/'runs/round26_1_direction_aux_corrected_v1/training/C1_corrected/checkpoints/epoch_017.pt'
SEEDS=(20261023,20261024)
ARMS=('A','B')
DATASETS={'apo153':153,'dev282':282,'external34':34}

def j(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return v.item()
    return v
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)
def require(ok,msg):
    if not ok:raise ValueError(msg)
def unique(rows,fields):
    d={tuple(r[k] for k in fields):r for r in rows};require(len(d)==len(rows),'duplicate '+str(fields));return d
def verify(files,root=ROOT):
    for rel,h in files.items():
        p=(root/rel).resolve();require(p.is_relative_to(root.resolve()) and sha(p)==h,'binding mismatch '+rel)
def binary(p):return np.asarray(Image.open(p).convert('L'))>127
def image_identity(p):
    with Image.open(p) as im:return hashlib.sha256(im.convert('L').tobytes()).hexdigest(),list(im.size)
def stats(values):
    a=np.asarray([v for v in values if v is not None],float);require(np.isfinite(a).all(),'nonfinite stats')
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None,'max':float(a.max()) if len(a) else None}
def fail(out,e):
    (out/'success.json').unlink(missing_ok=True);write(out/'validation_failed.json',{'error':str(e),'type':type(e).__name__})
