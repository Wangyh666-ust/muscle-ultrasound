"""Paths and immutable-source helpers for Round37 only."""
import gzip, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/round37_stability_repair_v1'
OLD=ROOT/'runs/round36_candidate_consensus_v1'
S01=ROOT/'runs/s01_round36_stability_v1'
CFG=ROOT/'configs/round37_stability_repair.json'
SEEDS=(20261023,20261024)
DS=('external34','apo153','dev282')
ARMS=('R','F0','F1','S','J','I','H','HI')
CP=ROOT/'runs/round26_1_direction_aux_corrected_v1/training/C1_corrected/checkpoints/epoch_017.pt'
def j(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return v.item()
    return v
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
def gzread(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def gzwrite(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(p,'wt',encoding='utf8',compresslevel=5) as f:json.dump(clean(v),f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def key(r):return r['dataset']+'|'+r['image_id']
def binary(p):return np.array(Image.open(p).convert('L'))>127
def inputs():return j(S01/'groups.json')['rows']
def originals(seed):return {key(r):r for d in DS for r in gzread(OLD/f'predictions/{seed}/{d}.json.gz')}
def maskpath(network,name,r):return BASE/f"masks/{network}/{name}/{r['dataset']}/{Path(r['image_id']).stem}.npz"
def status(stage,payload):write(BASE/f'{stage}_success.json',payload)
def verify_binding():
    b=j(BASE/'binding.json');assert sha(CFG)==b['config_sha256']
    for p,h in b['protected'].items():assert sha(ROOT/p)==h,'Protected changed: '+p
    return b
