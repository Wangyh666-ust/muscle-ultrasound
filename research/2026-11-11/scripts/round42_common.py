"""Round42 isolated evidence paths; never modifies prior experiment outputs."""
import os
for _n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_n]='1'
import sys,copy,time,json,math,gzip,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2
from scipy import ndimage
from scripts.round41_measure import (ROOT,DEST as R41,R40,CFG,KINDS,SEEDS,DS,NEW41,
    j,sha,write,gzwrite,gzread,key,ordered_inputs,inferpath,resultpath as old_resultpath,
    cachepath,unpack,finish)
from scripts.round38_common import GENTLE,STRESS,R37,S01
from scripts.round40_common import NEW
from scripts.round32_common import inverse_spec,map_segment
DEST=ROOT/'runs/round42_track_direction_v1'
SCOPES={'old11':GENTLE+NEW,'new5':NEW41,'combined16':GENTLE+NEW+NEW41,'stress4':STRESS}
SPLIT=j(R41/'partitions.json')

def rows(partition=None):
    return [r for r in ordered_inputs() if partition is None or SPLIT['assignment'][key(r)]==partition]

def extractpath(r):return DEST/f"extraction/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
def outpath(r,s):return DEST/f"measurements/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
def getboundary(r,tr,s):
    p=R41/f"inference_local_boundary/{s}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz" if tr in NEW41 else cachepath(r,tr,str(s))
    rec=j(p.with_suffix('.json'));assert sha(p)==rec['sha256']
    return (unpack(p) if tr in NEW41 else unpack(p)[3:]),rec,p

def axis(a,b):return float(abs((a-b+90)%180-90))

def stats(x):
    a=np.asarray([v for v in x if v is not None and np.isfinite(v)],float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
        p95=float(np.quantile(a,.95)) if len(a) else None,maximum=float(a.max()) if len(a) else None)

def bind():
    from scripts.round40_common import verify_lock
    verify_lock();DEST.mkdir(exist_ok=True)
    path=DEST/'execution_protocol.json'
    protected={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'scripts').glob('*.py')) if not p.name.startswith('round42')}
    protected.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'configs').glob('*.json')})
    protocol=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),approved_plan_sha256=sha(ROOT/'thinking/ROUND42_TRACK_DIRECTION_FL_PLAN_20261104.txt'),
        training=False,default_changed=False,subagents=False,source_run=str(R41),
        split_sha256=sha(R41/'partitions.json'),training_lock_sha256=sha(R41/'training_lock.json'),
        protected=protected,primary='locked decoder versus original K for B of each seed; A/F controls',
        development=138,confirmation=331,all=469,models=list(KINDS),boundary_seeds=list(SEEDS),
        scopes=SCOPES,matching=dict(reference_roi_width=1046,radii_reference_pixels=[1,2,4],minimum_fraction=.1,angle_gate=None),
        candidate_policy='controlled development evidence required; no confirmation tuning',
        inference_policy='frozen checkpoint only; numerical bridge required; no preview substitution')
    if not path.exists():write(path,protocol)
    else:
        old=j(path)
        for p,h in old['protected'].items():assert sha(ROOT/p)==h,p
    return j(path)

if __name__=='__main__':print('ROUND42_REGISTERED',len(bind()['protected']),flush=True)
