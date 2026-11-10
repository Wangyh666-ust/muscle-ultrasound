"""Shared C1 component cache: identical calculation reused across boundary seeds."""
import sys,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.round38_common import *
from scripts.round38_core import state

def paths(r,tr):
    p=BASE/f"components/{tr}/{r['dataset']}/{Path(r['image_id']).stem}"
    return p.with_suffix('.json.gz'),p.with_suffix('.npz')

def get_states(r,tr,cfg):
    jp,npz=paths(r,tr);src=cachepath(r,tr,'c1');rec=j(src.with_suffix('.json'));assert sha(src)==rec['sha256']
    if jp.exists():
        data=gzread(jp);assert data['source_sha256']==rec['sha256'] and data['config_sha256']==sha(CFG)
        assert sha(npz)==data['evidence_sha256'];ev=unpack(npz)
        return [{**s,'evidence':e} for s,e in zip(data['states'],ev)]
    masks=unpack(src);states=[state(m,rec['roi'],cfg) for m in masks];ev=np.stack([s['evidence'] for s in states])
    packsave(npz,ev,np.zeros((1,1,1),np.float16),dict(kind='filtered_component_pixels'))
    gzwrite(jp,dict(source_sha256=rec['sha256'],config_sha256=sha(CFG),evidence_sha256=sha(npz),
        states=[{k:v for k,v in s.items() if k!='evidence'} for s in states],core_sha256=sha(ROOT/'scripts/round38_core.py')))
    return states

def main(watch):
    verify_code('extract');cfg=j(CFG);cv2.setNumThreads(1);n=0;start=time.time();rows=ordered_inputs();total=sum(len(transforms(r)) for r in rows)
    for r in rows:
        for tr in transforms(r):
            src=cachepath(r,tr,'c1')
            while not src.with_suffix('.json').exists():
                if not watch:raise FileNotFoundError(src)
                time.sleep(5)
            get_states(r,tr,cfg);n+=1
            if n%20==0:
                print('EXTRACT',n,total,'seconds',round(time.time()-start),flush=True)
                write(BASE/'progress/components.json',dict(done=n,total=total,elapsed=time.time()-start))
    write(BASE/'components_success.json',dict(records=n,seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');main(p.parse_args().watch)
