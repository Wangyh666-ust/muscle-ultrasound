"""Round39 paths and source locks; never changes historical implementations."""
from scripts.round38_common import *
R38=BASE
DEST=ROOT/'runs/round39_stability_mechanisms_v1'
PROTOCOL=ROOT/'configs/round39_stability_protocol.json'

def outpath(r,tr,seed,folder='predictions'):
    return DEST/f"{folder}/{seed}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"

def verify_preparation():
    verify_code('Round39')
    p=j(DEST/'preparation.json')
    assert sha(PROTOCOL)==p['protocol_sha256']
    assert sha(DEST/'partitions.json')==p['split_sha256']
    for name,h in p['source_sha256'].items():assert sha(ROOT/name)==h,name

def implementation_files():
    names=('round39_core.py','round39_common.py','round39_run.py','round39_controls.py','round39_evaluate.py',
        'round38_core.py','round38_common.py','round38_extract.py','round38_speed.py','round38_evaluate.py')
    return [ROOT/'scripts'/n for n in names]+[PROTOCOL,CFG,R38/'binding.json',DEST/'calibration.json',DEST/'partitions.json',DEST/'frozen_denominators.json',DEST/'cache_manifest.json']

def verify_lock():
    q=j(DEST/'implementation_lock.json')
    for p,h in q['files'].items():assert sha(ROOT/p)==h,'Implementation changed after lock: '+p
    return q
