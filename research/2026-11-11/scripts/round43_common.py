"""Isolated Round43 paths, protocol and immutable sources."""
from scripts.round42_common import *
R42=DEST
DEST=ROOT/'runs/round43_input_support_v1'
KINDS=('B_20261023','B_20261024')
GROUPS=('R','I','D','ID','IDS','IDQ','IDSQ')
NEW43=('r43_black08','r43_asymmetric','r43_gray08')
SCOPES={**SCOPES,'new_black':NEW43[:2],'new_gray':NEW43[2:]}
MANIFEST=j(R41/'evaluation_manifest.json')
TRS={r['key']:r['transforms'] for r in MANIFEST['rows']}
MISSING={'dev282|image_'+n+'.tif' for n in ('2419','2420','2422','2461','2472','2475','2482','2486','2491','2509','2518')}
TAIL={'dev282|image_'+n+'.tif' for n in ('1467','1477','1483','1485','1593','1707','1761','1763','1773','1777','1789')}

def resultpath(r,s):return DEST/f"measurements/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
def probpath(r,tr,kind,mode):return DEST/f"inference/{mode}/{kind}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz"
def fieldpath(r,tr):return DEST/f"fields/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz"
def register():
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/'protocol.json'
    if p.exists():
        q=j(p)
        for f,h in q['protected'].items():assert sha(ROOT/f)==h,f
        return q
    q=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),plan_sha256=sha(ROOT/'thinking/ROUND43_BOUNDARY_INPUT_SUPPORT_PLAN_20261104.txt'),
        training=False,default_changed=False,subagents=False,groups=GROUPS,models=KINDS,boundary_seeds=SEEDS,scopes=SCOPES,
        split_sha256=sha(R41/'partitions.json'),protected={p.relative_to(ROOT).as_posix():sha(p) for folder,pat in [('scripts','*.py'),('configs','*.json')] for p in (ROOT/folder).glob(pat) if not p.name.startswith('round43')},
        main='ID-R; all469 primary denominator; component stratified paired bootstrap5000; no confirmation tuning')
    write(p,q);return q
