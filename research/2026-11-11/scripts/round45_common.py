"""Round45 isolated measurement-frame experiment; previous runs stay frozen."""
from scripts.round44_common import *
R44=DEST
DEST=ROOT/'runs/round45_measurement_frame_v1'
GROUPS=('H','B','D','Q','DQ','G')

def resultpath(r,s):return DEST/f"measurements/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"

def register():
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/'protocol.json'
    if p.exists():
        q=j(p)
        for f,h in q['protected'].items():assert sha(ROOT/f)==h,f
        return q
    files=[p for folder in ('scripts','configs','src') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.toml') and not p.name.startswith('round45') and '__pycache__' not in p.parts]
    q=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),plan_sha256=sha(ROOT/'thinking/ROUND45_MEASUREMENT_FRAME_EXECUTION_PLAN_20261104.txt'),training=False,default_changed=False,subagents=False,groups=GROUPS,kinds=KINDS,boundary_seeds=SEEDS,scopes=SCOPES,split_sha256=sha(R41/'partitions.json'),protected={p.relative_to(ROOT).as_posix():sha(p) for p in files},primary='DQ-B; combined16 and new_gray; all469; historical331; component bootstrap5000; no confirmation tuning')
    write(p,q);return q
