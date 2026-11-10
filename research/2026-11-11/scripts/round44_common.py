"""Isolated Round44 experiment; immutable previous rounds and defaults."""
from scripts.round43_common import *
R43=DEST
DEST=ROOT/'runs/round44_region_evidence_v1'
GROUPS=('H','B','L','E','LE')
NEW44=('r44_gray04','r44_gray32_asym','r44_bright_gray','r44_scale_gray')
OLD_SCOPES=dict(SCOPES)
SCOPES={**SCOPES,'r44_new':NEW44,'all19':tuple(OLD_SCOPES['combined16'])+NEW43,'all23':tuple(OLD_SCOPES['combined16'])+NEW43+NEW44}

def resultpath(r,s):return DEST/f"measurements/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
def probpath(r,tr,kind,mode):return DEST/f"inference/{mode}/{kind}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz"
def register():
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/'protocol.json'
    if p.exists():
        q=j(p)
        for f,h in q['protected'].items():assert sha(ROOT/f)==h,f
        return q
    src=[p for folder in ('scripts','configs','src') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.toml') and not p.name.startswith('round44') and '__pycache__' not in p.parts]
    q=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),plan_sha256=sha(ROOT/'thinking/ROUND44_REGION_CONTEXT_EVIDENCE_PLAN_20261104.txt'),
        training=False,default_changed=False,subagents=False,groups=GROUPS,models=KINDS,boundary_seeds=SEEDS,scopes=SCOPES,
        split_sha256=sha(R41/'partitions.json'),protected={p.relative_to(ROOT).as_posix():sha(p) for p in src},main='LE-B; old16 and gray16; all469 denominator; bootstrap5000 historical331; no confirmation tuning')
    write(p,q);return q
