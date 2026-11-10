"""Round47 frozen-input boundary x actual-contributor selection experiment."""
from scripts.round46_common import *
R46=DEST
DEST=ROOT/'runs/round47_boundary_selection_v1'
GROUPS=('R','E','S','ES')

def resultpath(r,s):return DEST/f"measurements/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"

def register():
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/'protocol.json'
    if p.exists():
        q=j(p)
        for f,h in q['protected'].items():assert sha(ROOT/f)==h,f
        return q
    checkpoint=ROOT/'thinking/evidence/checklist_at_round46_completion_20261105.txt'
    source=ROOT/'thinking/things_for_considering.txt'
    if not checkpoint.exists():checkpoint.write_bytes(source.read_bytes())
    assert sha(checkpoint)==j(R46/'success.json')['checklist_sha256']
    files=[p for folder in ('scripts','configs','src') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.toml') and not p.name.startswith('round47') and '__pycache__' not in p.parts]
    q=dict(round=47,created=time.strftime('%Y-%m-%d %H:%M:%S'),training=False,default_changed=False,subagents=False,groups=GROUPS,kinds=KINDS,boundary_seeds=SEEDS,development=138,historical_confirmation=331,split_sha256=sha(R41/'partitions.json'),protected={p.relative_to(ROOT).as_posix():sha(p) for p in files},baseline='Exact Round46 ID; same network inputs/masks, no neural inference planned',primary='ES-R; combined16 and all26; component bootstrap5000 on historical331; R/E/S/ES separate; missing stays in denominator',selection_metrics='Actual observed spatial mass, third coverage, effective contributors, physically matched weight turnover, 1D transport of observed contributor mass; no ID equality matching',gate='Synthetic boundary and selection checks, development replay before lock; no confirmation-dependent retuning',scope=SCOPES,limits='Historical331 repeatedly used, not blind external validation. Similar pairs may contain real anatomical differences. No forced three-fascicle availability.')
    write(p,q);return q
