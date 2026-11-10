"""Round46 isolated, gate-first input/decoding experiment. No training."""
from scripts.round45_common import *
R45=DEST
DEST=ROOT/'runs/round46_input_decoding_v1'
GROUPS=('B','I','D','ID')
NEW46=('r46_gray_left','r46_gray_unequal','r46_bright_scale_gray')
SCOPES={**SCOPES,'r46_new':NEW46,'all26':tuple(SCOPES['all23'])+NEW46}

def resultpath(r,s):return DEST/f"measurements/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"

def register():
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/'protocol.json'
    if p.exists():
        q=j(p)
        for f,h in q['protected'].items():assert sha(ROOT/f)==h,f
        return q
    files=[p for folder in ('scripts','configs','src') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.toml') and not p.name.startswith('round46') and '__pycache__' not in p.parts]
    q=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),plan_sha256=sha(ROOT/'thinking/ROUND46_INPUT_DECODING_PLAN_20261105.txt'),training=False,default_changed=False,subagents=False,groups=GROUPS,kinds=KINDS,boundary_seeds=SEEDS,split_sha256=sha(R41/'partitions.json'),protected={p.relative_to(ROOT).as_posix():sha(p) for p in files},primary='ID-B; all469 denominator; historical331 component bootstrap5000; old16/gray and unseen combinations separate; no confirmation tuning',gate='No full inference or confirmation evaluation before crop protection and effective intervention controls pass',new_holdout_combinations=NEW46,locator_candidates=('C','T0','T1'),locator_parameters=dict(max_flat_dark=64,minimum_nonzero_flat_run=5,context_margin=.03),selection='zero boundary-reference clipping; synthetic protective controls; minimum worst inverse reference-width drift, p95, then fallback, C/T0/T1 lexical simplicity',status='controlled_checks')
    write(p,q);return q
