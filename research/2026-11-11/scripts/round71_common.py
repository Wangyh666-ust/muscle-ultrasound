"""Isolated whole-field completion experiment; no model or default changes."""
from scripts.round63_common import ROOT,R62,MODEL,SIDES,np,j,sha,write,readgz,array_sha,Path,stats
R63=ROOT/'runs/round63_boundary_paths_v1'
R68=ROOT/'runs/round68_boundary_paths_v1'
R69=ROOT/'runs/round69_measurement_support_v1'
R70=ROOT/'runs/round70_WJ_fidelity_v1'
OUT=ROOT/'runs/round71_full_boundary_completion_v1'
ARMS=('B','G','E','GE')
GRID=[dict(decay_width=v) for v in (.10,.25)]

def register():
    files=[ROOT/'thinking/ROUND71_FULL_COMPLETION_PLAN_20261109.txt',ROOT/'thinking/FULL_BOUNDARY_COMPLETION_REQUIREMENTS_20261109.txt',
           R63/'inputs.json',R63/'metadata.json',R68/'parameter_lock.json',R70/'delivery.json',MODEL]
    files+=list((ROOT/'src').rglob('*.py'))
    files += [ROOT/'scripts'/n for n in ('round71_common.py','round71_geometry.py','round71_roi.py','round71_controls.py','round70_geometry.py','round66_geometry.py')]
    obj=dict(round=71,revision=2,predecessor='audit/v1_anchor_fallback',training=False,default_changed=False,subagents=False,GT_algorithm_input=False,arms=ARMS,grid=GRID,
       sources=48,confirmation_sources=624,views=1470,originals=469,manual_originals=143,
       protected={p.relative_to(ROOT).as_posix():sha(p) for p in files},
       baseline='frozen R68 W + archived R68 WJ proposal accepted only by R70 edit guard; no alternative proposal search',
       candidate_confidence='R69 frozen W sampled fused probabilities; original J winner/weighting not changed',
       selection='training controls only; freeze before development scoring; failed modules full shadow diagnostic, not adopted',
       domain='image-only texture component hull, registered before reference scoring; ambiguity explicitly reported',
       scope='one frozen boundary seed, C1 frozen and affine mapped; no robustness claim for other seeds')
    p=OUT/'protocol.json'
    if p.exists():assert j(p)==obj,'registered source changed'
    else:write(p,obj)
    return obj
