"""Round64-only paths and pre-evaluation registration."""
from scripts.round63_common import ROOT, R62, MODEL, SIDES, j, sha, clean, write, readgz, array_sha, stats
from scripts.round63_common import OUT as R63
import numpy as np
from pathlib import Path

OUT=ROOT/'runs/round64_boundary_continuation_v1'
ARMS=('D','R','Q','B','S','X','BS','BSX')


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    files=[ROOT/'thinking/ROUND64_BOUNDARY_CONTINUATION_PLAN_20261109.txt',
           ROOT/'configs/round64_boundary_continuation_plan.json',
           ROOT/'scripts/round63_paths.py',ROOT/'scripts/round63_measure.py',
           R63/'inputs.json',R63/'metadata.json',R63/'lock.json',MODEL]
    q=dict(status='executing_controlled_stage_not_adopted',training=False,default_changed=False,subagents=False,
           protected={p.relative_to(ROOT).as_posix():sha(p) for p in files},
           source_model='Round62 C 20261023 epoch06',source_c1='B_20261023',
           views=1470,originals=469,manual_images=143,arms=ARMS,
           gates=j(ROOT/'configs/round64_boundary_continuation_plan.json')['prespecified_engineering_targets'])
    if path.exists():assert j(path)==clean(q),'registration changed'
    else:write(path,q)
    return q


def parameters():
    return dict(band_low=.18,band_high=.5,band_nodes=9,dp_stride=128,
                band_direction_weight=.01,band_curvature_weight=.00001,band_image_weight=.20,
                seam_trim_fractions=[0.,.02,.04],extension_fraction=.08,
                extension_decay=.04,extension_beam=3,extension_bonus=.4,
                max_step_slope=1.5,source='development_values_not_yet_locked')
