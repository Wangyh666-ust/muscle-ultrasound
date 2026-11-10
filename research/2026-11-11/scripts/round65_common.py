"""Bindings, registration and pre-declared finite parameter grids."""
from scripts.round64_common import ROOT,R62,R63,MODEL,SIDES,j,sha,write,readgz,array_sha,clean,stats,np,Path
R64=ROOT/'runs/round64_boundary_continuation_v1'
OUT=ROOT/'runs/round65_local_repair_v1'
ARMS=('R','S','N','N_T','N_P','N_T_P','N_E','N_T_P_E')


def register():
    OUT.mkdir(exist_ok=True)
    paths=[ROOT/'thinking/ROUND65_LOCAL_REPAIR_PLAN_20261109.txt',ROOT/'configs/round65_local_repair_plan.json',
        ROOT/'scripts/round63_paths.py',ROOT/'scripts/round63_measure.py',
        ROOT/'scripts/round64_seams.py',ROOT/'scripts/round64_continue.py',
        ROOT/'scripts/round65_common.py',R63/'inputs.json',R63/'metadata.json',
        R64/'controls/S_confirmation_lock.json',R64/'bs_interaction_audit.json',MODEL,
        ROOT/'runs/round60_boundary_objective_v1/payload/training_manifest.json',
        ROOT/'thinking/ROUND62_PILOT_REVIEW_QUEUE_20261109.json']
    paths.extend(sorted((ROOT/'src').rglob('*.py')))
    protocol=dict(round=65,training=False,default_changed=False,subagents=False,
        protected={p.relative_to(ROOT).as_posix():sha(p) for p in paths},arms=ARMS,
        direction='no left/right preference; horizontal-reflection controls on both sides',
        known_cases_not_independent=['image_1011.tif','image_0405.tif','image_0612.tif','image_0685.tif','image_0197.tif','image_0174.tif'],
        grid=dict(trim_fraction=[.04,.08,.12,.16],anchor_fraction=[.06,.10],
                  curvature_cost=[.025,.10],extension_fraction=[.04,.08,.12],
                  replacement_margin=[0.,.025,.05]),
        control_target_each_side=.95,false_reliable_links=0,outside_window_violations=0,
        modules_independent=True,confirmation_must_not_retune=True,
        model_seed_count=1,source_model='Round62 C 20261023 epoch06',source_C1='B_20261023')
    path=OUT/'protocol.json'
    if path.exists():assert j(path)==clean(protocol),'registration changed'
    else:write(path,protocol)
    return protocol


def defaults():
    return dict(low=.18,high=.5,anchor_fraction=.10,trim_fraction=.12,
        local_window=.20,curvature_cost=.025,extension_fraction=.08,
        replacement_margin=.025,source='development_unlocked',
        # Same physical scale in either horizontal orientation.
        nominal_width=1046.,noise_pixels=2.,max_track_turn=1.5,
        direction_test_required=True)


if __name__=='__main__':
    q=register();print('ROUND65_REGISTERED',len(q['protected']),'protected files',flush=True)
