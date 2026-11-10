"""Bindings and immutable registration for the authorized Round68 experiment."""
from scripts.round65_common import ROOT,R63,R64,R62,MODEL,np,j,sha,write,readgz,array_sha,Path,SIDES
OUT=ROOT/'runs/round68_boundary_paths_v1'
R67=ROOT/'runs/round67_local_evidence_v1'
ARMS=('S','W','J','X','WJ','WX','JX','WJX')

def register():
    paths=[ROOT/'thinking/ROUND68_BOUNDARY_PATH_PLAN_20261109.txt',ROOT/'configs/round68_boundary_path_plan.json',
        MODEL,R63/'inputs.json',R63/'metadata.json',R63/'config.json',R67/'parameter_lock.json',
        R67/'summary.json',R67/'execution_state.json',R64/'controls/S_confirmation_lock.json']
    paths+=list((ROOT/'src').rglob('*.py'))
    for f in ('round68_geometry.py','round68_controls.py','round68_work.py','round68_evaluate.py',
              'round63_paths.py','round64_seams.py','round65_core.py','round65_measure.py',
              'round64_measure_adapter.py','round66_geometry.py','round67_geometry.py'):
        paths.append(ROOT/'scripts'/f)
    obj=dict(training=False,default_changed=False,subagents=False,arms=ARMS,history='RX67',
        protected={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        boundary_seed='C_20261023_epoch06',C1='B_20261023_frozen',status='registered_before_controls',
        full_scope=1470,GT_algorithm_input=False,known_cases_not_for_tuning=['image_0960.tif','image_0405.tif','image_1011.tif'])
    p=OUT/'protocol.json'
    if p.exists():assert j(p)==j_clean(obj),'registered implementation changed'
    else:write(p,obj)
    return obj

def j_clean(obj):
    from scripts.round63_common import clean
    return clean(obj)

def configs():
    return dict(W=[dict(min_width_sigma=v) for v in (4.,6.)],
        J=[dict(max_retreat=r,improvement=i) for r in (.12,.20) for i in (.25,.50)],
        X=[dict(support_fraction=v) for v in (.85,.95)])

def normalize(p,field):
    from scripts.round67_geometry import original_with_confidence
    if p is None:return None
    q={**p}
    for k in ('x','y','observed','kind'):q[k]=np.asarray(q[k],bool if k=='observed' else float if k in ('x','y') else int)
    q['ids']=tuple(q['ids'])
    return original_with_confidence(q,field)

def signature(p,observed_only=False):
    if p is None:return 'none'
    use=p['observed'] if observed_only else np.ones(len(p['x']),bool)
    return array_sha(np.stack([p['x'][use],p['y'][use]]))

def main():print('ROUND68_REGISTERED',len(register()['protected']),flush=True)
