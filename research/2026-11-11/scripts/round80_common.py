"""Registered, opt-in connection/ranking/FL-fidelity experiment."""
from scripts.round43_cpu_imports import install
install()
from pathlib import Path
from scripts.round63_common import ROOT, j, sha, write, readgz, np
OUT=ROOT/'runs/round80_connection_fidelity_v1'
R77=ROOT/'runs/round77_subject_layer_v1'
R78=ROOT/'runs/round78_mt_geometry_adapter_v1'
R79=ROOT/'runs/round79_fl_weighting_v1'
AUDIT=ROOT/'review/boundary_connection_audit_20261110'
SEEDS=('20261023','20261024')
ARMS=('B','C','R','G','CR','CG','RG','CRG')


def register():
    if (OUT/'protocol.json').exists():return j(OUT/'protocol.json')
    manifest=j(ROOT/'runs/round60_boundary_objective_v1/payload/training_manifest.json')
    reviewed={r['image_id'] for r in j(ROOT/'thinking/ROUND62_PILOT_REVIEW_QUEUE_20261109.json')['cases']}
    samples=manifest['samples'];assert len(samples)==672
    bindings=[]
    for s in samples:
        for v in s['variants']:
            path=ROOT/'runs/round60_boundary_objective_v1/payload'/v['path'];assert sha(path)==v['sha256']
        bindings.append(dict(id=s['image_id'],group=s['group_id'],phase='calibration' if s['image_id'] in reviewed else 'confirmation',variants=s['variants']))
    assert sum(r['phase']=='calibration' for r in bindings)==48
    protected=[p for p in (ROOT/'src').rglob('*.py')]
    protected += [ROOT/'scripts'/n for n in ('round63_paths.py','round64_seams.py','round68_geometry.py','round71_work.py','round74_geometry.py','round77_layer.py','round64_measure_adapter.py','round76_core.py','round78_mt_core.py','round79_core.py')]
    p=dict(round=80,status='registered_before_control_results',training=False,new_inference=False,default_changed=False,subagents=False,
           plan_sha=sha(ROOT/'thinking/BOUNDARY_CONNECTION_AND_FL_PLAN_20261110.txt'),audit_summary=j(AUDIT/'summary.json'),
           controls=bindings,source_controls='real reviewed label centerlines; controlled probability/fragment corruption, NOT a model-generalization test',
           grids=dict(C=[dict(endpoint_window=w,required_probability_hit=h) for w in (.12,.20) for h in (.5,.8)],R=[{}],G=[{}]),
           fixed_C=dict(trim_fractions=[0.,.02,.04,.08,.12],maximum_retreat=.12,minimum_stable_span=.08,minimum_bridge_probability=.10,maximum_bridge_blind_run=.02,
                        stable_trend_limit_deg=25.,stable_holdout_error_grid=2.,max_new_paths=10,max_chain=4),
           C_gate=dict(normal_originals_unchanged=True,negative_false_connections=0,positive_correct_coverage_mean_gain_min=.05,accepted_bridge_correct_fraction_min=.95),
           R_gate=dict(no_protected_wrong_band=True,unknown_not_claimed_absent=True,positive_correct_rate_min=.8,normal_selection_preserved=True),
           G_gate=dict(finite_geometry_exact=True,rays_unchanged=True,known_shape_MAE_no_increase=True,new_missing_FL=0),
           G='Only existing-qualified samples inside legacy finite support; unknown runs stay legacy. Outside rays and support interval retained exactly. Original FL contributors/weights fixed; MT and PA copied.',
           scopes=dict(originals=547,views=995,source_boundary_references=221,scalar_references=112,C1_seeds=SEEDS,boundary_model='C62 C seed23 epoch06',natural_pairs=117,strict_pairs=18),
           admission='Each module must pass calibration and frozen confirmation. Failed modules do not enter full combinations; no retuning on112 or named cases.',
           arms_if_all_pass=list(ARMS),baseline='current retained R78M / R79B; NOT R79 SE or previous Q',
           protected={p.relative_to(ROOT).as_posix():sha(p) for p in protected},historical_regression_not_blind=True)
    write(OUT/'protocol.json',p);return p


def verify_default():
    for rel,h in j(OUT/'protocol.json')['protected'].items():assert sha(ROOT/rel)==h,rel

