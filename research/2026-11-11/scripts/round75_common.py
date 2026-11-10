"""Round75 isolated paths and registration; no imports or edits to the default pipeline."""
from pathlib import Path
from scripts.round63_common import ROOT,j,sha,write,readgz,clean,np
OUT=ROOT/'runs/round75_density_measurement_v1'
V112=ROOT/'runs/validation112_current_20261110'
SEEDS=(20261023,20261024)
VIEWS=('identity','translate2_-1','brightness105','scale1015')

def records():return j(OUT/'protocol.json')['rows']

def prepare():
    prior=j(V112/'protocol.json');rows=prior['rows'].copy()
    md=j(ROOT/'runs/frozen_boundary_20261110/source_round63_boundary_paths_v1_metadata.json')
    for key,m in md.items():
        row=m['row'];ds=row['dataset']
        if ds=='external34':continue
        rows.append(dict(dataset=ds,image_id=row['image_id'],image_path=row['image_path'],group=m['group'],
             px_per_mm=m['px_per_mm'],width=row['width'],height=row['height'],image_sha256=sha(ROOT/row['image_path']),
             reference=None,reference_class='boundary_reference_only' if ds=='apo153' else 'unlabelled_similarity',assisted_MT=False))
    assert len(rows)==547
    sources={p:sha(ROOT/p) for p in ['thinking/ROUND75_BOUNDARY_MEASUREMENT_REPAIR_PLAN_20261110.txt',
       'scripts/round63_paths.py','scripts/round68_geometry.py','scripts/round74_geometry.py','scripts/round64_measure_adapter.py',
       'scripts/round48_length_replay.py','scripts/round40_core.py','runs/round40_fl_qualification_v1/calibration.json'] if (ROOT/p).exists()}
    write(OUT/'protocol.json',dict(round=75,status='registered_before_controls',training=False,default_changed=False,subagents=False,
       rows=rows,models=prior['models'],protected=sources,originals=547,measurement_references=112,
       perturbation_scope='all112 measurement-reference images; identity additionally all435 historical boundary/similarity images',
       views=VIEWS,seeds=SEEDS,boundary_seed=20261023,
       S='only C becomes C*D in existing .20*(C-.5); other score terms unchanged',
       Q='exact boundary geometry + independent qualified support, original formal aggregation and FL eligibility retained',
       F='relative-FL sensitivity reweighting; no fixed FL clipping and no sample deletion',
       arms=['B','S','Q','F','SQF_if_controls_pass'],
       controls='48 previously reviewed training sources for calibration, remaining624 training sources for frozen confirmation; controlled geometric tests are not model generalization',
       independence='112 already inspected: regression only, never a new blind test',
       gates=dict(S_normal_change=0,S_density_positive_min=.8,Q_normal_numeric_tolerance=1e-7,Q_recoverable_support_required=.95,
          F_unperturbed_reference_change_max_relative=.01,F_damaged_error_improvement_required=True),
       fallback='Failed mechanism does not enter combined arm; failed control is disclosed, not silently retuned on112'))
    print('R75_REGISTERED_547',flush=True)

if __name__=='__main__':prepare()
