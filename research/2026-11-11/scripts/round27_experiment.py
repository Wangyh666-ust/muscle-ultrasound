"""Frozen, reference-blind Round27: two independent diagnostics, no training/inference."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.round26_1_common import OLD, NEW, j, sha, expected_inputs
from scripts.round26_2_provenance import verify_closeout_gate, prepare_run_output
from scripts.round26_1_evaluate import pred_map
from scripts.round26_1_legacy_regression import mmt
from scripts.round21_boundary_convention import (parse_dev, parse_external, read_mask,
    trace_for_A, make_fits_from_trace, evaluate_with_fits, selection_bundle)
from scripts.round25_boundary_fit import fit_curve
from src.umud_baseline.apo_baseline import measure_conservative, load_frozen_rule
from scripts.fascicle_geometry import extract_segments
from scripts.round27_geometry import (compact_fl, piecewise_fl, fixed_selection,
    correspond, mean_change_decomposition, aggregation_diagnostic)

BASE = ROOT / 'runs/geometry_eval/round27_fl_attribution_v1'
SOURCES = ROOT / 'manifest/round27_sources.json'
CONFIG = ROOT / 'manifest/round27_fl_attribution_v1.json'
PREV = ROOT / 'runs/round26_2_closeout_v1'
APO_INPUT = ROOT / 'runs/geometry_eval/round24_apo_pair_reference_v1/inputs/prediction_input.json'
REFERENCE = ROOT / 'runs/geometry_eval/round24_apo_pair_reference_v1/inputs/reference_143.json'
CODE = ('scripts/round27_geometry.py', 'scripts/round27_experiment.py',
        'scripts/round27_evaluate.py', 'scripts/round25_boundary_fit.py',
        'scripts/round25_evaluate.py', 'scripts/round20_evaluate.py',
        'scripts/round21_boundary_convention.py', 'scripts/fascicle_pa_fl.py',
        'scripts/fascicle_geometry.py', 'scripts/fl_acceptance_policy.py',
        'scripts/candidate_validity.py', 'scripts/pair_selection.py',
        'src/umud_baseline/apo_baseline.py', 'src/umud_baseline/geometry.py')


def checked_output(path):
    p = Path(path).resolve()
    if ROOT == p or ROOT in p.parents:
        if p != BASE.resolve():
            raise ValueError('Round27 writes only its own new result directory inside project')
    return p


def clean(obj):
    if isinstance(obj, np.ndarray): return clean(obj.tolist())
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)): return [clean(v) for v in obj]
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, Path): return str(obj)
    return obj


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def unique(rows, key='image_id', count=None):
    d = {r[key]: r for r in rows}
    if len(d) != len(rows) or (count is not None and len(rows) != count):
        raise ValueError('duplicate or unexpected identity count: ' + key)
    return d


def check_equal(a, b, path=''):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys(): raise ValueError('replay keys: ' + path)
        for k in a: check_equal(a[k], b[k], path + '/' + str(k))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b): raise ValueError('replay length: ' + path)
        for i, (x, y) in enumerate(zip(a, b)): check_equal(x, y, path + '/' + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if b is None or not np.isclose(a, b, atol=1e-8, rtol=1e-12):
            raise ValueError('replay numeric: ' + path)
    elif a != b: raise ValueError('replay value: ' + path)


def verify_sources():
    doc = j(SOURCES)
    if sha(CONFIG) != doc['config_sha256']: raise ValueError('config hash mismatch')
    for rel, expected in doc['files'].items():
        if sha(ROOT / rel) != expected: raise ValueError('source hash mismatch: ' + rel)
    if set(doc['code']) != set(CODE): raise ValueError('code binding set mismatch')
    for rel, expected in doc['code'].items():
        if sha(ROOT / rel) != expected: raise ValueError('code hash mismatch: ' + rel)
    return doc


def boundary(mask, rule, height):
    ap = measure_conservative(mask, rule)
    fits = ap['fits'] if ap['status'] == 'candidate' else None
    if fits is None:
        return ap, {'fits': None, 'curve': None, 'points': [], 'failure': ap['reason']}
    trace = trace_for_A(mask, rule, ap)
    replay, reasons = make_fits_from_trace(trace, fits, edge=False)
    if reasons or replay is None: raise ValueError('F0 fit replay failed')
    for k in ('sup', 'deep', 'support', 'paired_x', 'max_sample_gap'):
        check_equal(clean(fits[k]), clean(replay[k]), k)
    points = [(g['x'], g['sup']['median'], g['deep']['median']) for g in trace['grid']]
    curve, reason = fit_curve(points, fits['support'], height, fits['max_sample_gap'])
    return ap, {'fits': fits, 'curve': curve, 'points': points, 'failure': reason,
                'pair_source': ap['source'], 'support': fits['support'],
                'paired_x': fits['paired_x'], 'max_sample_gap': fits['max_sample_gap']}


def one_arm(inp, apo, bounds, mask, expected, mask_sha):
    fits = bounds['fits']; segments = extract_segments(mask)['segments']
    raw = evaluate_with_fits(segments, fits, fits is not None, inp['width'], inp['height'], inp['image_id'])
    sel = selection_bundle(raw)
    ids = {'PA': [int(c['segment']['label_id']) for c in sel['g0']['pa']],
           'G0': sel['g0_ids'], 'G1': sel['g1_ids']}
    vals = {'MT': mmt(fits, apo), 'PA': sel['g0']['pa_mean_deg'],
            'G0': sel['g0']['fl_mean_px'], 'G1': sel['g1']['fl_mean_px']}
    if inp['px_per_mm']:
        vals = {k: v / inp['px_per_mm'] if k != 'PA' and v is not None else v for k, v in vals.items()}
    invalid = {int(c['segment']['label_id']) for c in raw if c.get('candidate_validity_status') == 'invalid'}
    stage = ('apo_failure' if fits is None else 'no_evaluable_fascicle' if not raw else
             'no_finite_intersection' if not any(c.get('fl_px_raw') is not None for c in raw) else
             'all_rejected' if vals['G1'] is None else 'output')
    baseline = {'values': vals, 'selected_ids': ids,
        'selected_invalid': {k: sorted(set(v) & invalid) for k, v in ids.items()},
        'stage': stage, 'candidate_count': len(raw), 'image_id': inp['image_id'],
        'group_id': inp['group_id'], 'width': inp['width'], 'height': inp['height'],
        'px_per_mm': inp['px_per_mm'], 'fasc_mask_sha256': mask_sha,
        'apo_mask_sha256': sha(inp['apo_mask'])}
    check_equal(baseline, expected, inp['image_id'])
    candidates = []
    for c in raw:
        segment = c['segment']; label = int(segment['label_id'])
        c1 = piecewise_fl(segment, bounds['curve'], fits, inp['width'], inp['height'])
        candidates.append({'label_id': label, 'segment': segment,
            'selected': {k: label in v for k, v in ids.items()},
            'invalid': label in invalid,
            'invalid_conditions': c.get('candidate_validity_conditions'),
            'F0': compact_fl(c), 'F1': c1})
    fixed = {p: {f: fixed_selection(candidates, ids[p], f, p) for f in ('F0', 'F1')}
             for p in ('G0', 'G1')}
    return {'baseline': baseline, 'candidates': candidates, 'fixed': fixed,
            'aggregation': aggregation_diagnostic(candidates, ids['G1'])}


def selection_diagnosis(a, b):
    c0, c1 = a['candidates'], b['candidates']
    m = correspond([x['segment'] for x in c0], [x['segment'] for x in c1])
    decomp = mean_change_decomposition(c0, c1, a['baseline']['selected_ids']['G1'],
                                      b['baseline']['selected_ids']['G1'], m['pairs'])
    return {'correspondence': m, 'decomposition': decomp,
            'F1_used': False, 'expert_values_used': False}


def predict(out):
    src = verify_sources()
    verify_closeout_gate(labels=False)
    identities = expected_inputs()
    rule, rule_sha = load_frozen_rule()
    pm = {'C0': pred_map(OLD / 'predictions/C0'), 'C1': pred_map(NEW / 'predictions/C1_corrected')}
    emitted = []
    for ds, inputs, refpath in (
        ('dev282', parse_dev(), PREV / 'evaluation/dev282_measurement.json'),
        ('external34', parse_external(), PREV / 'evaluation/external34_measurement_predictions.json')):
        original = j(refpath)['rows']
        expected = {a: unique(original[a], count=len(inputs)) for a in ('C0', 'C1')}
        unique(inputs, count=282 if ds == 'dev282' else 34)
        rows = []
        for n, s in enumerate(inputs):
            apo_mask = read_mask(s['apo_mask'])
            if apo_mask.shape != (s['height'], s['width']): raise ValueError('apo dimensions mismatch')
            ap, bound = boundary(apo_mask, rule, s['height'])
            ident = identities[(ds, s['image_id'])]
            rec = {k: s[k] for k in ('image_id', 'group_id', 'width', 'height', 'px_per_mm')}
            rec.update(dataset=ds, image_path=ident['image_path'], bounds=bound,
                       apo_status=ap['status'], apo_reason=ap['reason'], arms={})
            for arm in ('C0', 'C1'):
                path = pm[arm][(ds, s['image_id'])]
                mask = read_mask(path)
                if mask.shape != apo_mask.shape: raise ValueError('fasc dimensions mismatch')
                rec['arms'][arm] = one_arm(s, ap, bound, mask, expected[arm][s['image_id']], sha(path))
            rec['selection_diagnosis'] = selection_diagnosis(rec['arms']['C0'], rec['arms']['C1'])
            rows.append(rec)
            if (n+1) % 40 == 0: print(ds, n+1, flush=True)
        path = out / 'predictions' / (ds + '.json'); write(path, rows); emitted.append(path)
    # Independent apo validation set. Same frozen R pair; never import historical J pairs.
    apo_rows = []
    for r in j(APO_INPUT):
        mask = read_mask(ROOT / r['apo_prediction_mask_path'])
        ap, bound = boundary(mask, rule, r['image_height'])
        apo_rows.append({'image_id': r['image_id'], 'group_id': r['group_id'],
                         'image_path': r['image_path'], 'width': r['image_width'],
                         'height': r['image_height'], 'apo_status': ap['status'], 'bounds': bound})
    unique(apo_rows, count=153)
    if sum(r['apo_status'] == 'candidate' for r in apo_rows) != 147: raise ValueError('R apo replay count')
    path = out / 'predictions/apo153.json'; write(path, apo_rows); emitted.append(path)
    # No evaluator, workbook, or reference arrays have been read above.
    write(out / 'predictions/manifest.json', {
        'config_sha256': sha(CONFIG), 'sources_sha256': sha(SOURCES), 'rule_sha256': rule_sha,
        'files': {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in emitted},
        'expert_values_read': False, 'human_reference_arrays_read': False,
        'baseline_replay': '632/632 full per-image measurement records identical',
        'counts': {'dev_images': 282, 'dev_groups': 279, 'external': 34, 'apo': 153},
    })
    verify_sources()
    write(out / 'prediction_success.json', {'manifest_sha256': sha(out / 'predictions/manifest.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE)
    args = parser.parse_args()
    out = prepare_run_output(checked_output(args.out))
    for name in ('prediction_success.json', 'evaluation_success.json'):
        (out / name).unlink(missing_ok=True)
    try: predict(out)
    except Exception as e:
        write(out / 'validation_failed.json', {'error': str(e)})
        raise


if __name__ == '__main__': main()
