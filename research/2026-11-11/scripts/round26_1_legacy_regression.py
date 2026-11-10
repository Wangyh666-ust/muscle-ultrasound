"""A3 regression: corrected old-C0/legacy-C1 statistics and measurements."""
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.umud_baseline.apo_baseline import load_frozen_rule, measure_conservative
from src.umud_baseline.data import UmudTrainData, align_mask_to_image, read_mask
from scripts.fascicle_geometry import extract_segments, match_segments
from scripts.round21_boundary_convention import (
    evaluate_with_fits, mt_value, parse_dev, parse_external, read_mask as read_mask_pil,
    selection_bundle,
)
from scripts.round20_evaluate import (
    build_expert_reference, parse_workbook, summarize_errors,
)
from scripts.round26_1_common import (
    OLD, NEW, ROOT, CONFIG_OLD, OLD_C0_CKPT, OLD_C0_SHA,
    OLD_LEGACY_C1_CKPT, OLD_LEGACY_C1_SHA, ISOLATION,
    expected_inputs, j, sha, write_json, validate_prediction_arm,
)
from scripts.evaluate_fasc_structure import PRIMARY

EXPERT = ROOT / 'external/umud_expert/expert_measurements.json'
WORKBOOK = ROOT / ('external/umud_expert/unpacked/'
                   'benchmark_dataset_architecture_v0.1.0/'
                   'Results_benchmark_architecture_v0.1.0.xlsx')


def pred_map(arm):
    m = j(OLD / f'predictions/{arm}/manifest.json')
    return {(r['dataset'], r['image_id']): OLD / f'predictions/{arm}' / r['mask_path']
            for r in m['records']}


def mmt(fits, apo):
    pos = (apo.get('result') or {}).get('x_positions') or []
    if not pos or fits is None:
        return None
    values, _ = mt_value(fits, pos)
    return float(np.mean(values)) if values else None


def arm_geometry(apo, fasc, fits, width, height, image_id):
    segments = extract_segments(fasc)['segments']
    cands = evaluate_with_fits(segments, fits, fits is not None, width, height, image_id)
    sel = selection_bundle(cands)
    ids = {'PA': [int(c['segment']['label_id']) for c in sel['g0']['pa']],
           'G0': sel['g0_ids'], 'G1': sel['g1_ids']}
    invalid = {int(c['segment']['label_id']) for c in cands
               if c.get('candidate_validity_status') == 'invalid'}
    vals = {'MT': mmt(fits, apo),
            'PA': sel['g0']['pa_mean_deg'],
            'G0': sel['g0']['fl_mean_px'],
            'G1': sel['g1']['fl_mean_px']}
    if fits is None:
        stage = 'apo_failure'
    elif not cands:
        stage = 'no_evaluable_fascicle'
    elif not any(c.get('fl_px_raw') is not None for c in cands):
        stage = 'no_finite_intersection'
    elif vals['G1'] is None:
        stage = 'all_rejected'
    else:
        stage = 'output'
    return {'values': vals, 'selected_ids': ids,
            'selected_invalid': {k: sorted(set(v) & invalid) for k, v in ids.items()},
            'stage': stage, 'candidate_count': len(cands)}


def measurement(name):
    rule, rule_hash = load_frozen_rule()
    preds = {'C0': pred_map('C0'), 'C1': pred_map('C1')}
    inputs = parse_dev() if name == 'dev282' else parse_external()
    rows = {'C0': [], 'C1': []}
    apo_hashes = {}
    for i, r in enumerate(inputs):
        apo = read_mask_pil(r['apo_mask'])
        ap = measure_conservative(apo, rule)
        fits = ap.get('fits') if ap.get('status') == 'candidate' else None
        apo_hashes[r['image_id']] = sha(r['apo_mask'])
        for arm in ('C0', 'C1'):
            p = preds[arm][(name, r['image_id'])]
            pred = read_mask(p)
            if pred.shape != apo.shape:
                raise ValueError(f'{arm} {r["image_id"]}: mask shape mismatch')
            a = arm_geometry(ap, pred, fits, r['width'], r['height'], r['image_id'])
            if name == 'external34' and r.get('px_per_mm'):
                for k in ('MT', 'G0', 'G1'):
                    if a['values'].get(k) is not None:
                        a['values'][k] = a['values'][k] / float(r['px_per_mm'])
            a.update({'image_id': r['image_id'], 'group_id': r['group_id'],
                      'width': int(r['width']), 'height': int(r['height']),
                      'px_per_mm': r['px_per_mm'],
                      'fasc_mask_sha256': sha(p),
                      'apo_mask_sha256': apo_hashes[r['image_id']]})
            rows[arm].append(a)
        if (i + 1) % 60 == 0:
            print(name, 'geometry', i + 1, flush=True)
    summaries = {}
    for arm in ('C0', 'C1'):
        rs = rows[arm]
        groups = defaultdict(list)
        for r in rs:
            groups[r['group_id']].append(r['values']['G1'] is not None)
        summaries[arm] = {
            'n': len(rs),
            'coverage': {p: sum(r['values'][p] is not None for r in rs)
                         for p in ('MT', 'PA', 'G0', 'G1')},
            'G1_group_equal': float(np.mean([np.mean(v) for v in groups.values()])),
            'G1_ge3': sum(len(r['selected_ids']['G1']) >= 3 for r in rs),
            'stages': {k: sum(r['stage'] == k for r in rs) for k in
                       ('apo_failure', 'no_evaluable_fascicle',
                        'no_finite_intersection', 'all_rejected', 'output')},
            'invalid_counts': {p: sum(len(r['selected_invalid'][p]) for r in rs)
                               for p in ('PA', 'G0', 'G1')},
            'selected_counts': {p: sum(len(r['selected_ids'][p]) for r in rs)
                                for p in ('PA', 'G0', 'G1')},
        }
    a0 = {r['image_id']: r for r in rows['C0']}
    a1 = {r['image_id']: r for r in rows['C1']}
    changes = {}
    for p in ('MT', 'PA', 'G0', 'G1'):
        oldset = {i for i, r in a0.items() if r['values'][p] is not None}
        newset = {i for i, r in a1.items() if r['values'][p] is not None}
        changes[p] = {'new': sorted(newset - oldset),
                      'lost': sorted(oldset - newset),
                      'common': len(oldset & newset),
                      'common_mean_increase': sum(
                          a1[i]['values'][p] > a0[i]['values'][p]
                          for i in oldset & newset)}
    return {'rows': rows, 'summary': summaries, 'changes': changes,
            'rule_sha256': rule_hash}


def paired_dev():
    inc = set(j(ISOLATION)['included_filenames'])
    data = UmudTrainData('fasc', 'dev', include_filenames=inc)
    preds = {'C0': pred_map('C0'), 'C1': pred_map('C1')}
    per_variant = []
    for s in data.samples:
        iid = s['image']['filename']
        masks = {}
        for arm in ('C0', 'C1'):
            masks[arm] = read_mask(preds[arm][('dev282', iid)])
        human_wh = (int(s['image']['width']), int(s['image']['height']))
        for vi, mrec in enumerate(s['masks']):
            human = align_mask_to_image(read_mask(ROOT / mrec['path']), human_wh)
            matches = {}
            pred_ids = {}
            for arm in ('C0', 'C1'):
                pe = extract_segments(masks[arm])
                he = extract_segments(human)
                mm = match_segments(pe['segments'], he['segments'], **PRIMARY)
                matches[arm] = {int(x['human_index']): x for x in mm['matches']}
                pred_ids[arm] = [int(seg['label_id']) for seg in pe['segments']]
            common = sorted(set(matches['C0']) & set(matches['C1']))
            obs = []
            for k in common:
                obs.append({'human_component_id': k,
                            'C0_pred_component_id': pred_ids['C0'][matches['C0'][k]['pred_index']],
                            'C1_pred_component_id': pred_ids['C1'][matches['C1'][k]['pred_index']],
                            'C0_angle_diff_deg': matches['C0'][k]['angle_diff_deg'],
                            'C1_angle_diff_deg': matches['C1'][k]['angle_diff_deg'],
                            'C0_perp_distance_px': matches['C0'][k]['perp_distance_px'],
                            'C1_perp_distance_px': matches['C1'][k]['perp_distance_px'],
                            'C0_midpoint_distance_px': matches['C0'][k]['midpoint_distance_px'],
                            'C1_midpoint_distance_px': matches['C1'][k]['midpoint_distance_px']})
            per_variant.append({
                'image_id': iid, 'group_id': s['group_id'], 'variant_index': vi,
                'human_mask_path': Path(mrec['path']).as_posix(),
                'n_common': len(obs), 'common': obs,
                'only_C0': [{'human_component_id': k,
                             'C0_pred_component_id': pred_ids['C0'][matches['C0'][k]['pred_index']],
                             'C0_angle_diff_deg': matches['C0'][k]['angle_diff_deg'],
                             'C0_perp_distance_px': matches['C0'][k]['perp_distance_px']}
                            for k in sorted(set(matches['C0']) - set(matches['C1']))],
                'only_C1': [{'human_component_id': k,
                             'C1_pred_component_id': pred_ids['C1'][matches['C1'][k]['pred_index']],
                             'C1_angle_diff_deg': matches['C1'][k]['angle_diff_deg'],
                             'C1_perp_distance_px': matches['C1'][k]['perp_distance_px']}
                            for k in sorted(set(matches['C1']) - set(matches['C0']))]})
    # Hierarchy: variant mean, image mean over variants, group mean over images.
    byimg = defaultdict(list)
    groups = {}
    for r in per_variant:
        byimg[r['image_id']].append(r)
        groups[r['image_id']] = r['group_id']
    metrics = {}
    for field in ('angle_diff_deg', 'perp_distance_px'):
        perimg = {}
        for iid, rs in byimg.items():
            vals = {a: [] for a in ('C0', 'C1')}
            for r in rs:
                obs = r['common']
                if obs:
                    for a in vals:
                        vals[a].append(float(np.mean([
                            x[f'{a}_{field}'] for x in obs])))
            if vals['C0']:
                perimg[iid] = {a: float(np.mean(v)) for a, v in vals.items()}
        grouped = defaultdict(list)
        for iid, v in perimg.items():
            grouped[groups[iid]].append(v)
        gv = {g: {a: float(np.mean([x[a] for x in vs]))
                  for a in ('C0', 'C1')} for g, vs in grouped.items()}
        metrics[field] = {
            'n_images': len(perimg), 'n_groups': len(gv),
            'image_equal': {a: float(np.mean([v[a] for v in perimg.values()]))
                            for a in ('C0', 'C1')},
            'group_equal': {a: float(np.mean([v[a] for v in gv.values()]))
                            for a in ('C0', 'C1')},
            'improved': sum(v['C1'] < v['C0'] for v in perimg.values()),
            'degraded': sum(v['C1'] > v['C0'] for v in perimg.values())}
    return {'per_variant': per_variant, 'summary': metrics,
            'observation_unit': ('human component within image-mask variant; '
                                 'not independent anatomy across variants'),
            'only_C1': sum(len(r['only_C1']) for r in per_variant),
            'only_C0': sum(len(r['only_C0']) for r in per_variant)}


def close(a, b, tol=1e-8):
    return abs(float(a) - float(b)) <= tol


def external_expert(meas):
    rows = meas['rows']
    ids = [r['image_id'] for r in rows['C0']]
    doc = j(EXPERT)
    if sha(WORKBOOK) != doc['source_sha256']:
        raise ValueError('expert workbook hash mismatch')
    ref = build_expert_reference(parse_workbook(WORKBOOK), set(ids))
    b = {a: {r['image_id']: r for r in rows[a]} for a in ('C0', 'C1')}
    out = {'formal_errors': {'status': 'blocked',
                             'reason': 'expert units/protocol unconfirmed'},
           'exploratory_assumed_units': {'MT': 'mm', 'PA': 'degree',
                                          'FL': 'mm'},
           'parameters': {}}
    for p in ('MT', 'PA', 'FL'):
        key = {'MT': 'MT', 'PA': 'PA', 'FL': 'G1'}[p]
        common = []
        for iid in ids:
            rv = ref[iid][p]['mean']
            p0 = b['C0'][iid]['values'][key]
            p1 = b['C1'][iid]['values'][key]
            if rv is not None and p0 is not None and p1 is not None:
                common.append((iid, p0, p1, rv))
        out['parameters'][p] = {
            'n_common': len(common),
            'coverage': {a: sum(b[a][i]['values'][key] is not None for i in ids)
                         for a in ('C0', 'C1')},
            'C0': summarize_errors([(i, x, r) for i, x, y, r in common]),
            'C1': summarize_errors([(i, y, r) for i, x, y, r in common]),
            'improved_ids': [i for i, x, y, r in common if abs(y-r) < abs(x-r) - 1e-12],
            'degraded_ids': [i for i, x, y, r in common if abs(y-r) > abs(x-r) + 1e-12],
            'new_coverage': [{'image_id': i, 'C1_error': abs(y-r)}
                             for i, x, y, r in common if False],
        }
        # Coverage changes must be computed on all finite prediction/reference
        # pairs, preserving failed images rather than silently dropping them.
        old = {i for i in ids if b['C0'][i]['values'][key] is not None}
        new = {i for i in ids if b['C1'][i]['values'][key] is not None}
        out['parameters'][p]['coverage_changes'] = {
            'new': sorted(new - old), 'lost': sorted(old - new)}
    return out


def main():
    expected = expected_inputs()
    gate = {}
    gate['C0'] = validate_prediction_arm(
        'C0', OLD / 'predictions/C0', expected, CONFIG_OLD, OLD_C0_CKPT,
        OLD_C0_SHA, OLD / 'training/C0/run_meta.json',
        OLD / 'predictions/C0/prediction_success.json')
    # Legacy C1 has arm C1 in the old metadata; label it explicitly as legacy.
    gate['legacy_C1'] = validate_prediction_arm(
        'C1', OLD / 'predictions/C1', expected, CONFIG_OLD,
        OLD_LEGACY_C1_CKPT, OLD_LEGACY_C1_SHA,
        OLD / 'training/C1/run_meta.json',
        OLD / 'predictions/C1/prediction_success.json', allow_legacy=True)
    dev_meas = measurement('dev282')
    ext_meas = measurement('external34')
    write_json(NEW / 'closeout/legacy_external_measurement_predictions.json',
               {'expert_values_read': False,
                'frozen_before_expert_evaluation': True,
                'rows': ext_meas['rows']})
    ext_exp = external_expert(ext_meas)
    pair = paired_dev()
    # Regression assertions from the independent review, strict fixed tolerances.
    exp_dev = {
        'C0': {'coverage': {'MT': 268, 'PA': 265, 'G0': 22, 'G1': 162},
               'G1_group_equal': 0.5698924731182796, 'G1_ge3': 35,
               'stages': {'apo_failure': 14, 'no_evaluable_fascicle': 3,
                          'no_finite_intersection': 0, 'all_rejected': 103,
                          'output': 162},
               'invalid_counts': {'PA': 28, 'G0': 0, 'G1': 14}},
        'C1': {'coverage': {'MT': 268, 'PA': 267, 'G0': 18, 'G1': 147},
               'G1_group_equal': 0.5161290322580645, 'G1_ge3': 29,
               'stages': {'apo_failure': 14, 'no_evaluable_fascicle': 1,
                          'no_finite_intersection': 0, 'all_rejected': 120,
                          'output': 147},
               'invalid_counts': {'PA': 35, 'G0': 0, 'G1': 15}}}
    for arm, want in exp_dev.items():
        got = dev_meas['summary'][arm]
        for k in ('coverage', 'G1_ge3', 'stages', 'invalid_counts'):
            if got[k] != want[k]:
                raise AssertionError(f'dev {arm} {k}: {got[k]} != {want[k]}')
        if not close(got['G1_group_equal'], want['G1_group_equal'], 1e-10):
            raise AssertionError(f'dev {arm} G1 group equal mismatch')
    if dev_meas['changes']['G1']['new'] != ['image_0039.tif', 'image_0259.tif',
                                            'image_0678.tif', 'image_1650.tif',
                                            'image_1995.tif', 'image_2653.tif']:
        raise AssertionError('dev G1 new identities mismatch')
    if len(dev_meas['changes']['G1']['lost']) != 21:
        raise AssertionError('dev G1 lost count mismatch')
    if pair['only_C1'] != 421 or pair['only_C0'] != 517:
        raise AssertionError('paired new/lost component counts mismatch')
    exp_pair = {
        'angle_diff_deg': {'n_images': 276, 'n_groups': 273,
                           'image_equal': {'C0': 1.3910470279339,
                                           'C1': 1.377938052741152},
                           'group_equal': {'C0': 1.391084619383089,
                                           'C1': 1.3791369418959576},
                           'improved': 156, 'degraded': 120},
        'perp_distance_px': {'n_images': 276, 'n_groups': 273,
                             'image_equal': {'C0': 1.3381758577254204,
                                             'C1': 1.3187585462378888},
                             'group_equal': {'C0': 1.3360563419626745,
                                             'C1': 1.31746619247632},
                             'improved': 152, 'degraded': 124}}
    for field, want in exp_pair.items():
        got = pair['summary'][field]
        for k in ('n_images', 'n_groups', 'improved', 'degraded'):
            if got[k] != want[k]:
                raise AssertionError(f'paired {field} {k}: {got[k]} != {want[k]}')
        for lev in ('image_equal', 'group_equal'):
            for arm in ('C0', 'C1'):
                if not close(got[lev][arm], want[lev][arm], 1e-10):
                    raise AssertionError(f'paired {field} {lev} {arm} mismatch')
    exp_ext = {'MT': {'n_common': 33, 'C0': 2.864044457333383,
                      'C1': 2.864044457333383},
               'PA': {'n_common': 33, 'C0': 1.8443521133496725,
                      'C1': 2.0543072744969537},
               'FL': {'n_common': 17, 'C0': 7.663419370314246,
                      'C1': 7.982210564884925}}
    for p, want in exp_ext.items():
        got = ext_exp['parameters'][p]
        if got['n_common'] != want['n_common']:
            raise AssertionError(f'external {p} common count')
        for arm in ('C0', 'C1'):
            if not close(got[arm]['mae'], want[arm], 1e-9):
                raise AssertionError(f'external {p} {arm} MAE mismatch')
    out = {'version': 'round26_1_a3_legacy_regression_v1',
           'prediction_gate': {k: {'manifest_sha256': v['manifest_sha256'],
                                   'checkpoint_sha256': v['checkpoint_sha256']}
                               for k, v in gate.items()},
           'dev282': {'summary': dev_meas['summary'],
                      'changes': dev_meas['changes']},
           'external34': {'measurement': ext_meas['summary'],
                          'expert_exploratory': ext_exp},
           'corrected_paired': {'summary': pair['summary'],
                                'only_C1': pair['only_C1'],
                                'only_C0': pair['only_C0'],
                                'observation_unit': pair['observation_unit']},
           'all_expected_values_passed': True,
           'tolerances': {'absolute_float': 1e-8, 'ratio_equal': 1e-10},
           'success': True}
    write_json(NEW / 'closeout/legacy_regression.json', out)
    write_json(NEW / 'closeout/legacy_dev_rows.json',
               {'rows': dev_meas['rows']})
    write_json(NEW / 'closeout/legacy_external_rows.json',
               {'rows': ext_meas['rows'], 'expert_exploratory': ext_exp})
    write_json(NEW / 'closeout/corrected_paired_variants.json',
               {'per_variant': pair['per_variant']})
    print(json.dumps({'dev': dev_meas['summary'],
                      'paired': pair['summary'],
                      'external': {p: {'n': ext_exp['parameters'][p]['n_common'],
                                       'C0': ext_exp['parameters'][p]['C0']['mae'],
                                       'C1': ext_exp['parameters'][p]['C1']['mae']}
                                   for p in ('MT', 'PA', 'FL')}},
                     indent=2))


if __name__ == '__main__':
    main()
