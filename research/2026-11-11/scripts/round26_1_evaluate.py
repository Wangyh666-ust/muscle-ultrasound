"""Round26.1 Stage C: frozen C0 vs corrected C1 dev282 and external34."""
import hashlib
import argparse
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
from scripts.evaluate_fasc_structure import PRIMARY, variant_metrics
from scripts.round21_boundary_convention import (
    evaluate_with_fits, mt_value, parse_dev, parse_external,
    read_mask as read_mask_pil, selection_bundle,
)
from scripts.round20_evaluate import (
    build_expert_reference, parse_workbook, summarize_errors,
)
from scripts.round26_1_common import (
    OLD, NEW, CONFIG_OLD, CONFIG_NEW, OLD_C0_CKPT, OLD_C0_SHA,
    OLD_LEGACY_C1_CKPT, OLD_LEGACY_C1_SHA, ISOLATION,
    expected_inputs, j, sha, write_json, validate_prediction_arm,
    atomic_write_success, mark_failure,
)
from scripts.round26_1_legacy_regression import arm_geometry

EXPERT = ROOT / 'external/umud_expert/expert_measurements.json'
WORKBOOK = ROOT / ('external/umud_expert/unpacked/'
                   'benchmark_dataset_architecture_v0.1.0/'
                   'Results_benchmark_architecture_v0.1.0.xlsx')
EVAL = ROOT / 'runs/round26_2_closeout_v1/evaluation'
C1_CKPT = NEW / 'training/C1_corrected/checkpoints/epoch_017.pt'


def pred_map(base):
    m = j(base / 'manifest.json')
    return {(r['dataset'], r['image_id']): base / r['mask_path']
            for r in m['records']}


def mean(v):
    return float(np.mean(v)) if v else None


def paired_image_fields(variants):
    """Common components -> variant mean -> image mean, skipping undefined variants."""
    valid = [v for v in variants if v['common']]
    out = {'paired_common_n': sum(len(v['common']) for v in variants),
           'paired_defined_variants': len(valid)}
    for arm in ('C0', 'C1'):
        for source, target in (('angle_diff_deg', 'angle_mean_deg'),
                               ('perp_distance_px', 'perp_mean_px')):
            out[f'{arm}_{target}'] = mean([
                mean([x[f'{arm}_{source}'] for x in v['common']]) for v in valid])
    for kind, unit in (('angle', 'deg'), ('perp', 'px')):
        out[f'delta_{kind}_{unit}'] = (out[f'C1_{kind}_mean_{unit}'] -
            out[f'C0_{kind}_mean_{unit}']) if valid else None
    return out


def paired_structure(pred0, pred1):
    include = set(j(ISOLATION)['included_filenames'])
    data = UmudTrainData('fasc', 'dev', include_filenames=include)
    variants = []
    per_image = []
    for s in data.samples:
        iid = s['image']['filename']
        masks = {'C0': read_mask(pred0[('dev282', iid)]),
                 'C1': read_mask(pred1[('dev282', iid)])}
        h, w = masks['C0'].shape
        vrs = []
        for vi, mrec in enumerate(s['masks']):
            human = align_mask_to_image(read_mask(ROOT / mrec['path']), (w, h))
            row = {'image_id': iid, 'group_id': s['group_id'],
                   'variant_index': vi, 'mask_path': Path(mrec['path']).as_posix()}
            for arm in ('C0', 'C1'):
                vm = variant_metrics(masks[arm], human)
                inter = float((masks[arm].astype(bool) & human.astype(bool)).sum())
                ps = float(masks[arm].astype(bool).sum())
                hs = float(human.astype(bool).sum())
                row[arm] = {
                    'dice': (2 * inter + 1e-6) / (ps + hs + 1e-6),
                    'iou': inter / max(ps + hs - inter, 1e-6),
                    **vm}
            matches = {}
            predids = {}
            for arm in ('C0', 'C1'):
                pe = extract_segments(masks[arm])
                he = extract_segments(human)
                mm = match_segments(pe['segments'], he['segments'], **PRIMARY)
                matches[arm] = {int(x['human_index']): x for x in mm['matches']}
                predids[arm] = [int(x['label_id']) for x in pe['segments']]
            common = sorted(set(matches['C0']) & set(matches['C1']))
            obs = []
            for k in common:
                a = matches['C0'][k]; b = matches['C1'][k]
                obs.append({
                    'human_component_id': k,
                    'C0_pred_component_id': predids['C0'][a['pred_index']],
                    'C1_pred_component_id': predids['C1'][b['pred_index']],
                    'C0_angle_diff_deg': a['angle_diff_deg'],
                    'C1_angle_diff_deg': b['angle_diff_deg'],
                    'C0_perp_distance_px': a['perp_distance_px'],
                    'C1_perp_distance_px': b['perp_distance_px'],
                    'C0_midpoint_distance_px': a['midpoint_distance_px'],
                    'C1_midpoint_distance_px': b['midpoint_distance_px']})
            row['common'] = obs
            row['only_C0'] = [{
                'human_component_id': k,
                'C0_pred_component_id': predids['C0'][matches['C0'][k]['pred_index']],
                'C0_angle_diff_deg': matches['C0'][k]['angle_diff_deg'],
                'C0_perp_distance_px': matches['C0'][k]['perp_distance_px']}
                for k in sorted(set(matches['C0']) - set(matches['C1']))]
            row['only_C1'] = [{
                'human_component_id': k,
                'C1_pred_component_id': predids['C1'][matches['C1'][k]['pred_index']],
                'C1_angle_diff_deg': matches['C1'][k]['angle_diff_deg'],
                'C1_perp_distance_px': matches['C1'][k]['perp_distance_px']}
                for k in sorted(set(matches['C1']) - set(matches['C0']))]
            for field in ('common', 'only_C0', 'only_C1'):
                for observation in row[field]:
                    observation.update(image_id=iid, group_id=s['group_id'], variant_index=vi)
            variants.append(row); vrs.append(row)
        image_row = {'image_id': iid, 'group_id': s['group_id'],
                     'n_variants': len(vrs)}
        for arm in ('C0', 'C1'):
            for key in ('dice', 'iou', 'human_pixel_coverage_2px',
                        'pred_pixel_precision_2px', 'n_human_segments_evaluable',
                        'n_pred_segments_evaluable', 'n_human_short',
                        'n_human_crossing_or_branched', 'n_pred_short',
                        'n_pred_crossing_or_branched'):
                vals = [v[arm][key] for v in vrs if v[arm].get(key) is not None]
                image_row[f'{arm}_{key}'] = mean(vals)
            for key in ('coverage', 'direction_mean_deg',
                        'position_perp_mean_px', 'extra_candidate_rate'):
                vals = [v[arm]['raw_primary'][key] for v in vrs
                        if v[arm]['raw_primary'][key] is not None]
                image_row[f'{arm}_raw_primary_{key}'] = mean(vals)
        image_row.update(paired_image_fields(vrs))
        per_image.append(image_row)
    # Hierarchy for paired fields: variant mean -> image mean -> group mean.
    paired = {}
    for field in ('angle_diff_deg', 'perp_distance_px'):
        img = {}
        for r in per_image:
            if r.get(f'paired_common_n', 0):
                a = 'angle' if field == 'angle_diff_deg' else 'perp'
                img[r['image_id']] = {
                    'C0': r[f'C0_{a}_mean_{"deg" if a == "angle" else "px"}'],
                    'C1': r[f'C1_{a}_mean_{"deg" if a == "angle" else "px"}']}
        groups = defaultdict(list)
        for iid, v in img.items():
            gid = next(r['group_id'] for r in per_image if r['image_id'] == iid)
            groups[gid].append(v)
        gv = {g: {a: mean([x[a] for x in vs]) for a in ('C0', 'C1')}
              for g, vs in groups.items()}
        paired[field] = {
            'n_images': len(img), 'n_groups': len(gv),
            'image_equal': {a: mean([v[a] for v in img.values()]) for a in ('C0', 'C1')},
            'group_equal': {a: mean([v[a] for v in gv.values()]) for a in ('C0', 'C1')},
            'improved': sum(v['C1'] < v['C0'] for v in img.values()),
            'degraded': sum(v['C1'] > v['C0'] for v in img.values()),
            'improved_ids': [i for i, v in img.items() if v['C1'] < v['C0']],
            'degraded_ids': [i for i, v in img.items() if v['C1'] > v['C0']],
            'image_rows': [{'image_id': i, 'C0': v['C0'], 'C1': v['C1']}
                           for i, v in img.items()],
            'group_rows': [{'group_id': g, **v} for g, v in gv.items()]}
    summary = {}
    for arm in ('C0', 'C1'):
        def agg(key):
            vals = [r[f'{arm}_{key}'] for r in per_image
                    if r.get(f'{arm}_{key}') is not None]
            return {'n': len(vals), 'mean': mean(vals)}
        def group_agg(key):
            gg = defaultdict(list)
            for r in per_image:
                if r.get(f'{arm}_{key}') is not None:
                    gg[r['group_id']].append(r[f'{arm}_{key}'])
            return {'n_groups': len(gg),
                    'mean': mean([mean(v) for v in gg.values()])}
        summary[arm] = {
            'image_equal': {k: agg(k) for k in (
                'dice', 'iou', 'human_pixel_coverage_2px',
                'pred_pixel_precision_2px', 'raw_primary_coverage',
                'raw_primary_direction_mean_deg',
                'raw_primary_position_perp_mean_px',
                'raw_primary_extra_candidate_rate',
                'n_human_segments_evaluable', 'n_pred_segments_evaluable',
                'n_human_short', 'n_human_crossing_or_branched',
                'n_pred_short', 'n_pred_crossing_or_branched')},
            'group_equal': {k: group_agg(k) for k in (
                'dice', 'iou', 'human_pixel_coverage_2px',
                'raw_primary_direction_mean_deg',
                'raw_primary_position_perp_mean_px')}}
    summary['paired_common'] = paired
    only1 = [x for v in variants for x in v['only_C1']]
    only0 = [x for v in variants for x in v['only_C0']]
    summary['coverage_changes'] = {
        'n_observations_only_C1': len(only1),
        'n_observations_only_C0': len(only0),
        'new_angle_mean_deg': mean([x['C1_angle_diff_deg'] for x in only1]),
        'new_perp_mean_px': mean([x['C1_perp_distance_px'] for x in only1]),
        'lost_angle_mean_deg': mean([x['C0_angle_diff_deg'] for x in only0]),
        'lost_perp_mean_px': mean([x['C0_perp_distance_px'] for x in only0]),
        'only_C1': only1, 'only_C0': only0}
    summary['observation_unit'] = ('human component within image-mask variant; '
                                   'not independent anatomy across variants')
    return {'summary': summary, 'per_image': per_image,
            'per_variant': variants}


def measurement(inputs, preds, name):
    rule, rule_hash = load_frozen_rule()
    rows = {'C0': [], 'C1': []}
    for i, s in enumerate(inputs):
        apo = read_mask_pil(s['apo_mask'])
        ap = measure_conservative(apo, rule)
        fits = ap.get('fits') if ap.get('status') == 'candidate' else None
        for arm in ('C0', 'C1'):
            p = preds[arm][(name, s['image_id'])]
            pred = read_mask(p)
            if pred.shape != apo.shape:
                raise ValueError(f'{arm} {s["image_id"]}: mask shape mismatch')
            a = arm_geometry(ap, pred, fits, s['width'], s['height'],
                             s['image_id'])
            if name == 'external34' and s.get('px_per_mm'):
                for k in ('MT', 'G0', 'G1'):
                    if a['values'].get(k) is not None:
                        a['values'][k] = a['values'][k] / float(s['px_per_mm'])
            a.update({'image_id': s['image_id'], 'group_id': s['group_id'],
                      'width': int(s['width']), 'height': int(s['height']),
                      'px_per_mm': s['px_per_mm'],
                      'fasc_mask_sha256': sha(p),
                      'apo_mask_sha256': sha(s['apo_mask'])})
            rows[arm].append(a)
        if (i + 1) % 60 == 0:
            print(name, 'measurement', i + 1, flush=True)
    summary = {}
    for arm in ('C0', 'C1'):
        rs = rows[arm]
        groups = defaultdict(list)
        for r in rs:
            groups[r['group_id']].append(r['values']['G1'] is not None)
        summary[arm] = {
            'n': len(rs),
            'coverage': {p: sum(r['values'][p] is not None for r in rs)
                         for p in ('MT', 'PA', 'G0', 'G1')},
            'G1_group_equal': mean([mean(v) for v in groups.values()]),
            'G1_ge3': sum(len(r['selected_ids']['G1']) >= 3 for r in rs),
            'stages': {k: sum(r['stage'] == k for r in rs) for k in (
                'apo_failure', 'no_evaluable_fascicle',
                'no_finite_intersection', 'all_rejected', 'output')},
            'invalid_counts': {p: sum(len(r['selected_invalid'][p]) for r in rs)
                               for p in ('PA', 'G0', 'G1')}}
    a0 = {r['image_id']: r for r in rows['C0']}
    a1 = {r['image_id']: r for r in rows['C1']}
    changes = {}
    for p in ('MT', 'PA', 'G0', 'G1'):
        old = {i for i, r in a0.items() if r['values'][p] is not None}
        new = {i for i, r in a1.items() if r['values'][p] is not None}
        changes[p] = {'new': sorted(new - old), 'lost': sorted(old - new),
                      'common': len(old & new),
                      'common_mean_increase': sum(
                          a1[i]['values'][p] > a0[i]['values'][p]
                          for i in old & new)}
    return {'summary': summary, 'changes': changes, 'rows': rows,
            'rule_sha256': rule_hash}


def external_expert(meas):
    ids = [r['image_id'] for r in meas['rows']['C0']]
    doc = j(EXPERT)
    if sha(WORKBOOK) != doc['source_sha256']:
        raise ValueError('expert workbook hash mismatch')
    ref = build_expert_reference(parse_workbook(WORKBOOK), set(ids))
    b = {a: {r['image_id']: r for r in meas['rows'][a]} for a in ('C0', 'C1')}
    out = {'formal_errors': {'status': 'blocked',
                             'reason': 'expert units/protocol unconfirmed'},
           'exploratory_assumed_units': {'MT': 'mm', 'PA': 'degree', 'FL': 'mm'},
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
        m0 = summarize_errors([(i, x, r) for i, x, y, r in common])
        m1 = summarize_errors([(i, y, r) for i, x, y, r in common])
        old = {i for i in ids if b['C0'][i]['values'][key] is not None}
        new = {i for i in ids if b['C1'][i]['values'][key] is not None}
        out['parameters'][p] = {
            'n_common': len(common), 'coverage': {a: len(
                [i for i in ids if b[a][i]['values'][key] is not None])
                for a in ('C0', 'C1')},
            'C0': m0, 'C1': m1,
            'improved_ids': [i for i, x, y, r in common
                             if abs(y - r) < abs(x - r) - 1e-12],
            'degraded_ids': [i for i, x, y, r in common
                             if abs(y - r) > abs(x - r) + 1e-12],
            'new_coverage': [{'image_id': i, 'C1_error': abs(b['C1'][i]['values'][key] - ref[i][p]['mean'])}
                             for i in sorted(new - old)],
            'lost_coverage': [{'image_id': i, 'C0_error': abs(b['C0'][i]['values'][key] - ref[i][p]['mean'])}
                              for i in sorted(old - new)]}
        # Median-expert sensitivity, same common identities.
        median = []
        for iid in ids:
            rv = ref[iid][p]['median']
            p0 = b['C0'][iid]['values'][key]
            p1 = b['C1'][iid]['values'][key]
            if rv is not None and p0 is not None and p1 is not None:
                median.append((iid, p0, p1, rv))
        out['parameters'][p]['median_sensitivity'] = {
            'n_common': len(median),
            'C0_mae': summarize_errors([(i, x, r) for i, x, y, r in median])['mae'],
            'C1_mae': summarize_errors([(i, y, r) for i, x, y, r in median])['mae']}
    return out


def main(argv=None):
    from scripts.round26_2_provenance import DEFAULT_GATE, prepare_run_output, verify_closeout_gate
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=EVAL)
    ap.add_argument('--gate', type=Path, default=DEFAULT_GATE)
    args = ap.parse_args(argv)
    ev = prepare_run_output(args.out)
    try:
        gate = verify_closeout_gate(args.gate, labels=False)
        expected = expected_inputs()
        c0 = validate_prediction_arm(
            'C0', OLD / 'predictions/C0', expected, CONFIG_OLD, OLD_C0_CKPT,
            OLD_C0_SHA, ROOT / gate['files']['C0_meta']['path'],
            OLD / 'predictions/C0/prediction_success.json')
        c1 = validate_prediction_arm(
            'C1_corrected', NEW / 'predictions/C1_corrected', expected,
            CONFIG_NEW, C1_CKPT, sha(C1_CKPT),
            ROOT / gate['files']['C1_meta']['path'],
            NEW / 'predictions/C1_corrected/success.json',
            expected_config_sha=sha(CONFIG_NEW))
        # Legacy is verified only as an appendix source, never used as C1.
        legacy = validate_prediction_arm(
            'C1', OLD / 'predictions/C1', expected, CONFIG_OLD,
            OLD_LEGACY_C1_CKPT, OLD_LEGACY_C1_SHA,
            ROOT / gate['files']['legacy_meta']['path'],
            OLD / 'predictions/C1/prediction_success.json', allow_legacy=True)
        freeze = {
            'version': 'round26_1_stage_c_predictions_freeze_v1',
            'C0': {'manifest_sha256': c0['manifest_sha256'],
                   'checkpoint_sha256': OLD_C0_SHA},
            'C1_corrected': {'manifest_sha256': c1['manifest_sha256'],
                             'checkpoint_sha256': sha(C1_CKPT)},
            'legacy_C1_appendix': {'manifest_sha256': legacy['manifest_sha256'],
                                   'checkpoint_sha256': OLD_LEGACY_C1_SHA},
            'reference_values_read': False, 'expert_values_read': False}
        write_json(ev / 'stage_c_predictions_freeze.json', freeze)
        p0 = pred_map(OLD / 'predictions/C0')
        p1 = pred_map(NEW / 'predictions/C1_corrected')
        struct = paired_structure(p0, p1)
        write_json(ev / 'dev282_structure.json', struct)
        devm = measurement(parse_dev(), {'C0': p0, 'C1': p1}, 'dev282')
        write_json(ev / 'dev282_measurement.json', devm)
        # MT must be identical per image because apo is fixed.
        for r0, r1 in zip(devm['rows']['C0'], devm['rows']['C1']):
            if r0['values']['MT'] != r1['values']['MT']:
                raise ValueError(f'dev MT changed: {r0["image_id"]}')
        extm = measurement(parse_external(), {'C0': p0, 'C1': p1}, 'external34')
        write_json(ev / 'external34_measurement_predictions.json',
                   {'expert_values_read': False,
                    'frozen_before_expert_evaluation': True,
                    'rows': extm['rows']})
        write_json(ev / 'stage_c_external_measurement_manifest.json',
                   {'measurement_sha256': sha(ev / 'external34_measurement_predictions.json'),
                    'expert_values_read': False})
        ext = external_expert(extm)
        write_json(ev / 'external34_expert.json', ext)
        # Historical appendix: legacy C1 only, clearly labelled.
        legacy_reg = j(NEW / 'closeout/legacy_regression.json')
        summary = {
            'version': 'round26_2_stage_c_evaluation_v1',
            'predictions_freeze': freeze,
            'main_comparison': {'C0': 'C0_original',
                                'C1': 'C1_corrected_mean_lambda_0p1'},
            'dev282_structure': struct['summary'],
            'dev282_measurement': devm,
            'external34_measurement': extm,
            'external34_expert': ext,
            'legacy_C1_appendix_only': {
                'label': 'legacy_C1_sum_lambda_0p1',
                'checkpoint_sha256': OLD_LEGACY_C1_SHA,
                'legacy_regression': legacy_reg},
            'expert_values_used_for_training_or_selection': False,
            'formal_external_errors_blocked': True,
            'exploratory_units_assumption': {'MT': 'mm', 'PA': 'degree', 'FL': 'mm'}}
        write_json(ev / 'summary.json', summary)
        atomic_write_success(ev, {
            'version': 'round26_2_stage_c_success_v1',
            'gate_sha256': sha(args.gate),
            'C0_manifest_sha256': c0['manifest_sha256'],
            'C1_manifest_sha256': c1['manifest_sha256'],
            'n_dev282': 282, 'n_external34': 34})
        printed_paired = {k: {f: v for f, v in stats.items()
                             if f not in ('image_rows', 'group_rows', 'improved_ids', 'degraded_ids')}
                          for k, stats in struct['summary']['paired_common'].items()}
        print(json.dumps({
            'dev_dice': {a: struct['summary'][a]['image_equal']['dice']['mean']
                         for a in ('C0', 'C1')},
            'dev_coverage': {a: devm['summary'][a]['coverage'] for a in ('C0', 'C1')},
            'dev_stages': {a: devm['summary'][a]['stages'] for a in ('C0', 'C1')},
            'paired': printed_paired,
            'external': {p: {'n': ext['parameters'][p]['n_common'],
                             'C0': ext['parameters'][p]['C0']['mae'],
                             'C1': ext['parameters'][p]['C1']['mae']}
                         for p in ('MT', 'PA', 'FL')}}, indent=2))
    except Exception as e:
        mark_failure(ev, e)
        raise


if __name__ == '__main__':
    main()
