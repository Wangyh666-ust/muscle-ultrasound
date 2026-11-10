"""Round20 measurement baseline: formal coverage and exploratory direct errors.

Predictions must already be frozen. Expert values are read only here, after the
prediction hash is verified. Formal fields remain null unless explicit unit
evidence exists; exploratory fields use only the stated mm/degree working
assumption and are never called competition accuracy.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/geometry_eval/round20_measurement_baseline_v1'
PRED = BASE / 'predictions/parameters_frozen.json'
PRED_SHA = BASE / 'predictions/parameters_frozen.sha256'
EXPERT = ROOT / 'external/umud_expert/expert_measurements.json'
WORKBOOK = ROOT / ('external/umud_expert/unpacked/'
                   'benchmark_dataset_architecture_v0.1.0/'
                   'Results_benchmark_architecture_v0.1.0.xlsx')
PROTOCOL = ROOT / 'external/umud_expert/protocol_evidence/protocol_status.json'
ROUND19_EVAL = ROOT / ('runs/geometry_eval/round19_external_fl_reference_v1/'
                       'evaluation/evaluation.json')
ROUND19_PRED = ROOT / ('runs/geometry_eval/round19_external_fl_reference_v1/'
                       'predictions/predictions_frozen.json')
ROUND14 = ROOT / ('runs/geometry_eval/round14_extrapolation_policy_v1/'
                  'round13_b_unet34/summary.json')
V3_COVERAGE = ROOT / ('runs/round16_scale_and_test_coverage_v3/'
                      'test_coverage/coverage.json')
OVERLAYS = ROOT / 'runs/round15_measurement_prototype_v2/measurement/overlays'
ROUND19_REVIEW = ROOT / 'review/ROUND19_INDEPENDENT_REVIEW_20261019.md'
SYNTH = BASE / 'synthetic_closure/synthetic_closure.json'
EVIDENCE = ROOT / 'external/umud_expert/protocol_evidence/EVIDENCE.md'
# Existing Round19 diagnostic evidence, copied only for case context. It is not
# a new inference and does not change any frozen selection or threshold.
G1_REJECTION_EVIDENCE = {
    'im_03_arch': {'n_evaluable': 6, 'remaining_conditions': ['excessive_extrapolation'], 'min_extrapolation_ratio': 0.8729},
    'im_06_arch': {'n_evaluable': 18, 'remaining_conditions': ['excessive_extrapolation'], 'min_extrapolation_ratio': 0.8901},
    'im_09_arch': {'n_evaluable': 14, 'remaining_conditions': ['excessive_extrapolation'], 'min_extrapolation_ratio': 0.8428},
    'im_12_arch': {'n_evaluable': 17, 'remaining_conditions': ['apo_status_rejected', 'no_apo_boundaries'], 'min_extrapolation_ratio': None},
    'im_17_arch': {'n_evaluable': 8, 'remaining_conditions': ['excessive_extrapolation'], 'min_extrapolation_ratio': 0.8842},
    'im_18_arch': {'n_evaluable': 14, 'remaining_conditions': ['excessive_extrapolation'], 'min_extrapolation_ratio': 0.8831},
    'im_20_arch': {'n_evaluable': 21, 'remaining_conditions': ['excessive_extrapolation'], 'min_extrapolation_ratio': 1.0000},
    'im_23_arch': {'n_evaluable': 7, 'remaining_conditions': ['excessive_extrapolation'], 'min_extrapolation_ratio': 0.8591},
    'im_31_arch': {'n_evaluable': 13, 'remaining_conditions': ['excessive_extrapolation'], 'min_extrapolation_ratio': 0.8029},
}
OUT = BASE / 'evaluation'
RATERS = ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7')
PARAMS = ('MT', 'PA', 'FL')
METHODS = ('MT', 'PA', 'FL_G0', 'FL_G1', 'B')
ASSUMED_UNITS = {'MT': 'mm', 'PA': 'degree', 'FL': 'mm'}
COMPETITION_TOL = {'MT': 3.0, 'PA': 6.0, 'FL': 12.0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def j(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite_positive(v):
    """Deprecated compatibility helper; do not use for expert validity."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def finite(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def expert_value(raw):
    """Classify an expert cell without silently deleting finite values.

    Returns (value, issue). Missing/non-numeric/non-finite are missing;
    finite zero/negative values are retained and marked suspicious.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, 'missing'
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, 'not_numeric'
    if not math.isfinite(value):
        return None, 'non_finite'
    if value <= 0:
        return value, 'finite_nonpositive_suspicious'
    return value, ''


def safe(v):
    if isinstance(v, dict):
        return {k: safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [safe(x) for x in v]
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        v = int(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def parse_workbook(path):
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with zipfile.ZipFile(path) as z:
        shared = ET.fromstring(z.read('xl/sharedStrings.xml'))
        strings = [''.join(t.text or '' for t in si.iter(ns + 't'))
                   for si in shared]
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = sheet.findall('.//' + ns + 'sheetData/' + ns + 'row')
        parsed = []
        headers = None
        for row in rows:
            cells = {}
            for cell in row.findall(ns + 'c'):
                ref = cell.get('r')
                col = ''.join(c for c in ref if c.isalpha())
                t = cell.get('t')
                v = cell.find(ns + 'v')
                val = v.text if v is not None else None
                if t == 's' and val is not None:
                    val = strings[int(val)]
                cells[col] = val
            if headers is None:
                headers = cells
            else:
                parsed.append(cells)
    header_by_col = {col: name for col, name in headers.items()}
    out = {}
    for cells in parsed:
        row = {header_by_col[col]: value
               for col, value in cells.items() if col in header_by_col}
        iid = row.get('ImageID')
        if iid is None:
            continue
        out[iid] = row
    return out


def build_expert_reference(workbook_rows, allowed_ids, validate_distribution=True):
    ref = {}
    for iid in allowed_ids:
        row = workbook_rows.get(iid)
        if row is None:
            raise ValueError(f'workbook missing allowed image {iid}')
        params = {}
        for param in PARAMS:
            values = {}
            invalid = []
            suspicious = []
            for rater in RATERS:
                raw = row.get(f'{rater}_{param}')
                value, issue = expert_value(raw)
                if value is None:
                    invalid.append({'rater': rater, 'raw': raw,
                                    'reason': issue})
                else:
                    values[rater] = value
                    if issue:
                        suspicious.append({'rater': rater, 'raw': raw,
                                           'value': value, 'reason': issue})
            nums = list(values.values())
            params[param] = {
                'n_valid': len(nums),
                'n_positive': sum(v > 0 for v in nums),
                'n_suspicious': len(suspicious),
                'valid_values': values,
                'invalid_entries': invalid,
                'suspicious_entries': suspicious,
                'mean': float(np.mean(nums)) if nums else None,
                'median': float(np.median(nums)) if nums else None,
                'min': min(nums) if nums else None,
                'max': max(nums) if nums else None,
                'q1': float(np.percentile(nums, 25)) if nums else None,
                'q3': float(np.percentile(nums, 75)) if nums else None,
                'iqr': (float(np.percentile(nums, 75) - np.percentile(nums, 25))
                        if nums else None),
            }
        ref[iid] = params
    if validate_distribution:
        for param in PARAMS:
            dist = Counter(ref[i][param]['n_valid'] for i in allowed_ids)
            if dist != {7: 16, 6: 18}:
                raise ValueError(
                    f'{param}: unexpected expert n_valid distribution {dist}')
    return ref


def compare_expert_json(ref, expert_doc, allowed_ids):
    by_id = {r['image_id']: r for r in expert_doc['records']}
    mismatches = []
    for iid in allowed_ids:
        for param in PARAMS:
            raw = (by_id[iid].get('expert_values_raw') or {})
            for rater in RATERS:
                a = ref[iid][param]['valid_values'].get(rater)
                b, _ = expert_value((raw.get(rater) or {}).get(param))
                if a is None or b is None:
                    if a != b:
                        mismatches.append((iid, param, rater, a, b))
                elif abs(a - b) > 1e-9:
                    mismatches.append((iid, param, rater, a, b))
    return mismatches


def summarize_errors(pairs):
    """pairs: list (image_id, prediction, reference); prediction-reference."""
    if not pairs:
        return {
            'n': 0, 'mae': None, 'bias': None, 'medae': None,
            'rmse': None, 'abs_p90': None, 'abs_max': None,
            'abs_max_id': None, 'signed_min': None, 'signed_max': None,
        }
    diffs = [p - r for _, p, r in pairs]
    absd = [abs(d) for d in diffs]
    idx = int(np.argmax(absd))
    return {
        'n': len(pairs),
        'mae': float(np.mean(absd)),
        'bias': float(np.mean(diffs)),
        'medae': float(np.median(absd)),
        'rmse': float(math.sqrt(np.mean([d * d for d in diffs]))),
        'abs_p90': float(np.percentile(absd, 90)),
        'abs_max': float(max(absd)),
        'abs_max_id': pairs[idx][0],
        'signed_min': float(min(diffs)),
        'signed_max': float(max(diffs)),
    }


def pair_rows(rows, method, reference):
    out = []
    for r in rows:
        pred = finite(r['pred'].get(method))
        ref = finite(r['ref'].get(reference))
        if pred is not None and ref is not None:
            out.append((r['image_id'], pred, ref))
    return out


def image_equal_mean(records, key):
    """Mean within image, then equal mean across images; missing excluded."""
    by_image = defaultdict(list)
    for rec in records:
        value = rec.get(key)
        if value is not None and math.isfinite(float(value)):
            by_image[rec['image_id']].append(abs(float(value)))
    per_image = [float(np.mean(v)) for v in by_image.values() if v]
    return (float(np.mean(per_image)) if per_image else None,
            len(per_image))


def loo_image_summary(values):
    errors = []
    for rater, value in values.items():
        others = [v for k, v in values.items() if k != rater]
        if others:
            errors.append(abs(value - float(np.mean(others))))
    return {
        'n_valid': len(values),
        'image_mean_abs_expert_loo_error': (
            float(np.mean(errors)) if errors else None),
    }


def coverage(rows, method, reference):
    pred_ok = sum(finite(r['pred'].get(method)) is not None for r in rows)
    ref_ok = sum(finite(r['ref'].get(reference)) is not None for r in rows)
    common = sum(finite(r['pred'].get(method)) is not None
                 and finite(r['ref'].get(reference)) is not None
                 for r in rows)
    pred_missing = [r['image_id'] for r in rows
                    if finite(r['pred'].get(method)) is None]
    ref_missing = [r['image_id'] for r in rows
                   if finite(r['ref'].get(reference)) is None]
    reasons = Counter()
    for r in rows:
        if finite(r['pred'].get(method)) is None:
            reasons[r['pred_reason'].get(method, 'missing_prediction')] += 1
        if finite(r['ref'].get(reference)) is None:
            reasons['reference_missing'] += 1
    pred_success_ids = [r['image_id'] for r in rows
                        if finite(r['pred'].get(method)) is not None]
    ref_valid_ids = [r['image_id'] for r in rows
                     if finite(r['ref'].get(reference)) is not None]
    return {
        'n_images': len(rows),
        'prediction_ok': pred_ok,
        'reference_ok': ref_ok,
        'common': common,
        'prediction_success_ids': pred_success_ids,
        'reference_valid_ids': ref_valid_ids,
        'prediction_missing_ids': pred_missing,
        'reference_missing_ids': ref_missing,
        'failure_reasons': dict(sorted(reasons.items())),
        'failures_not_in_numerator_and_not_zero': True,
    }


def metrics_table(rows, method, reference):
    pairs = pair_rows(rows, method, reference)
    return {
        'coverage': coverage(rows, method, reference),
        'metrics': summarize_errors(pairs),
        'ids': [iid for iid, _, _ in pairs],
    }


def average_ranks(values):
    vals = [float(v) for v in values]
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman(x, y):
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    if len(xs) < 3:
        return {'rho': None, 'n': len(xs), 'reason': 'n_lt_3'}
    if not all(math.isfinite(v) for v in xs + ys):
        return {'rho': None, 'n': len(xs), 'reason': 'non_finite'}
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return {'rho': None, 'n': len(xs), 'reason': 'constant_vector'}
    rx = np.asarray(average_ranks(xs)); ry = np.asarray(average_ranks(ys))
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if den == 0:
        return {'rho': None, 'n': len(xs), 'reason': 'constant_rank_vector'}
    return {'rho': float(max(-1.0, min(1.0, float((rx * ry).sum() / den)))),
            'n': len(xs), 'reason': ''}


def loo_disagreement(expert_ref, rows, method, param):
    image_rows = []
    per_expert = defaultdict(list)
    for r in rows:
        iid = r['image_id']
        vals = expert_ref[iid][param]['valid_values']
        model = finite(r['pred'].get(method))
        if len(vals) < 2:
            continue
        exp_errors = []
        mod_errors = []
        for rater, value in vals.items():
            others = [v for k, v in vals.items() if k != rater]
            if not others:
                continue
            loo = float(np.mean(others))
            e = value - loo
            exp_errors.append(abs(e))
            rec = {
                'image_id': iid, 'param': param, 'rater': rater,
                'expert_value': value, 'reference_without_r': loo,
                'expert_error': e, 'expert_abs_error': abs(e),
                'model': model,
            }
            if model is not None:
                me = model - loo
                mod_errors.append(abs(me))
                rec['model_error'] = me
                rec['model_abs_error'] = abs(me)
            else:
                rec['model_error'] = None
                rec['model_abs_error'] = None
            per_expert[rater].append(rec)
        image_rows.append({
            'image_id': iid,
            'n_experts': len(vals),
            'expert_abs_error_mean': float(np.mean(exp_errors)),
            'model_abs_error_mean': (float(np.mean(mod_errors))
                                     if mod_errors else None),
            'model_available': model is not None,
        })
    model_images = [r for r in image_rows if r['model_available']]
    expert_images = image_rows
    def summ(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return float(np.mean(v)) if v else None
    return {
        'param': param, 'method': method,
        'all_expert_images': {
            'n_images': len(expert_images),
            'image_equal_mean_expert_abs_error': summ(
                expert_images, 'expert_abs_error_mean'),
        },
        'model_common_images': {
            'n_images': len(model_images),
            'image_equal_mean_expert_abs_error': summ(
                model_images, 'expert_abs_error_mean'),
            'image_equal_mean_model_abs_error': summ(
                model_images, 'model_abs_error_mean'),
            'same_images_and_loo_reference': True,
        },
        'image_rows': image_rows,
        'per_expert': {
            rater: {
                'n_image_values': len(recs),
                'image_equal_mean_expert_abs_error': image_equal_mean(
                    recs, 'expert_error')[0],
                'image_equal_mean_model_abs_error': image_equal_mean(
                    recs, 'model_error')[0],
                'n_images_expert': image_equal_mean(
                    recs, 'expert_error')[1],
                'n_images_model': image_equal_mean(
                    recs, 'model_error')[1],
                'rows': recs,
            } for rater, recs in sorted(per_expert.items())
        },
    }


def merge_records(records, expert):
    """Combine frozen predictions with expert references.

    The prediction side is copied verbatim; only the `ref`/`expert` fields
    depend on expert input. This is the isolation point used by the
    expert-copy regression test.
    """
    merged = []
    for r in records:
        iid = r['image_id']
        preds = {
            'MT': r['MT'].get('value_mm'),
            'PA': r['PA'].get('value_deg'),
            'FL_G0': r['FL_G0'].get('selected_mean_mm'),
            'FL_G1': r['FL_G1'].get('selected_mean_mm'),
            'B': r['B'].get('value_mm') if 'B' in r else None,
        }
        reasons = {
            'MT': r['MT'].get('reason') or r['MT'].get('status'),
            'PA': r['PA'].get('reason') or r['PA'].get('status'),
            'FL_G0': r['FL_G0'].get('reason') or r['FL_G0'].get('status'),
            'FL_G1': r['FL_G1'].get('reason') or r['FL_G1'].get('status'),
            'B': (r.get('B') or {}).get('reason')
                 or (r.get('B') or {}).get('prediction_status'),
        }
        ref = {
            'MT': expert[iid]['MT']['mean'],
            'PA': expert[iid]['PA']['mean'],
            'FL': expert[iid]['FL']['mean'],
            'MT_median': expert[iid]['MT']['median'],
            'PA_median': expert[iid]['PA']['median'],
            'FL_median': expert[iid]['FL']['median'],
        }
        merged.append({
            'image_id': iid, 'path': r['path'],
            'width': r['width'], 'height': r['height'],
            'pixel_sha256': r['pixel_sha256'], 'px_per_mm': r['px_per_mm'],
            'pred': preds, 'pred_reason': reasons, 'ref': ref,
            'apo': r['apo'], 'MT': r['MT'], 'PA': r['PA'],
            'FL_G0': r['FL_G0'], 'FL_G1': r['FL_G1'],
            'B': r.get('B'),
            'expert': expert[iid],
        })
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', default=str(PRED))
    ap.add_argument('--predictions-sha', default=str(PRED_SHA))
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    def resolve(v):
        p = Path(v)
        return p if p.is_absolute() else ROOT / p
    pred_path = resolve(args.predictions)
    pred_sha_path = resolve(args.predictions_sha)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_sha = sha(pred_path)
    if pred_sha != pred_sha_path.read_text(encoding='utf-8').strip():
        raise ValueError('parameter prediction hash mismatch; refusing expert read')
    pred = j(pred_path)
    rows = pred['records']
    ids = [r['image_id'] for r in rows]
    if len(rows) != 34 or len(set(ids)) != 34 or 'im_07_arch' in ids:
        raise ValueError('parameter table is not the fixed 34-image set')
    if pred.get('coverage') != {'MT': 33, 'PA': 33, 'FL_G0': 4,
                                'FL_G1': 25, 'B': 33}:
        raise ValueError(f'unexpected frozen coverage: {pred.get("coverage")}')
    expert_doc = j(EXPERT)
    if sha(WORKBOOK) != expert_doc['source_sha256']:
        raise ValueError('expert workbook hash mismatch')
    workbook_rows = parse_workbook(WORKBOOK)
    expert = build_expert_reference(workbook_rows, set(ids))
    mismatches = compare_expert_json(expert, expert_doc, ids)
    if mismatches:
        raise ValueError(f'expert workbook/JSON mismatches: {mismatches[:5]}')
    protocol = j(PROTOCOL)
    round19_eval = j(ROUND19_EVAL)
    merged = merge_records(rows, expert)
    # Main availability table, one parameter/reference at a time.
    table1 = {}
    for method, refkey in (('MT', 'MT'), ('PA', 'PA'),
                           ('FL_G0', 'FL'), ('FL_G1', 'FL'),
                           ('B', 'FL')):
        table1[method] = {
            'mean_reference': metrics_table(merged, method, refkey),
            'median_reference': metrics_table(merged, method,
                                              refkey + '_median'),
        }
    common_ids = [r['image_id'] for r in merged
                  if finite(r['pred']['MT']) is not None
                  and finite(r['pred']['PA']) is not None
                  and finite(r['pred']['FL_G1']) is not None
                  and finite(r['ref']['MT']) is not None
                  and finite(r['ref']['PA']) is not None
                  and finite(r['ref']['FL']) is not None]
    common25 = [r for r in merged if r['image_id'] in set(common_ids)]
    if len(common_ids) != 25:
        raise ValueError(f'unexpected MT+PA+FL_G1 common set: {len(common_ids)}')
    table2 = {
        'ids': common_ids,
        'n': len(common_ids),
        'MT': metrics_table(common25, 'MT', 'MT'),
        'PA': metrics_table(common25, 'PA', 'PA'),
        'FL_G1': metrics_table(common25, 'FL_G1', 'FL'),
        'MT_median_reference': metrics_table(common25, 'MT', 'MT_median'),
        'PA_median_reference': metrics_table(common25, 'PA', 'PA_median'),
        'FL_G1_median_reference': metrics_table(common25, 'FL_G1', 'FL_median'),
    }
    common_metrics = {k: table2[k]['metrics'] for k in ('MT', 'PA', 'FL_G1')}
    table2['normalized_components_from_this_common_set'] = {
        'assumed_units': True,
        'label': '按比赛容差归一化的探索性分量 / exploratory components normalized by competition tolerances; not a Kaggle score',
        'MAE_MT_div_3': (
            common_metrics['MT']['mae'] / 3.0
            if common_metrics['MT']['mae'] is not None else None),
        'MAE_PA_div_6': (
            common_metrics['PA']['mae'] / 6.0
            if common_metrics['PA']['mae'] is not None else None),
        'MAE_FL_div_12': (
            common_metrics['FL_G1']['mae'] / 12.0
            if common_metrics['FL_G1']['mae'] is not None else None),
        'note': ('computed on the same common 25 IDs; external protocol differs '
                 'from competition; no ranking, weights, or Kaggle score'),
    }
    # A_G1 vs B on the same 25 IDs.
    ab_ids = [r['image_id'] for r in merged
              if finite(r['pred']['FL_G1']) is not None
              and finite(r['pred']['B']) is not None
              and finite(r['ref']['FL']) is not None]
    if len(ab_ids) != 25:
        raise ValueError(f'unexpected A_G1/B common set: {len(ab_ids)}')
    ab_rows = [r for r in merged if r['image_id'] in set(ab_ids)]
    a_vals = [r['pred']['FL_G1'] for r in ab_rows]
    b_vals = [r['pred']['B'] for r in ab_rows]
    ref_mean = [r['ref']['FL'] for r in ab_rows]
    ref_med = [r['ref']['FL_median'] for r in ab_rows]
    table3 = {
        'ids': ab_ids,
        'n': len(ab_ids),
        'A_G1_mean_reference': metrics_table(ab_rows, 'FL_G1', 'FL'),
        'B_mean_reference': metrics_table(ab_rows, 'B', 'FL'),
        'A_G1_median_reference': metrics_table(ab_rows, 'FL_G1', 'FL_median'),
        'B_median_reference': metrics_table(ab_rows, 'B', 'FL_median'),
        'spearman_expert_mean': {
            'A_G1': spearman(a_vals, ref_mean),
            'B': spearman(b_vals, ref_mean),
        },
        'spearman_expert_median': {
            'A_G1': spearman(a_vals, ref_med),
            'B': spearman(b_vals, ref_med),
        },
    }
    g0b_ids = [r['image_id'] for r in merged
               if finite(r['pred']['FL_G0']) is not None
               and finite(r['pred']['B']) is not None
               and finite(r['ref']['FL']) is not None]
    g0b_rows = [r for r in merged if r['image_id'] in set(g0b_ids)]
    table3['A_G0_B_appendix'] = {
        'ids': g0b_ids,
        'n': len(g0b_ids),
        'A_G0': metrics_table(g0b_rows, 'FL_G0', 'FL'),
        'B': metrics_table(g0b_rows, 'B', 'FL'),
    }
    b_ids = [r['image_id'] for r in merged
             if finite(r['pred']['B']) is not None
             and finite(r['ref']['FL']) is not None]
    b_rows = [r for r in merged if r['image_id'] in set(b_ids)]
    table3['B_only_appendix'] = {
        'ids': b_ids, 'n': len(b_ids),
        'B_mean_reference': metrics_table(b_rows, 'B', 'FL'),
        'note': 'different denominator; not directly compared with A_G1',
    }
    # Missing lists for every parameter.
    table4 = {}
    for method, refkey in (('MT', 'MT'), ('PA', 'PA'),
                           ('FL_G0', 'FL'), ('FL_G1', 'FL'), ('B', 'FL')):
        missing = [{
            'image_id': r['image_id'],
            'prediction': r['pred'].get(method),
            'prediction_reason': r['pred_reason'].get(method),
            'reference_valid': finite(r['ref'].get(refkey)) is not None,
            'expert_n': r['expert'][refkey]['n_valid'],
            'g1_failure_evidence': (
                G1_REJECTION_EVIDENCE.get(r['image_id'])
                if method == 'FL_G1' else None),
        } for r in merged
            if finite(r['pred'].get(method)) is None
            and finite(r['ref'].get(refkey)) is not None]
        table4[method] = {'n_missing_with_reference': len(missing),
                          'rows': missing}
    # Expert table and disagreement.
    expert_rows = []
    for r in merged:
        for param in PARAMS:
            e = r['expert'][param]
            expert_rows.append({
                'image_id': r['image_id'], 'param': param,
                'n_valid': e['n_valid'], 'mean': e['mean'],
                'median': e['median'], 'min': e['min'], 'max': e['max'],
                'q1': e['q1'], 'q3': e['q3'], 'iqr': e['iqr'],
                'valid_values': e['valid_values'],
                'invalid_entries': e['invalid_entries'],
            })
    disagreement = {
        'definition': ('leave-one-rater-out mean; expert_error_r = expert_r '
                       '- mean(other experts); model_error_r = model - same '
                       'loo reference; image-equal aggregation'),
        'MT': loo_disagreement(expert, merged, 'MT', 'MT'),
        'PA': loo_disagreement(expert, merged, 'PA', 'PA'),
        'FL_G1': loo_disagreement(expert, merged, 'FL_G1', 'FL'),
        'FL_G0_appendix': loo_disagreement(expert, merged, 'FL_G0', 'FL'),
        'B_appendix': loo_disagreement(expert, merged, 'B', 'FL'),
    }
    # im_19 R7 MT outlier diagnostics: retained, not removed.
    im19 = next((r for r in merged if r['image_id'] == 'im_19_arch'), None)
    im19_diag = None
    if im19:
        vals = im19['expert']['MT']['valid_values']
        without = {k: v for k, v in vals.items() if k != 'R7'}
        im19_diag = {
            'image_id': 'im_19_arch',
            'R7_MT': vals.get('R7'),
            'valid_values': vals,
            'mean_all': float(np.mean(list(vals.values()))),
            'median_all': float(np.median(list(vals.values()))),
            'mean_without_R7_diagnostic': (
                float(np.mean(list(without.values()))) if without else None),
            'MT_metrics_with_im19_main': summarize_errors(
                pair_rows(merged, 'MT', 'MT')),
            'MT_median_metrics_with_im19_main': summarize_errors(
                pair_rows(merged, 'MT', 'MT_median')),
            'MT_metrics_without_im19_diagnostic_only': summarize_errors(
                pair_rows([r for r in merged if r['image_id'] != 'im_19_arch'],
                          'MT', 'MT')),
            'MT_median_metrics_without_im19_diagnostic_only': summarize_errors(
                pair_rows([r for r in merged if r['image_id'] != 'im_19_arch'],
                          'MT', 'MT_median')),
            'note': ('R7 is retained in every main summary; the without-im19 '
                     'row is a sensitivity display only and is not a replacement summary'),
        }
    # Case selection, fixed rule.
    top = {}
    for method, refkey in (('MT', 'MT'), ('PA', 'PA'), ('FL_G1', 'FL')):
        pairs = []
        for r in merged:
            p = finite(r['pred'].get(method))
            q = finite(r['ref'].get(refkey))
            if p is not None and q is not None:
                pairs.append((abs(p - q), r['image_id']))
        top[method] = [iid for _, iid in sorted(
            pairs, key=lambda z: (-z[0], z[1]))[:3]]
    g1_missing = [r['image_id'] for r in merged
                  if finite(r['pred']['FL_G1']) is None
                  and finite(r['ref']['FL']) is not None]
    pa_invalid = [r['image_id'] for r in merged
                  if r['PA'].get('selected_invalid_count')]
    case_ids = sorted(set(top['MT'] + top['PA'] + top['FL_G1']
                          + g1_missing + pa_invalid + ['im_06_arch']))
    cases = []
    for iid in case_ids:
        r = next(x for x in merged if x['image_id'] == iid)
        tags = []
        if iid in top['MT'] or iid == 'im_06_arch':
            tags.append('boundary_selection_or_thickness')
        if iid in top['PA'] or iid in pa_invalid:
            tags.append('angle_or_PA_selection')
        if iid in top['FL_G1'] or iid in g1_missing:
            tags.append('FL_visible_support_or_extrapolation')
        if ((r['expert']['MT']['iqr'] or 0) > 2.0
                or (r['expert']['PA']['iqr'] or 0) > 4.0
                or (r['expert']['FL']['iqr'] or 0) > 8.0):
            tags.append('expert_or_protocol_disagreement')
        if not tags:
            tags.append('cannot_distinguish')
        tags = sorted(set(tags))
        if len(tags) > 1:
            tags.append('multiple_hypotheses_allowed')
        overlay = OVERLAYS / f'{iid}.png'
        selected_by = {
            'top3_MT': iid in top['MT'],
            'top3_PA': iid in top['PA'],
            'top3_FL_G1': iid in top['FL_G1'],
            'G1_no_output': iid in g1_missing,
            'PA_selected_invalid': iid in pa_invalid,
            'fixed_im_06': iid == 'im_06_arch',
        }
        selection_reasons = [k for k, v in selected_by.items() if v]
        cases.append({
            'image_id': iid,
            'selected_by': selected_by,
            'selection_reasons': selection_reasons,
            'hypothesis_tags': tags,
            'provisional_hypotheses': tags,
            'hypothesis_status': 'unverified; multiple hypotheses may apply',
            'overlay': str(overlay.relative_to(ROOT)) if overlay.exists() else None,
            'overlay_asset': (f'assets/{iid}{overlay.suffix}'
                              if overlay.exists() else None),
            'expert': {p: r['expert'][p] for p in PARAMS},
            'model': {
                'MT': r['MT'], 'PA': r['PA'],
                'FL_G0': r['FL_G0'], 'FL_G1': r['FL_G1'], 'B': r['B']},
            'error_vs_expert_mean': {
                'MT': (r['pred']['MT'] - r['ref']['MT']
                       if finite(r['pred']['MT']) is not None
                       and finite(r['ref']['MT']) is not None else None),
                'PA': (r['pred']['PA'] - r['ref']['PA']
                       if finite(r['pred']['PA']) is not None
                       and finite(r['ref']['PA']) is not None else None),
                'FL_G1': (r['pred']['FL_G1'] - r['ref']['FL']
                          if finite(r['pred']['FL_G1']) is not None
                          and finite(r['ref']['FL']) is not None else None),
            },
            'expert_loo': {p: loo_image_summary(
                r['expert'][p]['valid_values']) for p in PARAMS},
            'g1_failure_evidence': G1_REJECTION_EVIDENCE.get(iid),
            'apo': r['apo'],
        })
    # Formal and exploratory outputs.
    formal = {
        'version': 'round20_formal_measurement_status_v1',
        'assumed_units': False,
        'unit_status': protocol['unit_status'],
        'comparison_label': protocol['comparison_label'],
        'parameters': {
            'MT': {'coverage': table1['MT']['mean_reference']['coverage'],
                   'metrics': {'mean_reference': None,
                               'median_reference': None},
                   'blocked_reason': 'no explicit MT unit evidence; scale reconstruction is not confirmation'},
            'PA': {'coverage': table1['PA']['mean_reference']['coverage'],
                   'metrics': {'mean_reference': None,
                               'median_reference': None},
                   'blocked_reason': 'no explicit PA unit evidence and measurement protocol partially confirmed'},
            'FL_G0': {'coverage': table1['FL_G0']['mean_reference']['coverage'],
                      'metrics': {'mean_reference': None,
                                  'median_reference': None},
                      'blocked_reason': 'no explicit FL unit evidence'},
            'FL_G1': {'coverage': table1['FL_G1']['mean_reference']['coverage'],
                      'metrics': {'mean_reference': None,
                                  'median_reference': None},
                      'blocked_reason': 'no explicit FL unit evidence; G1 experimental'},
            'B': {'coverage': table1['B']['mean_reference']['coverage'],
                  'metrics': {'mean_reference': None,
                              'median_reference': None},
                  'blocked_reason': 'shadow formula only; no unit-confirmed formal metric'},
        },
        'formal_competition_accuracy': False,
        'usable_for_model_selection_tuning_or_calibration': False,
        'sources': {
            'predictions': str(pred_path.relative_to(ROOT)),
            'predictions_sha256': pred_sha,
            'expert_values': str(EXPERT.relative_to(ROOT)),
            'expert_json_sha256': sha(EXPERT),
            'expert_workbook': str(WORKBOOK.relative_to(ROOT)),
            'expert_workbook_sha256': sha(WORKBOOK),
            'provided_scale_column': 'Scale_pixel_per_cm/10',
        },
        'metric_normalization': 'none; formal absolute error fields are null while units are unconfirmed',
        'algorithm_columns_used': False,
    }
    exploratory = {
        'version': 'round20_exploratory_assumed_units_v1',
        'assumed_units': True,
        'unit_status': 'unconfirmed',
        'protocol_status': protocol['protocol_status'],
        'unit_assumption': {'MT': 'mm', 'FL': 'mm', 'PA': 'degree'},
        'comparison_label': 'external_expert_benchmark_consistency_protocol_not_fully_verified',
        'not_competition_accuracy': True,
        'usable_for_model_selection_tuning_or_calibration': False,
        'provided_scale': 'Scale_pixel_per_cm/10 from the allowed workbook rows',
        'predictions_source': str(pred_path.relative_to(ROOT)),
        'predictions_sha256': pred_sha,
        'expert_source': str(EXPERT.relative_to(ROOT)),
        'expert_workbook_source': str(WORKBOOK.relative_to(ROOT)),
        'algorithm_columns_used': False,
        'unit_factor_search_performed': False,
        'calibration_or_offset_applied': False,
        'threshold_search_performed': False,
        'missing_values_filled': False,
        'outliers_deleted': False,
        'clipping_applied': False,
        'table1_parameter_subsets': table1,
        'table2_common_MT_PA_FL_G1': table2,
        'table3_FL_A_G1_vs_B': table3,
        'table4_missing_with_reference': table4,
        'expert_rows': expert_rows,
        'expert_disagreement': disagreement,
        'im19_R7_MT_diagnostic': im19_diag,
        'cases': {
            'ids': case_ids,
            'rows': cases,
            'evidence_interpretation': {
                'program_facts': 'statuses, selected ids, counts, extrapolation ratios, failures',
                'visible_pixel_phenomena': 'described only where directly visible in overlays',
                'unverified_hypotheses': 'anatomy/protocol labels are hypotheses, not program facts',
                'model_deviation_does_not_prove_boundary_wrong': True,
                'invalid_diagnostic_is_not_expert_truth': True,
            },
            'g1_failure_evidence_source': {
                'path': str(ROUND19_REVIEW.relative_to(ROOT)),
                'sha256': sha(ROUND19_REVIEW),
                'note': ('candidate-level remaining conditions copied from the '
                         'prior independent review; no new segmentation/inference'),
            },
        },
        'expert_workbook_validation': {
            'workbook_sha256': sha(WORKBOOK),
            'expert_json_sha256': sha(EXPERT),
            'mismatch_count': len(mismatches),
            'workbook_data_rows': len(workbook_rows),
            'expert_json_records': len(expert_doc['records']),
            'allowed_ids': ids,
            'excluded_ids': ['im_07_arch'],
            'unexpected_workbook_ids': sorted(set(workbook_rows) - set(ids)),
            'allowed_rows_present': all(i in workbook_rows for i in ids),
            'n_valid_distribution': {
                p: dict(Counter(expert[i][p]['n_valid'] for i in ids))
                for p in PARAMS},
            'n_positive_total': sum(
                expert[i][p]['n_positive'] for i in ids for p in PARAMS),
            'n_suspicious_total': sum(
                expert[i][p]['n_suspicious'] for i in ids for p in PARAMS),
            'algorithm_columns_used': False,
            'rater_identity_guessed': False,
        },
    }
    # Coverage background table (cached, no new experiment).
    r14 = j(ROUND14)
    v3 = j(V3_COVERAGE)['summary']
    coverage_background = {
        'isolation_dev_282_round13_b': {
            'n_images': 282,
            'source': str(ROUND14.relative_to(ROOT)),
            'G0_output': r14['g0']['n_images_at_least_1'],
            'G1_output': r14['g1']['n_images_at_least_1'],
            'old_stage2_not_current': {'G0': 30, 'G1': 148},
        },
        'external_expert_34_provided_scale': {
            'n_images': 34,
            'G0_output': pred['coverage']['FL_G0'],
            'G1_output': pred['coverage']['FL_G1'],
        },
        'competition_test_309': {
            'n_images': 309,
            'G0_px_output': v3['output_coverage']['FL_G0_px'],
            'G1_px_output': v3['output_coverage']['FL_G1_px_experimental'],
            'strict_v3_scale_reliable': v3['scale']['status_counts'].get('ok'),
            'G0_mm_output': v3['output_coverage']['FL_G0_mm'],
            'G1_mm_output': v3['output_coverage']['FL_G1_mm_experimental'],
        },
        'note': ('end-to-end coverage only; different image collections are '
                 'not a model-generalization comparison and do not provide '
                 'parameter accuracy truth for the 282 isolation images'),
    }
    protocol_table = {
        'source': str(PROTOCOL.relative_to(ROOT)),
        'source_sha256': sha(PROTOCOL),
        'unit_status': protocol['unit_status'],
        'protocol_status': protocol['protocol_status'],
        'parameters': {
            'MT': {'unit_evidence': 'no explicit unit label; scale reconstruction supports working mm hypothesis only',
                   'assumed_unit': 'mm', 'protocol_status': 'partially_confirmed',
                   'unconfirmed': ['line orientation', 'left/middle/right placement', 'endpoint convention']},
            'FL': {'unit_evidence': 'no explicit unit label; scale reconstruction supports working mm hypothesis only',
                   'assumed_unit': 'mm', 'protocol_status': 'unconfirmed',
                   'unconfirmed': ['fascicle segment endpoint convention', 'extrapolation/protocol alignment']},
            'PA': {'unit_evidence': 'no explicit degree label; degree is a working assumption',
                   'assumed_unit': 'degree', 'protocol_status': 'unconfirmed',
                   'unconfirmed': ['angle definition relative to which boundary', 'line selection protocol']},
        },
        'evidence_correction_note': (
            'EVIDENCE.md now carries a dated Round20 correction: dimensional '
            'reconstruction supports a working hypothesis, not explicit '
            'confirmation, and image-bound heuristics are not a general unit proof.'),
    }
    # Write outputs.
    def dump(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(safe(obj), indent=2, ensure_ascii=True),
                        encoding='utf-8')
    dump(out / 'formal/metrics.json', formal)
    dump(out / 'exploratory_assumed_units/metrics.json', exploratory)
    dump(out / 'exploratory_assumed_units/common_sets.json', {
        'table1': table1, 'table2': table2, 'table3': table3,
        'table4': table4})
    dump(out / 'exploratory_assumed_units/expert_rows.json', expert_rows)
    dump(out / 'exploratory_assumed_units/expert_disagreement.json', disagreement)
    dump(out / 'exploratory_assumed_units/cases.json', {'ids': case_ids, 'rows': cases})
    dump(out / 'exploratory_assumed_units/coverage_background.json', coverage_background)
    dump(out / 'exploratory_assumed_units/protocol_status.json', protocol_table)
    # Per-image full table, all 34 images (not only common subsets).
    per_image_records = []
    for r in merged:
        rec = {'image_id': r['image_id'], 'px_per_mm': r['px_per_mm']}
        for method, expert_param, label in (
                ('MT', 'MT', 'MT'), ('PA', 'PA', 'PA'),
                ('FL_G0', 'FL', 'FL_G0'), ('FL_G1', 'FL', 'FL_G1'),
                ('B', 'FL', 'B')):
            e = r['expert'][expert_param]
            p = finite(r['pred'].get(method))
            em = e['mean']; ed = e['median']
            rec.update({
                f'{label}_pred': p,
                f'{label}_ref_mean': em,
                f'{label}_ref_median': ed,
                f'{label}_ref_min': e['min'],
                f'{label}_ref_max': e['max'],
                f'{label}_ref_iqr': e['iqr'],
                f'{label}_ref_n': e['n_valid'],
                f'{label}_error_mean': (p - em if p is not None
                                        and em is not None else None),
                f'{label}_error_median': (p - ed if p is not None
                                          and ed is not None else None),
                f'{label}_status': ((r.get(label) or {}).get('status')
                                    if label != 'B'
                                    else (r['B'] or {}).get('prediction_status')),
                f'{label}_reason': ((r.get(label) or {}).get('reason')
                                    if label != 'B'
                                    else (r['B'] or {}).get('reason')),
            })
        per_image_records.append(rec)
    per_image_fields = list(per_image_records[0].keys())
    with open(out / 'exploratory_assumed_units/per_image.csv', 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=per_image_fields)
        w.writeheader()
        for rec in per_image_records:
            w.writerow(rec)
    dump(out / 'exploratory_assumed_units/per_image.json', {
        'assumed_units': True,
        'n_images': len(per_image_records),
        'reference': 'expert mean; median/min/max/IQR/n also listed',
        'rows': per_image_records,
    })
    # Explicit hypothetical direct errors under the stated unit assumption.
    hypothetical = []
    for r in merged:
        for method, refkey, unit in (('MT', 'MT', 'mm'), ('PA', 'PA', 'degree'),
                                     ('FL_G0', 'FL', 'mm'), ('FL_G1', 'FL', 'mm'),
                                     ('B', 'FL', 'mm')):
            p = finite(r['pred'].get(method)); q = finite(r['ref'].get(refkey))
            hypothetical.append({
                'image_id': r['image_id'], 'method': method, 'unit': unit,
                'prediction': p, 'reference_mean': q,
                'signed_error_prediction_minus_reference': (
                    p - q if p is not None and q is not None else None),
                'absolute_error': (abs(p - q) if p is not None
                                   and q is not None else None),
                'status': 'exploratory_assumed_units',
                'used_in_main_metrics': method in ('MT', 'PA', 'FL_G1'),
                'note': 'not formal accuracy; no tuning/calibration/replacement',
            })
    dump(out / 'exploratory_assumed_units/hypothetical_errors.json', {
        'assumed_units': True,
        'unit_assumption': {'MT': 'mm', 'FL': 'mm', 'PA': 'degree'},
        'n_rows': len(hypothetical), 'rows': hypothetical})
    # Per-expert long table.
    with open(out / 'exploratory_assumed_units/per_expert.csv', 'w',
              newline='', encoding='utf-8') as f:
        fields = ['image_id', 'param', 'rater', 'expert_value',
                  'reference_without_r', 'expert_error', 'expert_abs_error',
                  'model', 'model_error', 'model_abs_error']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for key, label in (('MT', 'MT'), ('PA', 'PA'),
                           ('FL_G1', 'FL_G1'), ('FL_G0_appendix', 'FL_G0'),
                           ('B_appendix', 'B')):
            for rater, entry in disagreement[key]['per_expert'].items():
                for rec in entry['rows']:
                    row = dict(rec)
                    row['param'] = label
                    w.writerow(row)
    # Cases gallery: copy overlays into a page-local assets directory so every
    # src resolves relative to the HTML file, independent of the working tree.
    cases_dir = out / 'exploratory_assumed_units/cases'
    assets_dir = cases_dir / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for c in cases:
        asset = c.get('overlay_asset')
        if asset and c.get('overlay'):
            src = ROOT / c['overlay']
            if src.exists():
                shutil.copy2(src, cases_dir / asset)
            else:
                asset = None
        detail = html.escape(json.dumps({
            'selection_reasons': c['selection_reasons'],
            'provisional_hypotheses': c['provisional_hypotheses'],
            'hypothesis_status': c['hypothesis_status'],
            'expert': c['expert'],
            'model': c['model'],
            'error_vs_expert_mean': c['error_vs_expert_mean'],
            'apo': c['apo'],
        }, indent=2, ensure_ascii=False))
        cards.append(
            f"<section><h2>{html.escape(c['image_id'])}</h2>"
            f"<p><b>Selection:</b> {html.escape(', '.join(c['selection_reasons']))}</p>"
            f"<p><b>Provisional hypotheses (unverified):</b> "
            f"{html.escape(', '.join(c['provisional_hypotheses']))}</p>"
            + (f"<img src='{html.escape(asset)}' style='max-width:760px'>"
               if asset else '<p>overlay not found</p>')
            + f"<pre>{detail}</pre></section>")
    gallery = ("<!doctype html><meta charset='utf-8'><title>Round20 cases</title>"
               "<style>body{background:#111;color:#eee;font-family:sans-serif;margin:20px}"
               "section{border:1px solid #444;padding:12px;margin:20px 0}"
               "pre{white-space:pre-wrap;background:#191919;padding:8px;font-size:12px}"
               "img{border:1px solid #333}</style><h1>Round20 fixed case selection</h1>"
               + "".join(cards))
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / 'index.html').write_text(gallery, encoding='utf-8')
    # Validation record.
    validation = {
        'version': 'round20_evaluation_validation_v1',
        'predictions_sha256': pred_sha,
        'expert_workbook_sha256': sha(WORKBOOK),
        'expert_json_sha256': sha(EXPERT),
        'synthetic_closure_sha256': sha(SYNTH) if SYNTH.exists() else None,
        'expert_workbook_mismatches': len(mismatches),
        'n_images': 34,
        'n_excluded_im_07': 1,
        'coverage': pred['coverage'],
        'common_MT_PA_FL_G1_n': len(common_ids),
        'common_A_G1_B_n': len(ab_ids),
        'common_A_G0_B_n': len(g0b_ids),
        'common_ids_match_round19': set(common_ids) == set(
            round19_eval['main_common_ids']),
        'round19_common_ids': sorted(round19_eval['main_common_ids']),
        'missing_reference_predictions': {k: v['n_missing_with_reference']
                                          for k, v in table4.items()},
        'workbook_data_rows': len(workbook_rows),
        'workbook_allowed_rows_present': all(i in workbook_rows for i in ids),
        'workbook_unexpected_ids': sorted(set(workbook_rows) - set(ids)),
        'expert_n_valid_distribution': {
            p: dict(Counter(expert[i][p]['n_valid'] for i in ids))
            for p in PARAMS},
        'expert_positive_values_total': sum(
            expert[i][p]['n_positive'] for i in ids for p in PARAMS),
        'expert_suspicious_values_total': sum(
            expert[i][p]['n_suspicious'] for i in ids for p in PARAMS),
        'unit_status': protocol['unit_status'],
        'protocol_status': protocol['protocol_status'],
        'formal_absolute_metrics_null': all(
            formal['parameters'][p]['metrics'] == {
                'mean_reference': None, 'median_reference': None}
            for p in formal['parameters']),
        'formal_competition_accuracy': False,
        'exploratory_assumed_units': True,
        'not_kaggle_score': True,
        'no_zero_fill': True,
        'no_nan_or_inf': True,
        'outlier_im_19_R7_retained_in_main': True,
        'algorithm_columns_used': False,
        'rater_identity_guessed': False,
        'round19_spearman_preserved': {
            'A_G1': table3['spearman_expert_mean']['A_G1']['rho'],
            'B': table3['spearman_expert_mean']['B']['rho'],
            'round19_A_G1': round19_eval['main']['expert_mean']['A_G1']['rho'],
            'round19_B': round19_eval['main']['expert_mean']['B']['rho'],
        },
        'round19_spearman_exact_match': (
            table3['spearman_expert_mean']['A_G1']['rho']
            == round19_eval['main']['expert_mean']['A_G1']['rho']
            and table3['spearman_expert_mean']['B']['rho']
            == round19_eval['main']['expert_mean']['B']['rho']),
    }
    dump(out / 'validation.json', validation)
    dump(out / 'input_hashes.json', {
        'parameters_frozen': pred_sha,
        'parameters_frozen_file': sha(PRED),
        'expert_json': sha(EXPERT),
        'expert_workbook': sha(WORKBOOK),
        'protocol_status': sha(PROTOCOL),
        'evidence_file': sha(EVIDENCE),
        'round19_evaluation': sha(ROUND19_EVAL) if ROUND19_EVAL.exists() else None,
        'round19_predictions': sha(ROUND19_PRED) if ROUND19_PRED.exists() else None,
        'round14_summary': sha(ROUND14),
        'v3_coverage': sha(V3_COVERAGE),
        'synthetic_closure': sha(SYNTH) if SYNTH.exists() else None,
        'round19_review': sha(ROUND19_REVIEW) if ROUND19_REVIEW.exists() else None,
        'round20_evaluate_script': sha(Path(__file__)),
        'round20_freeze_script': sha(ROOT / 'scripts/round20_freeze_parameters.py'),
        'round20_tests': sha(ROOT / 'tests/test_round20_measurement.py')
        if (ROOT / 'tests/test_round20_measurement.py').exists() else None,
    })
    print(json.dumps({
        'coverage': pred['coverage'],
        'common_MT_PA_FL_G1': len(common_ids),
        'common_A_G1_B': len(ab_ids),
        'MT_mae': table1['MT']['mean_reference']['metrics']['mae'],
        'PA_mae': table1['PA']['mean_reference']['metrics']['mae'],
        'FL_G1_mae': table1['FL_G1']['mean_reference']['metrics']['mae'],
        'A_G1_rho': table3['spearman_expert_mean']['A_G1']['rho'],
        'B_rho': table3['spearman_expert_mean']['B']['rho'],
        'expert_distribution': {p: dict(Counter(expert[i][p]['n_valid']
                                                for i in ids))
                                for p in PARAMS},
        'case_ids': case_ids,
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
