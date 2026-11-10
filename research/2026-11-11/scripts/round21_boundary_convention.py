"""Round21 boundary-convention experiment: median-run A vs inner-edge E.

This module changes one factor only: for the SAME selected apo runs and the
SAME paired x columns, arm E replaces each run's median y by the inner-edge
foreground pixel y. A is the frozen conservative entry reproduced exactly.

Expert values are never read here. Both arm prediction files are written and
hashed before the independent evaluator runs.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.umud_baseline.apo_baseline import (  # noqa: E402
    load_frozen_rule, measure_conservative,
)
from scripts.fascicle_pa_fl import (  # noqa: E402
    evaluate_segment, extract_segments, select_candidates,
)
from scripts.fl_acceptance_policy import (  # noqa: E402
    policy_fields, select_with_policy,
)
from scripts.pair_selection import (  # noqa: E402
    build_tracks, fit_selected_pair, run_centres_per_column, select_pair,
)

BASE = ROOT / 'runs/geometry_eval/round21_boundary_convention_v1'
EXTERNAL_PRED = ROOT / 'runs/round15_measurement_prototype_v2/measurement/predictions'
ROUND20_PRED = ROOT / ('runs/geometry_eval/round20_measurement_baseline_v1/'
                       'predictions/parameters_frozen.json')
ISOLATION = ROOT / 'runs/geometry_eval/isolation_audit_v1/isolation_view.json'
DEV_FASC = ROOT / ('runs/round13_arm_comparison_v1/predictions/'
                   'b_resnet34_unet/masks')
DEV_APO = ROOT / 'runs/geometry_eval/full_dev_apo_v1/masks'
ROUND14 = ROOT / 'runs/geometry_eval/round14_extrapolation_policy_v1'
ROUND14_IMAGES = ROUND14 / 'round13_b_unet34/images_g0_g1.jsonl'
ROUND14_CANDS = ROUND14 / 'round13_b_unet34/candidates_g0_g1.jsonl'
ROUND14_SUMMARY = ROUND14 / 'round13_b_unet34/summary.json'
TOL = 1e-8
MAX_EXTRAP = 0.80


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def j(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_mask(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L')) > 127


def column_runs(mask):
    """Per-column foreground runs, with the original split rule and members."""
    h, w = mask.shape
    out = []
    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if not len(ys):
            out.append([])
            continue
        splits = np.flatnonzero(np.diff(ys) > max(2, round(h * .002))) + 1
        runs = []
        for part in np.split(ys, splits):
            runs.append({
                'ys': [int(v) for v in part],
                'min': int(part[0]), 'max': int(part[-1]),
                'median': float(np.median(part)),
                'n': int(len(part)),
            })
        out.append(runs)
    return out


def baseline_grid(mask):
    """Trace the exact baseline extract_apo_bands paired columns."""
    h, w = mask.shape
    cols = column_runs(mask)
    grid = []
    for x, runs in enumerate(cols):
        if len(runs) != 2:
            continue
        sup, deep = runs
        if deep['median'] - sup['median'] <= max(2, h * .005):
            continue
        grid.append({
            'x': x,
            'sup': sup, 'deep': deep,
            'sup_index': 0, 'deep_index': 1,
        })
    return grid


def _point_run_index(runs, y):
    hits = [i for i, r in enumerate(runs)
            if abs(float(r['median']) - float(y)) <= 1e-10]
    if len(hits) != 1:
        raise ValueError(f'ambiguous run binding at y={y}: {len(hits)} hits')
    return hits[0]


def selected_pair_trace(mask, rule):
    """Replay the frozen new_pair_selection branch and bind each selected point."""
    h, w = mask.shape
    runs = run_centres_per_column(mask)
    cols = column_runs(mask)
    tracks = build_tracks(runs, track_tol=rule.get('track_tol_px', 8.0),
                          max_gap=rule.get('max_gap_columns', 3))
    selected, reason, meta = select_pair(tracks, w, h, rule)
    if reason:
        return None, reason, meta
    sup_track, deep_track, common_x = selected
    fits, reason = fit_selected_pair(sup_track, deep_track, common_x, w, h)
    if reason:
        return None, reason, meta
    sup_map = {p[0]: p[1] for p in sup_track['points']}
    deep_map = {p[0]: p[1] for p in deep_track['points']}
    grid = []
    for x in [int(v) for v in fits['paired_x']]:
        sup_y = sup_map[x]
        deep_y = deep_map[x]
        sup_idx = _point_run_index(cols[x], sup_y)
        deep_idx = _point_run_index(cols[x], deep_y)
        grid.append({
            'x': x,
            'sup': cols[x][sup_idx], 'deep': cols[x][deep_idx],
            'sup_index': sup_idx, 'deep_index': deep_idx,
        })
    return grid, None, meta


def edge_values(grid):
    """Inner-edge pixel center per selected column.

    For the upper run, use max foreground y; for the lower run, min foreground
    y. No +/-0.5 is added; values are integer pixel coordinates.
    """
    return {
        'xs': [int(g['x']) for g in grid],
        'sup': [int(g['sup']['max']) for g in grid],
        'deep': [int(g['deep']['min']) for g in grid],
        'sup_median': [float(g['sup']['median']) for g in grid],
        'deep_median': [float(g['deep']['median']) for g in grid],
    }


def fits_from_grid(grid, template, edge=False):
    """Fit A (median) or E (edge) with the same polynomial/check thresholds."""
    h = int(template['image_shape'][0])
    xs = np.asarray([g['x'] for g in grid], dtype=float)
    if edge:
        sup = np.asarray([g['sup']['max'] for g in grid], dtype=float)
        deep = np.asarray([g['deep']['min'] for g in grid], dtype=float)
    else:
        sup = np.asarray([g['sup']['median'] for g in grid], dtype=float)
        deep = np.asarray([g['deep']['median'] for g in grid], dtype=float)
    if len(xs) < max(8, .3 * int(template['image_shape'][1])):
        return None, ['insufficient_paired_support']
    if xs[-1] - xs[0] < .3 * int(template['image_shape'][1]):
        return None, ['short_common_support']
    fits = {'sup': np.polyfit(xs, sup, 1),
            'deep': np.polyfit(xs, deep, 1)}
    residuals = [np.abs(np.polyval(fits[k], xs) - v)
                 for k, v in (('sup', sup), ('deep', deep))]
    reasons = []
    if max(np.quantile(v, .9) for v in residuals) > max(3, .02 * h):
        reasons.append('nonlinear_or_unstable_bands')
    gap = np.polyval(fits['deep'] - fits['sup'], [xs[0], xs[-1]])
    if (gap <= 0).any():
        reasons.append('crossed_or_inverted_lines')
    if reasons:
        return None, reasons
    out = {
        'sup': fits['sup'], 'deep': fits['deep'],
        'support': [int(xs[0]), int(xs[-1])],
        'paired_x': xs.astype(int),
        'max_sample_gap': float(template['max_sample_gap']),
        'image_shape': tuple(template['image_shape']),
        'roi_source': template.get('roi_source', 'mask_paired_support'),
    }
    return out, []


def trace_for_A(mask, rule, a_out):
    """Return a trace bound to the exact A branch, or an explicit failure."""
    if a_out.get('status') != 'candidate' or not isinstance(a_out.get('fits'), dict):
        return {
            'status': 'apo_rejected', 'branch': a_out.get('source'),
            'reason': a_out.get('reason', ''), 'grid': None, 'meta': None,
        }
    source = a_out.get('source')
    # The frozen entry calls both baseline and new selection; use the branch
    # that actually supplied A's fits.
    if source == 'baseline':
        grid = baseline_grid(mask)
        branch = 'baseline'
        meta = {}
        expected_x = [int(x) for x in a_out['fits']['paired_x']]
        if [g['x'] for g in grid] != expected_x:
            raise ValueError('baseline grid does not match A paired_x')
    elif source == 'new_pair_selection':
        grid, reason, meta = selected_pair_trace(mask, rule)
        if reason:
            raise ValueError(f'new-pair trace failed after A candidate: {reason}')
        branch = 'new_pair_selection'
    else:
        raise ValueError(f'unknown A source: {source}')
    return {'status': 'ok', 'branch': branch, 'reason': '', 'grid': grid,
            'meta': meta}


def make_fits_from_trace(trace, template, edge=False):
    if trace['status'] != 'ok':
        return None, [trace.get('reason') or 'apo_rejected']
    return fits_from_grid(trace['grid'], template, edge=edge)


def fl_angles(seg, fits):
    """Acute angle between the candidate direction and each boundary line."""
    ang = math.radians(float(seg['direction_deg']))
    vx, vy = math.cos(ang), math.sin(ang)
    out = {}
    for key in ('sup', 'deep'):
        m = float(fits[key][0])
        b = 1.0 / math.sqrt(1.0 + m * m)
        cosang = abs(vx * 1.0 * b + vy * m * b)
        out[f'{key}_acute_deg'] = float(math.degrees(math.acos(
            max(-1.0, min(1.0, cosang)))))
    return out


def candidate_record(c, policy, width, height, apo_fits, apo_ok):
    seg = c['segment']
    fl_fields = policy_fields(c, policy)
    angles = fl_angles(seg, apo_fits) if apo_fits is not None else {}
    return {
        'candidate_identity': f"{c.get('image_id', '')}:{seg['label_id']}",
        'candidate_index': int(c.get('candidate_index', -1)),
        'label_id': int(seg['label_id']),
        'segment': seg,
        'apo_status': 'candidate' if apo_ok else 'rejected',
        'raw_fl_px': c.get('fl_px_raw'),
        'visible_length_px': c.get('visible_length_px'),
        'extrapolated_length_px': c.get('extrapolated_length_px'),
        'extrapolation_ratio': c.get('extrapolation_ratio'),
        'intersection_sup': c.get('intersection_sup'),
        'intersection_deep': c.get('intersection_deep'),
        'outside_image': (bool(c.get('fl_outside_image'))
                          if c.get('fl_px_raw') is not None else None),
        'outside_apo_support': (bool(c.get('fl_outside_apo_support'))
                                if c.get('fl_px_raw') is not None else None),
        'local_support_missing': (bool(c.get('fl_local_support_missing'))
                                  if c.get('fl_px_raw') is not None else None),
        'failure_conditions': c.get('fl_failure_conditions_pure'),
        **angles,
        'pa_deg': c.get('pa_deg'),
        'pa_status': c.get('pa_status'),
        'pa_conditions': c.get('pa_failure_conditions'),
        'policy': policy,
        'policy_accept': bool(fl_fields['accept']),
        'policy_conditions': fl_fields['conditions'],
        'policy_fl_px': fl_fields['fl_px_pure'],
        'candidate_validity_status': c.get('candidate_validity_status'),
        'candidate_validity_conditions': c.get('candidate_validity_conditions'),
        'region_status': c.get('region_status'),
        'local_support_status': c.get('local_support_status'),
    }


def evaluate_with_fits(segments, fits, apo_ok, width, height, image_id):
    """Evaluate identical segments with a supplied apo fit (A or E)."""
    out = []
    for i, seg in enumerate(segments):
        ev = evaluate_segment(seg, fits, width, height, apo_ok=apo_ok)
        ev['image_id'] = image_id
        ev['candidate_index'] = i
        out.append(ev)
    return out


def selection_bundle(cands):
    """A-policy and G1-policy selection from the original selector."""
    g0 = select_candidates(cands)
    g1, adjusted = select_with_policy(cands, 'G1')
    g0_ids = [int(c['segment']['label_id']) for c in g0['fl']]
    g1_ids = [int(c['segment']['label_id']) for c in g1['fl']]
    g0_qualified = [int(c['segment']['label_id']) for c in cands
                    if c.get('fl_accept_pure')]
    qualified = [int(c['segment']['label_id']) for c in adjusted
                 if c.get('policy_accept_pure')]
    return {
        'g0': g0, 'g1': g1, 'g1_adjusted': adjusted,
        'g0_ids': g0_ids, 'g1_ids': g1_ids,
        'g0_qualified_ids': g0_qualified,
        'g1_qualified_ids': qualified,
        'g0_selected_count': len(g0_ids),
        'g1_selected_count': len(g1_ids),
        'g0_qualified_count': len(g0_qualified),
        'g1_qualified_count': len(qualified),
    }


def fixed_metrics(selected_ids, candidates, fits, apo_ok, width, height,
                  policy):
    """Recompute the A-selected identities with E, never partially averaging."""
    by_label = {int(c['segment']['label_id']): c for c in candidates}
    rows = []
    values = []
    all_valid = True
    for label in selected_ids:
        c = by_label.get(int(label))
        if c is None:
            all_valid = False
            rows.append({'label_id': int(label), 'missing_candidate': True})
            continue
        pf = policy_fields(c, policy)
        rows.append({
            'label_id': int(label),
            'raw_fl_px': c.get('fl_px_raw'),
            'visible_length_px': c.get('visible_length_px'),
            'extrapolation_ratio': c.get('extrapolation_ratio'),
            'intersection_sup': c.get('intersection_sup'),
            'intersection_deep': c.get('intersection_deep'),
            'outside_image': (bool(c.get('fl_outside_image'))
                              if c.get('fl_px_raw') is not None else None),
            'outside_apo_support': (bool(c.get('fl_outside_apo_support'))
                                    if c.get('fl_px_raw') is not None else None),
            'conditions': pf['conditions'],
            'accept': bool(pf['accept']),
            'value_px': pf['fl_px_pure'],
            'validity': c.get('candidate_validity_status'),
            'validity_conditions': c.get('candidate_validity_conditions'),
        })
        if not pf['accept'] or pf['fl_px_pure'] is None:
            all_valid = False
        else:
            values.append(float(pf['fl_px_pure']))
    return {
        'selected_ids': [int(v) for v in selected_ids],
        'rows': rows,
        'all_selected_valid': bool(all_valid),
        'mean_px': float(np.mean(values)) if all_valid and values else None,
        'n_selected': len(selected_ids),
        'n_valid_values': len(values),
    }


def mt_value(fits, positions):
    from src.umud_baseline.geometry import mt_at_positions
    return mt_at_positions(fits, positions)


def candidate_table(image_id, dataset, cands, arm, selected_g0, selected_g1,
                    qualified_g1, apo_fits, apo_ok):
    rows = []
    for c in cands:
        rec = candidate_record(c, 'G1', 0, 0, apo_fits, apo_ok)
        rec.update({
            'image_id': image_id, 'dataset': dataset, 'arm': arm,
            'g0_accept': bool(c.get('fl_accept_pure')),
            'g1_accept': bool(policy_fields(c, 'G1')['accept']),
            'g1_qualified': bool(policy_fields(c, 'G1')['accept']),
            'a_g0_selected': int(c['segment']['label_id']) in selected_g0,
            'a_g1_selected': int(c['segment']['label_id']) in selected_g1,
            'g1_selected': int(c['segment']['label_id']) in selected_g1,
            'g1_qualified_flag': int(c['segment']['label_id']) in qualified_g1,
        })
        rows.append(rec)
    return rows


def image_arm_record(image_id, dataset, path, group_id, width, height,
                     apo_mask, fasc_mask, rule, rule_hash, px_per_mm=None):
    """Generate A and E geometry for one image without expert input."""
    a_out = measure_conservative(apo_mask, rule)
    segments = extract_segments(fasc_mask)['segments']
    a_fits = a_out.get('fits') if a_out.get('status') == 'candidate' else None
    a_ok = a_fits is not None
    a_cands = evaluate_with_fits(segments, a_fits, a_ok, width, height, image_id)
    a_sel = selection_bundle(a_cands)
    trace = trace_for_A(apo_mask, rule, a_out)
    e_fits, e_reasons = make_fits_from_trace(trace, a_out.get('fits') or {
        'image_shape': (height, width), 'max_sample_gap': max(2, .015 * width),
        'roi_source': 'unavailable'}, edge=True) if trace['status'] == 'ok' else (
        None, [trace.get('reason') or 'apo_rejected'])
    e_ok = e_fits is not None
    e_cands = evaluate_with_fits(segments, e_fits, e_ok, width, height, image_id)
    e_sel = selection_bundle(e_cands)

    # Fixed identities: MT positions and A-selected PA/G0/G1 identities.
    a_mt_positions = ((a_out.get('result') or {}).get('x_positions') or [])
    a_mt = None
    e_mt = None
    a_mt_reason = ''
    e_mt_reason = ''
    if a_mt_positions and a_fits is not None:
        a_mt, a_mt_reason = mt_value(a_fits, a_mt_positions)
    if a_mt_positions and e_fits is not None:
        e_mt, e_mt_reason = mt_value(e_fits, a_mt_positions)
    elif a_mt_positions:
        e_mt_reason = ';'.join(e_reasons) or 'e_fit_failed'

    a_pa_ids = [int(c['segment']['label_id']) for c in a_sel['g0']['pa']]
    e_pa = fixed_metrics(a_pa_ids, e_cands, e_fits, e_ok, width, height, 'G0')
    # PA fixed identity uses the same identities; store angle rows separately.
    by_label_a = {int(c['segment']['label_id']): c for c in a_cands}
    by_label_e = {int(c['segment']['label_id']): c for c in e_cands}
    e_pa_rows = []
    e_pa_values = []
    for label in a_pa_ids:
        ac = by_label_a.get(label); ec = by_label_e.get(label)
        e_pa_rows.append({
            'label_id': label,
            'a_pa_deg': ac.get('pa_deg') if ac else None,
            'e_pa_deg': ec.get('pa_deg') if ec else None,
            'a_status': ac.get('pa_status') if ac else 'missing',
            'e_status': ec.get('pa_status') if ec else 'missing',
            'e_conditions': (ec.get('pa_failure_conditions') if ec
                             else ['no_fits']),
        })
        if ec and ec.get('pa_status') == 'ok' and ec.get('pa_deg') is not None:
            e_pa_values.append(float(ec['pa_deg']))
    e_pa_mean = (float(np.mean(e_pa_values))
                 if len(e_pa_values) == len(a_pa_ids) and a_pa_ids else None)

    # Fixed FL identities, A G0 and G1.
    e_g0_fixed = fixed_metrics(a_sel['g0_ids'], e_cands, e_fits, e_ok,
                               width, height, 'G0')
    e_g1_fixed = fixed_metrics(a_sel['g1_ids'], e_cands, e_fits, e_ok,
                               width, height, 'G1')

    def fl_summary(cands, selected_ids, qualified_ids, selected_count,
                   mean_px, policy):
        return {
            'candidate_count': len(cands),
            'selected_count': selected_count,
            'qualified_count': len(qualified_ids),
            'selected_ids': [int(v) for v in selected_ids],
            'qualified_ids': [int(v) for v in qualified_ids],
            'selected_mean_px': mean_px,
            'selected_mean_mm': (None if px_per_mm is None or mean_px is None
                                 else mean_px / px_per_mm),
            'policy': policy,
        }

    a_g0 = fl_summary(a_cands, a_sel['g0_ids'], a_sel['g0_qualified_ids'],
                      a_sel['g0_selected_count'], a_sel['g0']['fl_mean_px'],
                      'G0')
    a_g1 = fl_summary(a_cands, a_sel['g1_ids'], a_sel['g1_qualified_ids'],
                      a_sel['g1_selected_count'], a_sel['g1']['fl_mean_px'],
                      'G1')
    e_g0_reselected = fl_summary(
        e_cands, e_sel['g0_ids'], e_sel['g0_qualified_ids'],
        e_sel['g0_selected_count'], e_sel['g0']['fl_mean_px'], 'G0')
    e_g1_reselected = fl_summary(
        e_cands, e_sel['g1_ids'], e_sel['g1_qualified_ids'],
        e_sel['g1_selected_count'], e_sel['g1']['fl_mean_px'], 'G1')
    a_record = {
        'image_id': image_id, 'dataset': dataset, 'path': str(path),
        'group_id': group_id, 'width': width, 'height': height,
        'px_per_mm': px_per_mm,
        'apo': {
            'status': a_out.get('status'), 'reason': a_out.get('reason', ''),
            'source': a_out.get('source'), 'baseline_status': a_out.get('baseline_status'),
            'baseline_reason': a_out.get('baseline_reason'),
            'new_status': a_out.get('new_status'),
            'new_reason': a_out.get('new_reason'),
            'support': (list(a_fits['support']) if a_fits is not None else None),
            'fits': ({'sup': [float(v) for v in a_fits['sup']],
                      'deep': [float(v) for v in a_fits['deep']]}
                     if a_fits is not None else None),
            'paired_x': ([int(v) for v in a_fits['paired_x']]
                         if a_fits is not None else None),
            'max_sample_gap': (float(a_fits['max_sample_gap'])
                               if a_fits is not None else None),
            'trace_branch': trace.get('branch'),
            'trace_status': trace.get('status'),
        },
        'MT': {
            'positions': a_mt_positions, 'values_px': a_mt,
            'mean_px': float(np.mean(a_mt)) if a_mt else None,
            'mean_mm': (float(np.mean(a_mt)) / px_per_mm
                        if a_mt and px_per_mm else None),
            'status': 'ok' if a_mt else 'failed', 'reason': a_mt_reason,
        },
        'PA': {
            'selected_ids': a_pa_ids,
            'selected_count': len(a_pa_ids),
            'mean_deg': a_sel['g0']['pa_mean_deg'],
            'status': 'ok' if a_sel['g0']['pa_mean_deg'] is not None else 'failed',
        },
        'FL_G0': a_g0,
        'FL_G1': a_g1,
        'candidate_count': len(a_cands),
    }
    e_record = {
        'image_id': image_id, 'dataset': dataset, 'path': str(path),
        'group_id': group_id, 'width': width, 'height': height,
        'px_per_mm': px_per_mm,
        'apo': {
            'status': ('candidate' if e_ok else 'failed'),
            'reason': ';'.join(e_reasons) if e_reasons else '',
            'fit_status': 'ok' if e_ok else 'failed',
            'fit_reasons': e_reasons,
            'support': (list(e_fits['support']) if e_fits is not None else None),
            'fits': ({'sup': [float(v) for v in e_fits['sup']],
                      'deep': [float(v) for v in e_fits['deep']]}
                     if e_fits is not None else None),
            'paired_x': ([int(v) for v in e_fits['paired_x']]
                         if e_fits is not None else None),
            'max_sample_gap': (float(e_fits['max_sample_gap'])
                               if e_fits is not None else None),
        },
        'MT': {
            'positions': a_mt_positions, 'values_px': e_mt,
            'mean_px': float(np.mean(e_mt)) if e_mt else None,
            'mean_mm': (float(np.mean(e_mt)) / px_per_mm
                        if e_mt and px_per_mm else None),
            'status': 'ok' if e_mt else 'failed', 'reason': e_mt_reason,
        },
        'PA_fixed': {
            'selected_ids': a_pa_ids, 'mean_deg': e_pa_mean,
            'rows': e_pa_rows,
            'status': 'ok' if e_pa_mean is not None else 'failed',
        },
        'PA_reselected': {
            'selected_ids': [int(c['segment']['label_id'])
                             for c in e_sel['g0']['pa']],
            'selected_count': e_sel['g0']['n_pa_selected'],
            'mean_deg': e_sel['g0']['pa_mean_deg'],
            'status': ('ok' if e_sel['g0']['pa_mean_deg'] is not None
                       else 'failed'),
        },
        'FL_G0_fixed': e_g0_fixed,
        'FL_G1_fixed': e_g1_fixed,
        'FL_G0_reselected': e_g0_reselected,
        'FL_G1_reselected': e_g1_reselected,
        'candidate_count': len(e_cands),
    }
    return a_record, e_record, trace, a_cands, e_cands


def parse_external():
    pred = j(ROUND20_PRED)
    records = []
    for r in pred['records']:
        iid = r['image_id']
        ap = EXTERNAL_PRED / f'{iid}_apo.png'
        fp = EXTERNAL_PRED / f'{iid}_fasc.png'
        records.append({
            'image_id': iid, 'dataset': 'external34', 'path': r['path'],
            'group_id': iid, 'width': int(r['width']), 'height': int(r['height']),
            'px_per_mm': r['px_per_mm'], 'apo_mask': ap, 'fasc_mask': fp,
            'frozen': r,
        })
    if len(records) != 34:
        raise ValueError('external34 must have 34 images')
    return records


def parse_dev():
    view = j(ISOLATION)
    names = view['included_filenames']
    if len(names) != 282 or len(set(names)) != 282:
        raise ValueError('isolation view is not 282 unique images')
    images = [json.loads(l) for l in ROUND14_IMAGES.read_text(
        encoding='utf-8').splitlines()]
    by_name = {r['filename']: r for r in images}
    if set(by_name) != set(names):
        raise ValueError('Round14 image mapping differs from isolation view')
    if len({r['group_id'] for r in images}) != 279:
        raise ValueError('Round14 mapping is not 279 groups')
    out = []
    for fn in names:
        r = by_name[fn]
        out.append({
            'image_id': fn, 'dataset': 'dev282', 'path': fn,
            'group_id': r['group_id'], 'width': int(r['width']),
            'height': int(r['height']), 'px_per_mm': None,
            'apo_mask': DEV_APO / (Path(fn).stem + '.png'),
            'fasc_mask': DEV_FASC / (Path(fn).stem + '.png'),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(BASE / 'config.json'))
    ap.add_argument('--out', default=str(BASE / 'predictions'))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--skip-replay', action='store_true')
    args = ap.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f'Round21 config must exist before extraction: {config_path}')
    config = j(config_path)
    if config.get('frozen_before_extraction') is not True:
        raise ValueError('Round21 config not marked frozen_before_extraction')
    edge_def = config.get('arms', {}).get('E', {}).get('definition', {})
    if edge_def.get('upper_run') != 'max foreground y':
        raise ValueError('unexpected edge definition')
    if edge_def.get('lower_run') != 'min foreground y':
        raise ValueError('unexpected edge definition')
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    rule, rule_hash = load_frozen_rule()
    records = parse_external() + parse_dev()
    if args.limit:
        records = records[:args.limit]
    ext = []
    dev = []
    traces_ext = []
    traces_dev = []
    cand_ext_a = []
    cand_ext_e = []
    cand_dev_a = []
    cand_dev_e = []
    for r in records:
        apo = read_mask(r['apo_mask'])
        fasc = read_mask(r['fasc_mask'])
        if apo.shape != (r['height'], r['width']) or fasc.shape != (r['height'], r['width']):
            raise ValueError(f"{r['image_id']}: mask shape mismatch")
        a, e, trace, ac, ec = image_arm_record(
            r['image_id'], r['dataset'], r['path'], r['group_id'],
            r['width'], r['height'], apo, fasc, rule, rule_hash,
            r['px_per_mm'])
        trace_row = {
            'image_id': r['image_id'], 'dataset': r['dataset'],
            'branch': trace.get('branch'), 'status': trace.get('status'),
            'reason': trace.get('reason'),
            'columns': [{
                'x': g['x'],
                'sup_ys': g['sup']['ys'], 'deep_ys': g['deep']['ys'],
                'sup_median': g['sup']['median'],
                'deep_median': g['deep']['median'],
                'sup_edge': g['sup']['max'], 'deep_edge': g['deep']['min'],
            } for g in (trace.get('grid') or [])],
        }
        if r['dataset'] == 'external34':
            ext.append((r, a, e, ac, ec))
            traces_ext.append(trace_row)
            cand_ext_a.extend(candidate_table(
                r['image_id'], r['dataset'], ac, 'A',
                a['FL_G0']['selected_ids'], a['FL_G1']['selected_ids'],
                a['FL_G1']['qualified_ids'],
                a['apo'].get('fits'), a['apo']['status'] == 'candidate'))
            cand_ext_e.extend(candidate_table(
                r['image_id'], r['dataset'], ec, 'E',
                e['FL_G0_reselected']['selected_ids'],
                e['FL_G1_reselected']['selected_ids'],
                e['FL_G1_reselected']['qualified_ids'],
                None if e['apo'].get('fits') is None else e['apo']['fits'],
                e['apo']['status'] == 'candidate'))
        else:
            dev.append((r, a, e, ac, ec))
            traces_dev.append(trace_row)
            cand_dev_a.extend(candidate_table(
                r['image_id'], r['dataset'], ac, 'A',
                a['FL_G0']['selected_ids'], a['FL_G1']['selected_ids'],
                a['FL_G1']['qualified_ids'],
                a['apo'].get('fits'), a['apo']['status'] == 'candidate'))
            cand_dev_e.extend(candidate_table(
                r['image_id'], r['dataset'], ec, 'E',
                e['FL_G0_reselected']['selected_ids'],
                e['FL_G1_reselected']['selected_ids'],
                e['FL_G1_reselected']['qualified_ids'],
                e['apo'].get('fits'), e['apo']['status'] == 'candidate'))
        print(r['dataset'], r['image_id'], a['apo']['status'],
              a['apo']['source'], a['FL_G1']['selected_count'],
              e['FL_G1_reselected']['selected_count'], flush=True)
    # A replay checks before any E result is accepted.
    replay = replay_checks(ext, dev, rule)
    (out / 'replay_validation.json').write_text(
        json.dumps(replay, indent=2, ensure_ascii=True), encoding='utf-8')
    if not args.skip_replay and not replay['all_passed']:
        raise RuntimeError(f"Round21 A replay failed: {replay['failures'][:5]}")
    ext_a = [a for _, a, _, _, _ in ext]
    ext_e = [e for _, _, e, _, _ in ext]
    dev_a = [a for _, a, _, _, _ in dev]
    dev_e = [e for _, _, e, _, _ in dev]
    for name, rows in (('external34_A', ext_a), ('external34_E', ext_e),
                       ('dev282_A', dev_a), ('dev282_E', dev_e)):
        (out / f'{name}.json').write_text(
            json.dumps(rows, indent=2, ensure_ascii=True), encoding='utf-8')
    for name, rows in (('external34_trace', traces_ext),
                       ('dev282_trace', traces_dev),
                       ('external34_candidates_A', cand_ext_a),
                       ('external34_candidates_E', cand_ext_e),
                       ('dev282_candidates_A', cand_dev_a),
                       ('dev282_candidates_E', cand_dev_e)):
        with open(out / f'{name}.jsonl', 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + '\n')
    manifest = {
        'version': 'round21_predictions_v1', 'config_sha256': sha(config_path),
        'rule_sha256': rule_hash,
        'external34': {'A': sha(out / 'external34_A.json'),
                       'E': sha(out / 'external34_E.json'),
                       'trace': sha(out / 'external34_trace.jsonl'),
                       'candidates_A': sha(out / 'external34_candidates_A.jsonl'),
                       'candidates_E': sha(out / 'external34_candidates_E.jsonl')},
        'dev282': {'A': sha(out / 'dev282_A.json'),
                   'E': sha(out / 'dev282_E.json'),
                   'trace': sha(out / 'dev282_trace.jsonl'),
                   'candidates_A': sha(out / 'dev282_candidates_A.jsonl'),
                   'candidates_E': sha(out / 'dev282_candidates_E.jsonl')},
        'n_external34': len(ext_a), 'n_dev282': len(dev_a),
        'expert_values_read': False,
        'replay': replay,
    }
    (out / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding='utf-8')
    (out / 'manifest.sha256').write_text(
        sha(out / 'manifest.json') + '\n', encoding='utf-8')
    print(json.dumps({'manifest_sha256': sha(out / 'manifest.json'),
                      'replay': {k: v for k, v in replay.items()
                                 if k != 'failures'}}, indent=2))


def replay_checks(ext, dev, rule):
    """A-only replay checks against frozen Round20/Round14 references."""
    failures = []
    ext_ok = 0
    qualified = []
    selected = []
    for r, a, e, ac, ec in ext:
        f = r['frozen']
        if a['apo']['status'] != f['apo'].get('status'):
            failures.append(f"{r['image_id']}: apo status mismatch")
        if a['apo']['source'] != f['apo'].get('source'):
            failures.append(f"{r['image_id']}: apo source mismatch")
        if a['apo']['support'] != f['apo']['support']:
            failures.append(f"{r['image_id']}: apo support mismatch")
        if a['MT']['positions'] != f['MT'].get('selected_positions_px'):
            failures.append(f"{r['image_id']}: MT positions mismatch")
        if a['PA']['selected_ids'] != [
                int(str(x).split(':')[-1])
                for x in (f['PA'].get('selected_candidate_identities') or [])]:
            failures.append(f"{r['image_id']}: PA selected identity mismatch")
        for key, fkey, valkey in (
                ('MT', 'MT', 'value_px'), ('PA', 'PA', 'value_deg')):
            av = a[key].get('mean_px') if key == 'MT' else a[key].get('mean_deg')
            fv = f[key].get(valkey)
            if (av is None) != (fv is None):
                failures.append(f"{r['image_id']}: {key} availability mismatch")
            elif av is not None and abs(float(av) - float(fv)) > TOL:
                failures.append(f"{r['image_id']}: {key} value mismatch")
        for key, fkey in (('FL_G0', 'FL_G0'), ('FL_G1', 'FL_G1')):
            av = a[key]['selected_mean_px']; fv = f[fkey].get('selected_mean_px')
            if (av is None) != (fv is None):
                failures.append(f"{r['image_id']}: {key} availability mismatch")
            elif av is not None and abs(float(av) - float(fv)) > TOL:
                failures.append(f"{r['image_id']}: {key} value mismatch")
            aids = [int(x) for x in a[key]['selected_ids']]
            fids = [int(str(x).split(':')[-1])
                    for x in (f[fkey].get('selected_candidate_identities') or [])]
            if aids != fids:
                failures.append(f"{r['image_id']}: {key} selected identity mismatch")
        ext_ok += int(not any(x.startswith(r['image_id'] + ':')
                              for x in failures))
        qualified.extend((r['image_id'], x) for x in a['FL_G1']['qualified_ids'])
        selected.extend((r['image_id'], x) for x in a['FL_G1']['selected_ids'])
    unselected = sorted(set(qualified) - set(selected))
    if len(qualified) != 55:
        failures.append(f'external G1 qualified {len(qualified)} != 55')
    if len(selected) != 47:
        failures.append(f'external G1 selected {len(selected)} != 47')
    allowed_unselected = {'im_10_arch', 'im_27_arch', 'im_35_arch'}
    if {i for i, _ in unselected} - allowed_unselected:
        failures.append(f'unexpected external unselected images: '
                        f'{sorted({i for i, _ in unselected} - allowed_unselected)}')

    # Round14 per-image / candidate comparison.
    ref_images = {r['filename']: r for r in
                  (json.loads(l) for l in ROUND14_IMAGES.read_text(
                      encoding='utf-8').splitlines())}
    ref_cands = {}
    for line in ROUND14_CANDS.read_text(encoding='utf-8').splitlines():
        c = json.loads(line)
        ref_cands.setdefault(c['fasc_filename'], []).append(c)
    dev_ok = 0
    for r, a, e, ac, ec in dev:
        ref = ref_images[r['image_id']]
        if a['apo']['status'] != ref['apo_status']:
            failures.append(f"{r['image_id']}: apo status mismatch")
        for key, akey in (('g0_n', 'FL_G0'), ('g1_n', 'FL_G1')):
            if int(a[akey]['selected_count']) != int(ref[key]):
                failures.append(f"{r['image_id']}: {key} mismatch")
        # Round14 image rows store sorted label-id sets; selection order is
        # checked separately at candidate level.
        if sorted(a['FL_G0']['selected_ids']) != sorted(ref['g0_ids']):
            failures.append(f"{r['image_id']}: g0 ids mismatch")
        if sorted(a['FL_G1']['selected_ids']) != sorted(ref['g1_ids']):
            failures.append(f"{r['image_id']}: g1 ids mismatch")
        refs = sorted(ref_cands.get(r['image_id'], []),
                      key=lambda x: x['candidate_index'])
        if len(refs) != len(ac):
            failures.append(f"{r['image_id']}: candidate count mismatch")
        else:
            for c, rc in zip(ac, refs):
                iid = r['image_id']
                label = int(c['segment']['label_id'])
                if label != int(rc['segment']['label_id']):
                    failures.append(f'{iid}:{label}: label mismatch')
                for got, want in (
                        (c.get('fl_px_raw'), rc.get('raw_fl_px')),
                        (c.get('visible_length_px'), rc.get('visible_length_px')),
                        (c.get('extrapolation_ratio'), rc.get('extrapolation_ratio'))):
                    if (got is None) != (want is None):
                        failures.append(f'{iid}:{label}: finite mismatch')
                    elif got is not None and abs(float(got) - float(want)) > TOL:
                        failures.append(f'{iid}:{label}: raw value mismatch')
                for got, want in (
                        (c.get('intersection_sup'), rc.get('intersection_sup')),
                        (c.get('intersection_deep'), rc.get('intersection_deep'))):
                    if got is None or want is None:
                        if got != want:
                            failures.append(f'{iid}:{label}: intersection missing')
                    elif max(abs(float(got[j]) - float(want[j]))
                             for j in (0, 1)) > TOL:
                        failures.append(f'{iid}:{label}: intersection mismatch')
                if bool(c.get('fl_accept_pure')) != bool(rc.get('g0_accept_pure')):
                    failures.append(f'{iid}:{label}: G0 accept mismatch')
                if bool(policy_fields(c, 'G1')['accept']) != bool(
                        rc.get('g1_accept_pure')):
                    failures.append(f'{iid}:{label}: G1 accept mismatch')
                if (label in a['FL_G0']['selected_ids']) != bool(
                        rc.get('g0_selected')):
                    failures.append(f'{iid}:{label}: G0 selected mismatch')
                if (label in a['FL_G1']['selected_ids']) != bool(
                        rc.get('g1_selected')):
                    failures.append(f'{iid}:{label}: G1 selected mismatch')
                if c.get('candidate_validity_status') != rc.get(
                        'candidate_validity_status'):
                    failures.append(f'{iid}:{label}: validity mismatch')
        dev_ok += int(not any(x.startswith(r['image_id'] + ':')
                              for x in failures))
    summary = j(ROUND14_SUMMARY)
    g0_images = sum(1 for row in dev if row[1]['FL_G0']['selected_count'] >= 1)
    g1_images = sum(1 for row in dev if row[1]['FL_G1']['selected_count'] >= 1)
    g1_three = sum(1 for row in dev if row[1]['FL_G1']['selected_count'] >= 3)
    g1_qual = sum(row[1]['FL_G1']['qualified_count'] for row in dev)
    g1_sel = sum(row[1]['FL_G1']['selected_count'] for row in dev)
    expected = {
        'n_images': 282, 'n_groups': 279, 'g0_images': 17,
        'g1_images': 154, 'g1_three': 21, 'g1_qualified': 252,
        'g1_selected': 244,
    }
    observed = {'n_images': len(dev), 'n_groups': len({row[1]['group_id'] for row in dev}),
                'g0_images': g0_images, 'g1_images': g1_images,
                'g1_three': g1_three, 'g1_qualified': g1_qual,
                'g1_selected': g1_sel}
    for k, v in expected.items():
        if observed[k] != v:
            failures.append(f'dev282 {k}: {observed[k]} != {v}')
    if summary['g0']['n_images_at_least_1'] != g0_images:
        failures.append('dev282 G0 summary mismatch')
    if summary['g1']['n_images_at_least_1'] != g1_images:
        failures.append('dev282 G1 summary mismatch')
    return {
        'all_passed': not failures,
        'failures': failures[:50],
        'external34': {'n': len(ext), 'a_replay_ok': ext_ok,
                       'qualified': len(qualified), 'selected': len(selected),
                       'unselected_ids': unselected},
        'dev282': {'n': len(dev), 'n_groups': len({row[1]['group_id'] for row in dev}),
                   'a_replay_ok': dev_ok, 'observed': observed,
                   'expected': expected},
    }


if __name__ == '__main__':
    main()
