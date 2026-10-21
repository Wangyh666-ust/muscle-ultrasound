"""PA/FL geometry with frozen apo conservative wiring and full FL diagnostics.

Step 1 (wiring): prediction-mode apo boundaries come ONLY from
`src.umud_baseline.apo_baseline.measure_conservative`, the frozen conservative
entry. The returned status is checked; source, rule version and rule hash are
recorded. `extract_apo_bands` is not imported or called here.

Step 2 (diagnostics): whenever a fascicle line intersects both apo fits, raw
intersections, boundary-to-boundary length, visible length, extrapolated
length and extrapolation ratio are retained even if the candidate is rejected.
All failing conditions are recorded (outside image, outside apo support, local
support missing, excessive extrapolation, apo status rejected), not just the
first one.

Step 4 (candidate validity) is diagnostic only and is not used by PA/FL
selection; region/local-support results are reported separately.

The 12 joint images are all retained in the denominator. Human masks are
reference/diagnosis only and never enter prediction-mode selection.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.umud_baseline.data import (  # noqa: E402
    read_mask, align_mask_to_image, read_image_gray3, ROOT as DATA_ROOT,
)
from src.umud_baseline.apo_baseline import (  # noqa: E402
    load_frozen_rule, measure_conservative, sha256_file,
)
from scripts.fascicle_geometry import (  # noqa: E402
    extract_segments, match_segments, summarize_match,
)
from scripts.candidate_validity import (  # noqa: E402
    check_candidate_region, nearest_paired_distance,
)

FASC_PRED = ROOT / 'runs/stage2_fasc/ls1024/dev_pred/masks'
APO_PRED = ROOT / 'runs/baseline_b/apo/dev_pred_v2/masks'
OUT = ROOT / 'runs/geometry_eval/fasc_pa_fl_v7'
MAX_EXTRAP = 0.80
SUPPORT_TOL_FRAC = 0.05
PA_SELECT_QUANTILES = (0.0, 0.5, 1.0)
FL_SELECT_N = 3
MATCH_PRIMARY = {'perp_tol': 10.0, 'angle_tol': 10.0, 'overlap_min': 0.2}
FL_CONDITION_ORDER = (
    'apo_status_rejected', 'no_apo_boundaries', 'parallel_to_aponeurosis',
    'zero_boundary_span', 'intersection_outside_image',
    'intersection_outside_apo_support', 'local_support_missing',
    'excessive_extrapolation',
)


def _ordered_conditions(conditions):
    unique = []
    for c in conditions:
        if c and c not in unique:
            unique.append(c)
    return sorted(unique, key=lambda c: (FL_CONDITION_ORDER.index(c)
                                         if c in FL_CONDITION_ORDER else 99, c))


def _condition_text(conditions):
    return ';'.join(_ordered_conditions(conditions))


def _support_tol(width):
    return max(10.0, SUPPORT_TOL_FRAC * width)


def _line_intersection_t(centroid, direction, poly):
    cx, cy = centroid
    dx, dy = direction
    a = float(poly[0]); b = float(poly[1])
    den = dy - a * dx
    if abs(den) < 1e-9:
        return None
    return (a * cx + b - cy) / den


def _point_at(centroid, direction, t):
    return (centroid[0] + t * direction[0], centroid[1] + t * direction[1])


def _local_support(fits, x):
    if fits is None:
        return False, None
    d = nearest_paired_distance(fits, x)
    if d is None or not np.isfinite(d):
        return False, None
    max_gap = float(fits.get('max_sample_gap', 0.0))
    return bool(d <= max_gap), float(d)


def pa_for_segment(seg, apo_fits, width, apo_ok=True):
    """PA relative to the deep aponeurosis; PA selection is unchanged."""
    conditions = []
    if not apo_ok:
        conditions.append('apo_status_rejected')
    if apo_fits is None:
        conditions.append('no_apo_boundaries')
        return {'pa_status': 'rejected', 'pa_reason': _condition_text(conditions),
                'pa_failure_conditions': _ordered_conditions(conditions),
                'pa_deg': None, 'pa_deg_raw': None}
    cx = float(seg['centroid'][0])
    lo, hi = apo_fits['support']
    tol = _support_tol(width)
    if not (lo - tol <= cx <= hi + tol):
        conditions.append('fascicle_outside_apo_support')
    deep_angle = math.degrees(math.atan2(float(apo_fits['deep'][0]), 1.0))
    diff = abs(seg['direction_deg'] - deep_angle) % 180.0
    pa = float(min(diff, 180.0 - diff))
    ok = not conditions
    return {
        'pa_status': 'ok' if ok else 'rejected',
        'pa_reason': _condition_text(conditions),
        'pa_failure_conditions': _ordered_conditions(conditions),
        'pa_deg': pa if ok else None,
        'pa_deg_raw': pa,
        'deep_direction_deg': float(deep_angle),
        'segment_direction_deg': float(seg['direction_deg']),
    }


def _fl_acceptance_fields(conditions, total):
    """Separate diagnostic failures from the two acceptance criteria.

    Pure wiring criterion: all original FL reliability conditions, but NOT the
    additional local-paired-support restriction. Strict criterion: pure plus
    local_support_missing. Both are reported; selection uses pure as primary.
    """
    diagnostic = _ordered_conditions(conditions)
    pure_conditions = [c for c in diagnostic if c != 'local_support_missing']
    strict_conditions = diagnostic
    pure_ok = not pure_conditions
    strict_ok = not strict_conditions
    return {
        'fl_status': 'ok' if pure_ok else 'rejected',
        'fl_reason': _condition_text(pure_conditions),
        'fl_failure_conditions': diagnostic,
        'fl_status_pure': 'ok' if pure_ok else 'rejected',
        'fl_reason_pure': _condition_text(pure_conditions),
        'fl_failure_conditions_pure': pure_conditions,
        'fl_accept_pure': pure_ok,
        'fl_status_strict': 'ok' if strict_ok else 'rejected',
        'fl_reason_strict': _condition_text(strict_conditions),
        'fl_failure_conditions_strict': strict_conditions,
        'fl_accept_strict': strict_ok,
        'fl_px': total if pure_ok else None,
        'fl_px_pure': total if pure_ok else None,
        'fl_px_strict': total if strict_ok else None,
    }


def fl_for_segment(seg, apo_fits, width, height, apo_ok=True,
                   max_extrap=MAX_EXTRAP):
    """FL with raw geometry, all diagnostic conditions, and pure/strict acceptance."""
    conditions = []
    if not apo_ok:
        conditions.append('apo_status_rejected')
    if apo_fits is None:
        conditions.append('no_apo_boundaries')
        return {
            **_fl_acceptance_fields(conditions, None),
            'fl_px_raw': None,
            'visible_length_px': None, 'extrapolated_length_px': None,
            'extrapolation_ratio': None,
            'intersection_sup': None, 'intersection_deep': None,
        }
    cx, cy = seg['centroid']
    ang = math.radians(seg['direction_deg'])
    d = (math.cos(ang), math.sin(ang))
    ts = _line_intersection_t((cx, cy), d, apo_fits['sup'])
    td = _line_intersection_t((cx, cy), d, apo_fits['deep'])
    if ts is None or td is None:
        conditions.append('parallel_to_aponeurosis')
        return {
            **_fl_acceptance_fields(conditions, None),
            'fl_px_raw': None,
            'visible_length_px': None, 'extrapolated_length_px': None,
            'extrapolation_ratio': None,
            'intersection_sup': None, 'intersection_deep': None,
        }
    if abs(td - ts) < 1e-9:
        conditions.append('zero_boundary_span')
        return {
            **_fl_acceptance_fields(conditions, None),
            'fl_px_raw': None,
            'visible_length_px': None, 'extrapolated_length_px': None,
            'extrapolation_ratio': None,
            'intersection_sup': None, 'intersection_deep': None,
        }
    ps = _point_at((cx, cy), d, ts)
    pd = _point_at((cx, cy), d, td)
    t_lo, t_hi = sorted([ts, td])
    t_end = [((px - cx) * d[0] + (py - cy) * d[1])
             for px, py in (seg['p0'], seg['p1'])]
    t_seg_lo, t_seg_hi = min(t_end), max(t_end)
    total = float(t_hi - t_lo)
    visible = float(max(0.0, min(t_hi, t_seg_hi) - max(t_lo, t_seg_lo)))
    extrap = float(max(0.0, total - visible))
    ratio = float(extrap / total) if total > 0 else None

    outside_image = any(not (0 <= pt[0] <= width - 1 and 0 <= pt[1] <= height - 1)
                        for pt in (ps, pd))
    lo, hi = apo_fits['support']
    tol = _support_tol(width)
    outside_support = any(not (lo - tol <= pt[0] <= hi + tol)
                          for pt in (ps, pd))
    sup_ok, sup_dist = _local_support(apo_fits, ps[0])
    deep_ok, deep_dist = _local_support(apo_fits, pd[0])
    centroid_ok, centroid_dist = _local_support(apo_fits, cx)
    local_missing = not (sup_ok and deep_ok and centroid_ok)
    excessive = ratio is not None and ratio > max_extrap

    if outside_image:
        conditions.append('intersection_outside_image')
    if outside_support:
        conditions.append('intersection_outside_apo_support')
    if local_missing:
        conditions.append('local_support_missing')
    if excessive:
        conditions.append('excessive_extrapolation')
    return {
        **_fl_acceptance_fields(conditions, total),
        'fl_px_raw': total,
        'visible_length_px': visible,
        'extrapolated_length_px': extrap,
        'extrapolation_ratio': ratio,
        'intersection_sup': list(map(float, ps)),
        'intersection_deep': list(map(float, pd)),
        't_sup': float(ts), 't_deep': float(td),
        'fl_outside_image': bool(outside_image),
        'fl_outside_apo_support': bool(outside_support),
        'fl_local_support_missing': bool(local_missing),
        'fl_excessive_extrapolation': bool(excessive),
        'fl_local_support_distance_sup_px': sup_dist,
        'fl_local_support_distance_deep_px': deep_dist,
        'fl_local_support_distance_centroid_px': centroid_dist,
        'fl_local_support_max_gap_px': float(apo_fits.get('max_sample_gap', 0.0)),
    }


def evaluate_segment(seg, apo_fits, width, height, apo_ok=True):
    rec = {'segment': seg}
    rec.update(pa_for_segment(seg, apo_fits, width, apo_ok=apo_ok))
    rec.update(fl_for_segment(seg, apo_fits, width, height, apo_ok=apo_ok))
    rec.update(check_candidate_region(seg, apo_fits))
    return rec


def select_candidates(candidates, n=FL_SELECT_N):
    pa_ok = [c for c in candidates if c['pa_status'] == 'ok']
    fl_pure = [c for c in candidates if c['fl_accept_pure']]
    fl_strict = [c for c in candidates if c['fl_accept_strict']]
    if pa_ok:
        by_x = sorted(pa_ok, key=lambda c: c['intersection_deep'][0]
                      if c.get('intersection_deep') else c['segment']['centroid'][0])
        idx = sorted(set(int(round(q * (len(by_x) - 1)))
                         for q in PA_SELECT_QUANTILES))
        pa_sel = [by_x[i] for i in idx]
    else:
        pa_sel = []

    def pick_fl(ok):
        if not ok:
            return []
        return sorted(ok, key=lambda c: (c['extrapolation_ratio'],
                                         -c['visible_length_px']))[:n]

    fl_sel = pick_fl(fl_pure)
    fl_sel_strict = pick_fl(fl_strict)
    return {
        'pa': pa_sel, 'fl': fl_sel, 'fl_strict': fl_sel_strict,
        'pa_mean_deg': float(np.mean([c['pa_deg'] for c in pa_sel])) if pa_sel else None,
        'fl_mean_px': float(np.mean([c['fl_px_pure'] for c in fl_sel])) if fl_sel else None,
        'fl_mean_px_strict': (float(np.mean([c['fl_px_strict'] for c in fl_sel_strict]))
                              if fl_sel_strict else None),
        'n_pa_selected': len(pa_sel),
        'n_fl_selected': len(fl_sel),
        'n_fl_selected_strict': len(fl_sel_strict),
        'pa_reason': None if pa_sel else 'no_valid_pa_candidate',
        'fl_reason': None if fl_sel else 'no_valid_fl_candidate',
        'fl_reason_strict': None if fl_sel_strict else 'no_valid_fl_candidate_strict',
    }


def _reason_counts(candidates, kind):
    return dict(sorted(Counter(
        c[f'{kind}_reason'] for c in candidates
        if c[f'{kind}_status'] != 'ok').items()))


def _condition_counts(candidates, kind, field=None, status_field=None):
    counts = Counter()
    field = field or f'{kind}_failure_conditions'
    status_field = status_field or f'{kind}_status'
    for c in candidates:
        if c[status_field] == 'ok':
            continue
        conds = c.get(field) or [c.get(f'{kind}_reason', '')]
        for cond in conds:
            if cond:
                counts[cond] += 1
    return dict(sorted(counts.items()))


def _validity_counts(candidates):
    return {
        'n_candidates': len(candidates),
        'region_status_counts': dict(sorted(Counter(
            c['region_status'] for c in candidates).items())),
        'local_support_status_counts': dict(sorted(Counter(
            c['local_support_status'] for c in candidates).items())),
        'invalid_candidate_count': sum(
            c['candidate_validity_status'] == 'invalid' for c in candidates),
        'condition_counts': dict(sorted(Counter(
            cond for c in candidates
            for cond in c['candidate_validity_conditions']).items())),
    }


def evaluate_mask_pair(fasc_mask, apo_mask, width, height, rule, rule_hash,
                       mask_role='prediction', apo_out=None):
    """Evaluate fascicle candidates against the frozen conservative apo entry."""
    if apo_out is None:
        apo_out = measure_conservative(apo_mask, rule)
    apo_status = apo_out['status']
    apo_reason = apo_out.get('reason', '')
    apo_ok = apo_status == 'candidate'
    diag_fits = apo_out.get('fits')
    if not isinstance(diag_fits, dict):
        diag_fits = None
    fasc = extract_segments(fasc_mask)
    candidates = [
        evaluate_segment(seg, diag_fits, width, height, apo_ok=apo_ok)
        for seg in fasc['segments']
    ]
    return {
        'apo_entry': 'frozen_conservative',
        'apo_mask_role': mask_role,
        'apo_status': apo_status,
        'apo_reason': apo_reason,
        'apo_source': apo_out.get('source', ''),
        'apo_baseline_status': apo_out.get('baseline_status', ''),
        'apo_baseline_reason': apo_out.get('baseline_reason', ''),
        'apo_new_status': apo_out.get('new_status', ''),
        'apo_new_reason': apo_out.get('new_reason', ''),
        'apo_rule_version': rule.get('version'),
        'apo_rule_sha256': rule_hash,
        'apo_support': list(diag_fits['support']) if diag_fits is not None else None,
        'apo_diagnostic_fits_available': diag_fits is not None,
        'apo_fits': diag_fits,
        'n_fascicle_evaluable': len(fasc['segments']),
        'n_fascicle_short': sum(r['category'] == 'short_segment'
                                for r in fasc['rejected']),
        'n_fascicle_crossing': sum(r['category'] == 'crossing_or_branched'
                                   for r in fasc['rejected']),
        'candidates': candidates,
        'pa_rejection_counts': _reason_counts(candidates, 'pa'),
        'pa_condition_counts': _condition_counts(candidates, 'pa'),
        'fl_rejection_counts': _reason_counts(candidates, 'fl'),
        'fl_condition_counts': _condition_counts(
            candidates, 'fl', 'fl_failure_conditions', 'fl_status_strict'),
        'fl_pure_condition_counts': _condition_counts(
            candidates, 'fl', 'fl_failure_conditions_pure', 'fl_status_pure'),
        'fl_strict_condition_counts': _condition_counts(
            candidates, 'fl', 'fl_failure_conditions_strict', 'fl_status_strict'),
        'n_fl_accept_pure': sum(c['fl_accept_pure'] for c in candidates),
        'n_fl_accept_strict': sum(c['fl_accept_strict'] for c in candidates),
        'validity': _validity_counts(candidates),
        'selection': select_candidates(candidates),
    }


def _synthetic_apo_and_fits():
    h = w = 512
    apo = np.zeros((h, w), np.uint8)
    apo[98:103, 50:450] = 1
    apo[298:303, 50:450] = 1
    rule, _ = load_frozen_rule()
    out = measure_conservative(apo, rule)
    fits = out.get('fits') if out.get('status') == 'candidate' else None
    return apo, fits, out


def synthetic_case():
    """Known 30-degree fascicle between horizontal boundaries, 200 px apart."""
    apo, fits, apo_out = _synthetic_apo_and_fits()
    if fits is None:
        return {'status': 'apo_reject', 'reason': apo_out.get('reason')}
    h = w = 512
    fasc = np.zeros((h, w), np.uint8)
    slope = math.tan(math.radians(30))
    for x in range(150, 351):
        y = int(round(200 + slope * (x - 250)))
        fasc[max(0, y - 1):y + 2, x] = 1
    segs = extract_segments(fasc)['segments']
    if not segs:
        return {'status': 'fascicle_extract_failed'}
    res = evaluate_segment(segs[0], fits, w, h, apo_ok=True)
    return {
        'status': 'ok' if (res['pa_status'] == 'ok' and res['fl_status'] == 'ok') else 'rejected',
        'pa_status': res['pa_status'], 'pa_reason': res['pa_reason'],
        'fl_status': res['fl_status'], 'fl_reason': res['fl_reason'],
        'pa_deg': res.get('pa_deg'), 'fl_px': res.get('fl_px'),
        'fl_px_raw': res.get('fl_px_raw'),
        'extrapolation_ratio': res.get('extrapolation_ratio'),
        'expected_pa_deg': 30.0,
        'expected_fl_px': 400.0,
        'expected_extrapolation_ratio': 1 - (200 / math.cos(math.radians(30))) / 400,
    }


def parallel_case():
    """Fascicle parallel to horizontal boundaries must be rejected for FL."""
    apo, fits, apo_out = _synthetic_apo_and_fits()
    if fits is None:
        return {'status': 'apo_reject', 'reason': apo_out.get('reason')}
    h = w = 512
    fasc = np.zeros((h, w), np.uint8)
    fasc[200:204, 100:400] = 1
    segs = extract_segments(fasc)['segments']
    if not segs:
        return {'status': 'fascicle_extract_failed'}
    res = evaluate_segment(segs[0], fits, w, h, apo_ok=True)
    return {'pa_status': res['pa_status'], 'pa_deg': res.get('pa_deg'),
            'fl_status': res['fl_status'], 'fl_reason': res['fl_reason'],
            'fl_failure_conditions': res.get('fl_failure_conditions')}


def find_joint_images():
    rows = list(csv.DictReader(open(ROOT / 'manifest/manifest.csv', encoding='utf-8')))
    idx = {(r['task'], r['role'], r['filename']): r for r in rows}
    by = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / 'manifest/splits_v2.csv', encoding='utf-8')):
        if r['split'] != 'dev':
            continue
        img = idx[(r['task'], 'image', r['filename'])]
        by[img['pixel_sha256']].setdefault(r['task'], []).append(r)
    out = []
    for pix, v in by.items():
        if 'apo' in v and 'fasc' in v:
            out.append({'pixel_sha256': pix, 'apo': v['apo'][0], 'fasc': v['fasc'][0]})
    return out


def draw_joint_overlay(image, apo_fits, pred_eval, human_eval, meta, out_path):
    h, w = image.shape[:2]
    pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil)
    if apo_fits is not None:
        xs = np.arange(0, w)
        for key, col in (('sup', (0, 255, 255)), ('deep', (255, 255, 0))):
            ys = np.polyval(apo_fits[key], xs)
            for x, y in zip(xs, ys):
                if 0 <= y < h:
                    pil.putpixel((int(x), int(round(y))), col)
        lo, hi = apo_fits['support']
        for x in (lo, hi):
            for y in range(h):
                pil.putpixel((int(x), y), (120, 120, 120))

    def draw_eval(eval_res, base_color, selected_pa, selected_fl):
        pa_ids = {id(c) for c in eval_res['selection']['pa']}
        fl_ids = {id(c) for c in eval_res['selection']['fl']}
        for c in eval_res['candidates']:
            seg = c['segment']
            if id(c) in pa_ids or id(c) in fl_ids:
                col = selected_pa if id(c) in pa_ids else selected_fl
            elif c['pa_status'] == 'ok' or c['fl_status'] == 'ok':
                col = base_color
            else:
                col = (150, 150, 150)
            draw.line([tuple(seg['p0']), tuple(seg['p1'])], fill=col, width=2)
            if c.get('intersection_sup') and c.get('intersection_deep'):
                draw.line([tuple(c['intersection_sup']),
                           tuple(c['intersection_deep'])], fill=(255, 165, 0), width=1)
                for pt in (c['intersection_sup'], c['intersection_deep']):
                    x, y = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= x < w and 0 <= y < h:
                        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=col)

    draw_eval(pred_eval, (255, 0, 0), (255, 0, 255), (0, 128, 255))
    draw_eval(human_eval, (0, 180, 0), (0, 220, 0), (0, 80, 160))
    font = ImageFont.truetype('arial.ttf', 12)

    def fmt(v, nd=2):
        return 'NA' if v is None else f'{v:.{nd}f}'

    raw_lines = [
        f"{meta['apo_filename']} / {meta['fasc_filename']} | {w}x{h}",
        f"pred PA={fmt(meta['pred_pa'])} deg FL={fmt(meta['pred_fl'])} px "
        f"apo={meta['apo_pred_status']}:{meta['apo_pred_reason'] or '-'} "
        f"source={meta['apo_pred_source'] or '-'}",
        f"human-ref PA={fmt(meta['human_pa'])} deg FL={fmt(meta['human_fl'])} px "
        f"apo={meta['apo_human_status']}:{meta['apo_human_reason'] or '-'}",
        f"pred FL conditions={meta['pred_fl_conditions']}",
        f"rule={meta['rule_version']} sha={meta['rule_sha256'][:12]}...",
        "legend: cyan/yellow=pred superficial/deep apo, gray=apo support, "
        "magenta/blue=pred PA/FL selected, orange=boundary span, green=human reference",
    ]
    scratch = ImageDraw.Draw(pil)
    lines = []
    for raw in raw_lines:
        cur = ''
        for word in raw.split(' '):
            trial = (cur + ' ' + word).strip()
            if scratch.textlength(trial, font=font) > w - 12 and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            lines.append(cur)
    header = 10 + len(lines) * 16
    canvas = Image.new('RGB', (w, h + header), 'white')
    canvas.paste(pil, (0, header))
    d = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        d.text((6, 4 + i * 16), line, fill='black', font=font)
    canvas.save(out_path)


def _median(vals):
    vals = [float(v) for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def _sum_count_dicts(rows, key):
    out = Counter()
    for r in rows:
        for k, v in json.loads(r[key]).items():
            out[k] += int(v)
    return dict(sorted(out.items()))


def _aggregate_validity(items):
    out = {
        'n_candidates': 0,
        'invalid_candidate_count': 0,
        'region_status_counts': Counter(),
        'local_support_status_counts': Counter(),
        'condition_counts': Counter(),
    }
    for item in items:
        out['n_candidates'] += int(item.get('n_candidates', 0))
        out['invalid_candidate_count'] += int(item.get('invalid_candidate_count', 0))
        out['region_status_counts'].update(item.get('region_status_counts', {}))
        out['local_support_status_counts'].update(
            item.get('local_support_status_counts', {}))
        out['condition_counts'].update(item.get('condition_counts', {}))
    return {
        'n_candidates': out['n_candidates'],
        'invalid_candidate_count': out['invalid_candidate_count'],
        'region_status_counts': dict(sorted(out['region_status_counts'].items())),
        'local_support_status_counts': dict(sorted(
            out['local_support_status_counts'].items())),
        'condition_counts': dict(sorted(out['condition_counts'].items())),
    }


def _combo_row(image_meta, combo_key, fasc_source, apo_source, ev):
    s = ev['selection']
    return {
        'apo_filename': image_meta['apo_filename'],
        'fasc_filename': image_meta['fasc_filename'],
        'combination': combo_key,
        'fasc_source': fasc_source,
        'apo_source': apo_source,
        'apo_mask_role': ev['apo_mask_role'],
        'apo_status': ev['apo_status'],
        'apo_reason': ev['apo_reason'],
        'apo_selection_source': ev['apo_source'],
        'apo_rule_version': ev['apo_rule_version'],
        'apo_rule_sha256': ev['apo_rule_sha256'],
        'apo_diagnostic_fits_available': ev['apo_diagnostic_fits_available'],
        'n_candidates': len(ev['candidates']),
        'n_pa_selected': s['n_pa_selected'],
        'n_fl_selected': s['n_fl_selected'],
        'n_fl_selected_pure': s['n_fl_selected'],
        'n_fl_selected_strict': s['n_fl_selected_strict'],
        'pa_mean_deg': s['pa_mean_deg'],
        'fl_mean_px': s['fl_mean_px'],
        'fl_mean_px_strict': s['fl_mean_px_strict'],
        'pa_condition_counts': json.dumps(ev['pa_condition_counts']),
        'fl_condition_counts': json.dumps(ev['fl_condition_counts']),
        'fl_pure_condition_counts': json.dumps(ev['fl_pure_condition_counts']),
        'fl_strict_condition_counts': json.dumps(ev['fl_strict_condition_counts']),
        'validity_invalid_count': ev['validity']['invalid_candidate_count'],
        'validity_condition_counts': json.dumps(ev['validity']['condition_counts']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)

    rule, rule_hash = load_frozen_rule()
    synth = {'known_30deg': synthetic_case(), 'parallel': parallel_case()}
    (out / 'synthetic_validation.json').write_text(
        json.dumps(synth, indent=2, ensure_ascii=False), encoding='utf-8')

    joint = find_joint_images()
    if len(joint) != 12:
        raise ValueError(f'Expected 12 joint apo/fasc dev images, got {len(joint)}')
    manifest = {}
    for r in csv.DictReader(open(ROOT / 'manifest/manifest.csv', encoding='utf-8')):
        manifest[(r['task'], r['role'], r['filename'])] = r
    split_rows = list(csv.DictReader(open(ROOT / 'manifest/splits_v2.csv', encoding='utf-8')))
    identity = []
    for j in joint:
        by_task = defaultdict(list)
        for r in split_rows:
            if r['split'] != 'dev':
                continue
            img = manifest[(r['task'], 'image', r['filename'])]
            if img['pixel_sha256'] == j['pixel_sha256']:
                by_task[r['task']].append(r['filename'])
        identity.append({
            'pixel_sha256': j['pixel_sha256'],
            'apo_filenames': sorted(set(by_task.get('apo', []))),
            'fasc_filenames': sorted(set(by_task.get('fasc', []))),
            'evaluated_pair': [j['apo']['filename'], j['fasc']['filename']],
            'width': int(manifest[('apo', 'image', j['apo']['filename'])]['width']),
            'height': int(manifest[('apo', 'image', j['apo']['filename'])]['height']),
        })
    (out / 'joint_identity.json').write_text(
        json.dumps(identity, indent=2, ensure_ascii=False), encoding='utf-8')

    joint_rows = []
    joint_candidates = []
    combo_rows = []
    direction_rows = []
    for j in joint:
        a, f = j['apo'], j['fasc']
        ai = manifest[('apo', 'image', a['filename'])]
        fi = manifest[('fasc', 'image', f['filename'])]
        if (ai['width'], ai['height']) != (fi['width'], fi['height']):
            raise ValueError('joint image dimension mismatch')
        image = read_image_gray3(Path(DATA_ROOT) / ai['path'])
        pix = hashlib.sha256(np.asarray(image[:, :, 0]).tobytes()).hexdigest()
        if pix != j['pixel_sha256']:
            raise ValueError('joint pixel hash mismatch')
        apo_pred = (np.asarray(Image.open(APO_PRED / (Path(a['filename']).stem + '.png'))
                               .convert('L')) > 127)
        fasc_pred = (np.asarray(Image.open(FASC_PRED / (Path(f['filename']).stem + '.png'))
                                .convert('L')) > 127)
        apo_h = align_mask_to_image(
            read_mask(Path(DATA_ROOT) / manifest[('apo', 'mask', a['filename'])]['path']),
            image.shape[1::-1])
        fasc_h = align_mask_to_image(
            read_mask(Path(DATA_ROOT) / manifest[('fasc', 'mask', f['filename'])]['path']),
            image.shape[1::-1])
        apo_pred_out = measure_conservative(apo_pred, rule)
        apo_h_out = measure_conservative(apo_h, rule)
        pred_eval = evaluate_mask_pair(fasc_pred, apo_pred, image.shape[1], image.shape[0],
                                       rule, rule_hash, 'prediction', apo_pred_out)
        human_eval = evaluate_mask_pair(fasc_h, apo_h, image.shape[1], image.shape[0],
                                        rule, rule_hash, 'reference', apo_h_out)
        pred_fasc_human_apo = evaluate_mask_pair(
            fasc_pred, apo_h, image.shape[1], image.shape[0], rule, rule_hash,
            'mixed_pred_fasc_human_apo', apo_h_out)
        human_fasc_pred_apo = evaluate_mask_pair(
            fasc_h, apo_pred, image.shape[1], image.shape[0], rule, rule_hash,
            'mixed_human_fasc_pred_apo', apo_pred_out)
        ps, hs = pred_eval['selection'], human_eval['selection']
        image_meta = {'apo_filename': a['filename'], 'fasc_filename': f['filename']}
        row = {
            'pixel_sha256': j['pixel_sha256'],
            'apo_filename': a['filename'], 'fasc_filename': f['filename'],
            'width': image.shape[1], 'height': image.shape[0],
            'apo_pred_entry': pred_eval['apo_entry'],
            'apo_pred_status': pred_eval['apo_status'],
            'apo_pred_reason': pred_eval['apo_reason'],
            'apo_pred_source': pred_eval['apo_source'],
            'apo_pred_baseline_status': pred_eval['apo_baseline_status'],
            'apo_pred_baseline_reason': pred_eval['apo_baseline_reason'],
            'apo_pred_new_status': pred_eval['apo_new_status'],
            'apo_pred_new_reason': pred_eval['apo_new_reason'],
            'apo_pred_rule_version': pred_eval['apo_rule_version'],
            'apo_pred_rule_sha256': pred_eval['apo_rule_sha256'],
            'apo_pred_support': pred_eval['apo_support'],
            'apo_pred_diagnostic_fits_available': pred_eval['apo_diagnostic_fits_available'],
            'apo_human_status': human_eval['apo_status'],
            'apo_human_reason': human_eval['apo_reason'],
            'apo_human_source': human_eval['apo_source'],
            'apo_human_support': human_eval['apo_support'],
            'n_fascicle_evaluable_pred': pred_eval['n_fascicle_evaluable'],
            'n_fascicle_evaluable_human': human_eval['n_fascicle_evaluable'],
            'n_candidates_pred': len(pred_eval['candidates']),
            'n_candidates_human': len(human_eval['candidates']),
            'pred_n_pa_ok': ps['n_pa_selected'],
            'pred_n_fl_ok': ps['n_fl_selected'],
            'pred_n_fl_ok_pure': ps['n_fl_selected'],
            'pred_n_fl_ok_strict': ps['n_fl_selected_strict'],
            'human_n_pa_ok': hs['n_pa_selected'],
            'human_n_fl_ok': hs['n_fl_selected'],
            'human_n_fl_ok_pure': hs['n_fl_selected'],
            'human_n_fl_ok_strict': hs['n_fl_selected_strict'],
            'pred_pa_reasons': json.dumps(pred_eval['pa_rejection_counts']),
            'pred_fl_reasons': json.dumps(pred_eval['fl_rejection_counts']),
            'pred_pa_conditions': json.dumps(pred_eval['pa_condition_counts']),
            'pred_fl_conditions': json.dumps(pred_eval['fl_condition_counts']),
            'pred_fl_pure_conditions': json.dumps(pred_eval['fl_pure_condition_counts']),
            'pred_fl_strict_conditions': json.dumps(pred_eval['fl_strict_condition_counts']),
            'human_pa_reasons': json.dumps(human_eval['pa_rejection_counts']),
            'human_fl_reasons': json.dumps(human_eval['fl_rejection_counts']),
            'human_pa_conditions': json.dumps(human_eval['pa_condition_counts']),
            'human_fl_conditions': json.dumps(human_eval['fl_condition_counts']),
            'human_fl_pure_conditions': json.dumps(human_eval['fl_pure_condition_counts']),
            'human_fl_strict_conditions': json.dumps(human_eval['fl_strict_condition_counts']),
            'pred_validity': json.dumps(pred_eval['validity']),
            'human_validity': json.dumps(human_eval['validity']),
            'pred_pa_mean_deg': ps['pa_mean_deg'], 'pred_fl_mean_px': ps['fl_mean_px'],
            'pred_fl_mean_px_strict': ps['fl_mean_px_strict'],
            'human_pa_mean_deg': hs['pa_mean_deg'], 'human_fl_mean_px': hs['fl_mean_px'],
            'human_fl_mean_px_strict': hs['fl_mean_px_strict'],
            'pa_diff_deg': (ps['pa_mean_deg'] - hs['pa_mean_deg'])
                           if ps['pa_mean_deg'] is not None and hs['pa_mean_deg'] is not None else None,
            'fl_diff_px': (ps['fl_mean_px'] - hs['fl_mean_px'])
                          if ps['fl_mean_px'] is not None and hs['fl_mean_px'] is not None else None,
            'fl_diff_px_strict': (ps['fl_mean_px_strict'] - hs['fl_mean_px_strict'])
                                 if ps['fl_mean_px_strict'] is not None and hs['fl_mean_px_strict'] is not None else None,
            'pred_pa_selection_reason': ps['pa_reason'],
            'pred_fl_selection_reason': ps['fl_reason'],
            'pred_fl_selection_reason_strict': ps['fl_reason_strict'],
            'human_pa_selection_reason': hs['pa_reason'],
            'human_fl_selection_reason': hs['fl_reason'],
            'human_fl_selection_reason_strict': hs['fl_reason_strict'],
        }
        joint_rows.append(row)
        pred_pa_ids = {id(c) for c in ps['pa']}
        pred_fl_ids = {id(c) for c in ps['fl']}
        human_pa_ids = {id(c) for c in hs['pa']}
        human_fl_ids = {id(c) for c in hs['fl']}

        def apo_meta(ev):
            return {
                'entry': ev['apo_entry'], 'status': ev['apo_status'],
                'reason': ev['apo_reason'], 'source': ev['apo_source'],
                'baseline_status': ev['apo_baseline_status'],
                'baseline_reason': ev['apo_baseline_reason'],
                'new_status': ev['apo_new_status'],
                'new_reason': ev['apo_new_reason'],
                'rule_version': ev['apo_rule_version'],
                'rule_sha256': ev['apo_rule_sha256'],
                'support': ev['apo_support'],
                'diagnostic_fits_available': ev['apo_diagnostic_fits_available'],
            }

        joint_candidates.append({
            'apo_filename': a['filename'], 'fasc_filename': f['filename'],
            'pixel_sha256': j['pixel_sha256'],
            'apo_pred': apo_meta(pred_eval),
            'apo_human': apo_meta(human_eval),
            'pred_candidates': [
                {**c, 'selected_pa': id(c) in pred_pa_ids,
                 'selected_fl': id(c) in pred_fl_ids}
                for c in pred_eval['candidates']],
            'human_candidates': [
                {**c, 'selected_pa': id(c) in human_pa_ids,
                 'selected_fl': id(c) in human_fl_ids}
                for c in human_eval['candidates']],
            'pred_selection': {
                'pa_mean_deg': ps['pa_mean_deg'], 'fl_mean_px': ps['fl_mean_px'],
                'fl_mean_px_strict': ps['fl_mean_px_strict'],
                'n_pa_selected': ps['n_pa_selected'], 'n_fl_selected': ps['n_fl_selected'],
                'n_fl_selected_strict': ps['n_fl_selected_strict'],
                'pa_reason': ps['pa_reason'], 'fl_reason': ps['fl_reason'],
                'fl_reason_strict': ps['fl_reason_strict'],
            },
            'human_selection': {
                'pa_mean_deg': hs['pa_mean_deg'], 'fl_mean_px': hs['fl_mean_px'],
                'fl_mean_px_strict': hs['fl_mean_px_strict'],
                'n_pa_selected': hs['n_pa_selected'], 'n_fl_selected': hs['n_fl_selected'],
                'n_fl_selected_strict': hs['n_fl_selected_strict'],
                'pa_reason': hs['pa_reason'], 'fl_reason': hs['fl_reason'],
                'fl_reason_strict': hs['fl_reason_strict'],
            },
        })
        for combo_key, fasc_source, apo_source, ev in (
            ('pred_fasc_pred_apo', 'pred', 'pred', pred_eval),
            ('human_fasc_human_apo', 'human', 'human', human_eval),
            ('pred_fasc_human_apo', 'pred', 'human', pred_fasc_human_apo),
            ('human_fasc_pred_apo', 'human', 'pred', human_fasc_pred_apo),
        ):
            combo_rows.append(_combo_row(image_meta, combo_key,
                                         fasc_source, apo_source, ev))

        # Direction and raw-intersection comparison: prediction vs human mask.
        pred_segs = extract_segments(fasc_pred)['segments']
        human_segs = extract_segments(fasc_h)['segments']
        pm = match_segments(pred_segs, human_segs, **MATCH_PRIMARY)
        psum = summarize_match(pred_segs, human_segs, pm)

        def fl_diag(ev):
            cands = ev['candidates']
            finite = [c for c in cands if c.get('fl_px_raw') is not None]
            ratios = [c.get('extrapolation_ratio') for c in finite]
            return {
                'n_finite_intersections': len(finite),
                'n_outside_image': sum(bool(c.get('fl_outside_image')) for c in cands),
                'n_outside_apo_support': sum(bool(c.get('fl_outside_apo_support')) for c in cands),
                'n_local_support_missing': sum(bool(c.get('fl_local_support_missing')) for c in cands),
                'n_excessive_extrapolation': sum(bool(c.get('fl_excessive_extrapolation')) for c in cands),
                'median_raw_fl_px': _median([c.get('fl_px_raw') for c in finite]),
                'median_extrapolation_ratio': _median(ratios),
            }
        pred_fl = fl_diag(pred_eval)
        human_fl = fl_diag(human_eval)
        direction_rows.append({
            'apo_filename': a['filename'], 'fasc_filename': f['filename'],
            'pred_n_evaluable': len(pred_segs), 'human_n_evaluable': len(human_segs),
            'n_matched': psum['n_matched'], 'coverage': psum['coverage'],
            'direction_median_deg': psum['direction_median_deg'],
            'position_perp_median_px': psum['position_perp_median_px'],
            'position_mid_median_px': psum['position_mid_median_px'],
            'pred_median_length_px': _median([s['length_px'] for s in pred_segs]),
            'human_median_length_px': _median([s['length_px'] for s in human_segs]),
            'pred_fragmented_segments': sum(s.get('fragmented_neighbor', False) for s in pred_segs),
            'human_fragmented_segments': sum(s.get('fragmented_neighbor', False) for s in human_segs),
            'pred_direction_median_all_deg': _median([s['direction_deg'] for s in pred_segs]),
            'human_direction_median_all_deg': _median([s['direction_deg'] for s in human_segs]),
            **{f'pred_{k}': v for k, v in pred_fl.items()},
            **{f'human_{k}': v for k, v in human_fl.items()},
        })

        meta = {
            'apo_filename': a['filename'], 'fasc_filename': f['filename'],
            'pred_pa': ps['pa_mean_deg'], 'pred_fl': ps['fl_mean_px'],
            'human_pa': hs['pa_mean_deg'], 'human_fl': hs['fl_mean_px'],
            'apo_pred_status': pred_eval['apo_status'],
            'apo_pred_reason': pred_eval['apo_reason'],
            'apo_pred_source': pred_eval['apo_source'],
            'apo_human_status': human_eval['apo_status'],
            'apo_human_reason': human_eval['apo_reason'],
            'pred_fl_conditions': json.dumps(pred_eval['fl_condition_counts']),
            'rule_version': rule['version'],
            'rule_sha256': rule_hash,
        }
        draw_joint_overlay(image, pred_eval['apo_fits'], pred_eval, human_eval, meta,
                           out / f"{Path(f['filename']).stem}_pa_fl.png")

    (out / 'joint_12_per_image.json').write_text(
        json.dumps(joint_rows, indent=2, ensure_ascii=False), encoding='utf-8')
    with open(out / 'joint_12_per_image.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(joint_rows[0].keys()))
        w.writeheader(); w.writerows(joint_rows)
    (out / 'joint_12_candidates.json').write_text(
        json.dumps(joint_candidates, indent=2, ensure_ascii=False), encoding='utf-8')
    with open(out / 'joint_12_combinations.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(combo_rows[0].keys()))
        w.writeheader(); w.writerows(combo_rows)
    with open(out / 'joint_12_direction.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(direction_rows[0].keys()))
        w.writeheader(); w.writerows(direction_rows)

    candidate_rows = []
    for x in joint_candidates:
        for source in ('pred', 'human'):
            apo = x[f'apo_{source}']
            for ci, c in enumerate(x[f'{source}_candidates']):
                seg = c['segment']
                candidate_rows.append({
                    'apo_filename': x['apo_filename'],
                    'fasc_filename': x['fasc_filename'],
                    'source': source,
                    'candidate_index': ci,
                    'apo_status': apo['status'], 'apo_reason': apo['reason'],
                    'apo_source': apo['source'],
                    'apo_rule_version': apo['rule_version'],
                    'apo_rule_sha256': apo['rule_sha256'],
                    'apo_support': json.dumps(apo['support']),
                    'centroid_x': seg['centroid'][0], 'centroid_y': seg['centroid'][1],
                    'direction_deg': seg['direction_deg'],
                    'length_px': seg['length_px'],
                    'pa_status': c['pa_status'], 'pa_reason': c['pa_reason'],
                    'pa_deg': c.get('pa_deg'), 'pa_deg_raw': c.get('pa_deg_raw'),
                    'pa_failure_conditions': json.dumps(c.get('pa_failure_conditions')),
                    'fl_status': c['fl_status'], 'fl_reason': c['fl_reason'],
                    'fl_status_pure': c.get('fl_status_pure'),
                    'fl_reason_pure': c.get('fl_reason_pure'),
                    'fl_accept_pure': c.get('fl_accept_pure'),
                    'fl_status_strict': c.get('fl_status_strict'),
                    'fl_reason_strict': c.get('fl_reason_strict'),
                    'fl_accept_strict': c.get('fl_accept_strict'),
                    'fl_failure_conditions': json.dumps(c.get('fl_failure_conditions')),
                    'fl_failure_conditions_pure': json.dumps(c.get('fl_failure_conditions_pure')),
                    'fl_failure_conditions_strict': json.dumps(c.get('fl_failure_conditions_strict')),
                    'fl_px': c.get('fl_px'), 'fl_px_pure': c.get('fl_px_pure'),
                    'fl_px_strict': c.get('fl_px_strict'), 'fl_px_raw': c.get('fl_px_raw'),
                    'visible_length_px': c.get('visible_length_px'),
                    'extrapolated_length_px': c.get('extrapolated_length_px'),
                    'extrapolation_ratio': c.get('extrapolation_ratio'),
                    'fl_outside_image': c.get('fl_outside_image'),
                    'fl_outside_apo_support': c.get('fl_outside_apo_support'),
                    'fl_local_support_missing': c.get('fl_local_support_missing'),
                    'fl_excessive_extrapolation': c.get('fl_excessive_extrapolation'),
                    'intersection_sup': json.dumps(c.get('intersection_sup')),
                    'intersection_deep': json.dumps(c.get('intersection_deep')),
                    'region_status': c.get('region_status'),
                    'region_inside_fraction': c.get('region_inside_fraction'),
                    'local_support_status': c.get('local_support_status'),
                    'local_support_fraction': c.get('local_support_fraction'),
                    'candidate_validity_status': c.get('candidate_validity_status'),
                    'candidate_validity_conditions': json.dumps(
                        c.get('candidate_validity_conditions')),
                    'selected_pa': c.get('selected_pa', False),
                    'selected_fl': c.get('selected_fl', False),
                })
    if candidate_rows:
        with open(out / 'joint_12_candidates.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(candidate_rows[0].keys()))
            w.writeheader(); w.writerows(candidate_rows)

    overlays = sorted(out.glob('*_pa_fl.png'))
    if overlays:
        ims = [Image.open(p).convert('RGB') for p in overlays]
        cw = 520
        ims = [im.resize((cw, int(im.height * cw / im.width)), Image.LANCZOS)
               for im in ims]
        ch = max(im.height for im in ims)
        cols = 3
        rows_n = (len(ims) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * cw, rows_n * ch), 'white')
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % cols) * cw, (i // cols) * ch))
        sheet.save(out / 'joint_12_montage.png')

    def med(vals):
        vals = [abs(v) for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    def combo_summary():
        summary = {}
        for key in ('pred_fasc_pred_apo', 'human_fasc_human_apo',
                    'pred_fasc_human_apo', 'human_fasc_pred_apo'):
            rs = [r for r in combo_rows if r['combination'] == key]
            summary[key] = {
                'n_images': len(rs),
                'apo_status_counts': dict(sorted(Counter(r['apo_status'] for r in rs).items())),
                'apo_selection_source_counts': dict(sorted(Counter(
                    r['apo_selection_source'] for r in rs).items())),
                'n_images_with_pa': sum(r['n_pa_selected'] > 0 for r in rs),
                'n_images_with_fl': sum(r['n_fl_selected'] > 0 for r in rs),
                'n_images_with_fl_strict': sum(r['n_fl_selected_strict'] > 0 for r in rs),
                'n_candidates': sum(r['n_candidates'] for r in rs),
                'n_pa_selected': sum(r['n_pa_selected'] for r in rs),
                'n_fl_selected': sum(r['n_fl_selected'] for r in rs),
                'n_fl_selected_strict': sum(r['n_fl_selected_strict'] for r in rs),
                'fl_condition_counts': _sum_count_dicts(rs, 'fl_condition_counts'),
                'fl_pure_condition_counts': _sum_count_dicts(rs, 'fl_pure_condition_counts'),
                'fl_strict_condition_counts': _sum_count_dicts(rs, 'fl_strict_condition_counts'),
                'validity_invalid_count': sum(r['validity_invalid_count'] for r in rs),
            }
        return summary

    def direction_summary():
        rs = direction_rows
        return {
            'n_images': len(rs),
            'n_images_with_matches': sum(r['n_matched'] > 0 for r in rs),
            'n_matched': sum(r['n_matched'] for r in rs),
            'median_direction_deg_conditional': _median(
                [r['direction_median_deg'] for r in rs]),
            'median_perp_px_conditional': _median(
                [r['position_perp_median_px'] for r in rs]),
            'median_midpoint_px_conditional': _median(
                [r['position_mid_median_px'] for r in rs]),
            'pred_median_length_px': _median([r['pred_median_length_px'] for r in rs]),
            'human_median_length_px': _median([r['human_median_length_px'] for r in rs]),
            'pred_fragmented_segments': sum(r['pred_fragmented_segments'] for r in rs),
            'human_fragmented_segments': sum(r['human_fragmented_segments'] for r in rs),
            'n_images_with_finite_pred_intersections': sum(
                r['pred_n_finite_intersections'] > 0 for r in rs),
            'n_images_with_finite_human_intersections': sum(
                r['human_n_finite_intersections'] > 0 for r in rs),
            'pred_finite_intersections': sum(r['pred_n_finite_intersections'] for r in rs),
            'human_finite_intersections': sum(r['human_n_finite_intersections'] for r in rs),
        }

    summary = {
        'wiring': {
            'apo_entry': 'frozen_conservative',
            'entry_module': 'src/umud_baseline/apo_baseline.py::measure_conservative',
            'old_extract_apo_bands_used': False,
            'rule_version': rule['version'],
            'rule_sha256': rule_hash,
            'n_joint_denominator': len(joint_rows),
            'all_12_retained': len(joint_rows) == 12,
        },
        'synthetic_validation': synth,
        'joint_n': len(joint_rows),
        'joint_apo_pred_sources': dict(sorted(Counter(
            r['apo_pred_source'] for r in joint_rows).items())),
        'joint_apo_pred_statuses': dict(sorted(Counter(
            r['apo_pred_status'] for r in joint_rows).items())),
        'joint_pred_pa_available': sum(r['pred_pa_mean_deg'] is not None for r in joint_rows),
        'joint_pred_pa_at_least_3': sum(int(r['pred_n_pa_ok']) >= 3 for r in joint_rows),
        'joint_pred_fl_available': sum(r['pred_fl_mean_px'] is not None for r in joint_rows),
        'joint_pred_fl_available_pure': sum(r['pred_fl_mean_px'] is not None for r in joint_rows),
        'joint_pred_fl_available_strict': sum(
            r['pred_fl_mean_px_strict'] is not None for r in joint_rows),
        'joint_pred_fl_at_least_3': sum(int(r['pred_n_fl_ok']) >= 3 for r in joint_rows),
        'joint_pred_fl_at_least_3_strict': sum(
            int(r['pred_n_fl_ok_strict']) >= 3 for r in joint_rows),
        'joint_human_pa_available': sum(r['human_pa_mean_deg'] is not None for r in joint_rows),
        'joint_human_fl_available': sum(r['human_fl_mean_px'] is not None for r in joint_rows),
        'joint_human_fl_available_strict': sum(
            r['human_fl_mean_px_strict'] is not None for r in joint_rows),
        # Explicit statistical names: absolute paired differences, with
        # sample counts. These are not MAE and not accuracy.
        'pa_abs_diff_median_deg': med([r['pa_diff_deg'] for r in joint_rows]),
        'n_pa_abs_diff_pairs': sum(r['pa_diff_deg'] is not None for r in joint_rows),
        'fl_abs_diff_median_px': med([r['fl_diff_px'] for r in joint_rows]),
        'n_fl_abs_diff_pairs': sum(r['fl_diff_px'] is not None for r in joint_rows),
        'fl_abs_diff_median_px_strict': med([r['fl_diff_px_strict'] for r in joint_rows]),
        'n_fl_abs_diff_pairs_strict': sum(
            r['fl_diff_px_strict'] is not None for r in joint_rows),
        # Legacy names retained for older consumers; report uses the explicit
        # names above.
        'pa_diff_median_deg': med([r['pa_diff_deg'] for r in joint_rows]),
        'fl_diff_median_px': med([r['fl_diff_px'] for r in joint_rows]),
        'fl_diff_median_px_strict': med([r['fl_diff_px_strict'] for r in joint_rows]),
        'pred_pa_condition_counts': _sum_count_dicts(joint_rows, 'pred_pa_conditions'),
        'pred_fl_condition_counts': _sum_count_dicts(joint_rows, 'pred_fl_conditions'),
        'pred_fl_pure_condition_counts': _sum_count_dicts(
            joint_rows, 'pred_fl_pure_conditions'),
        'pred_fl_strict_condition_counts': _sum_count_dicts(
            joint_rows, 'pred_fl_strict_conditions'),
        'human_pa_condition_counts': _sum_count_dicts(joint_rows, 'human_pa_conditions'),
        'human_fl_condition_counts': _sum_count_dicts(joint_rows, 'human_fl_conditions'),
        'human_fl_pure_condition_counts': _sum_count_dicts(
            joint_rows, 'human_fl_pure_conditions'),
        'human_fl_strict_condition_counts': _sum_count_dicts(
            joint_rows, 'human_fl_strict_conditions'),
        'combinations': combo_summary(),
        'direction_comparison': direction_summary(),
        'candidate_validity': {
            'pred': _aggregate_validity(
                [json.loads(r['pred_validity']) for r in joint_rows]),
            'human': _aggregate_validity(
                [json.loads(r['human_validity']) for r in joint_rows]),
            'note': 'diagnostic only; never used by PA/FL selection',
        },
        'meaning': ('Step-1 wiring check plus full FL failure diagnostics and candidate-validity '
                    'diagnostics. PA/FL are degree/pixel diagnostics; no true parameter-accuracy '
                    'claim without independent expert PA/FL labels and a reliable scale.'),
        'no_scale_mm_gt': 'FL is pixels only; no mm unless a reliable scale is available',
        'no_fl_from_mt_sin_pa': True,
        'human_masks_used_in_prediction_selection': False,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding='utf-8')
    verification = {
        'status': 'passed',
        'model_rerun': False,
        'threshold_changed': False,
        'holdout_read': False,
        'joint_identity_checked_by_pixel_hash': True,
        'n_joint': len(joint_rows),
        'apo_entry': 'frozen_conservative',
        'apo_entry_module': 'src/umud_baseline/apo_baseline.py',
        'apo_rule_version': rule['version'],
        'apo_rule_sha256': rule_hash,
        'old_extract_apo_bands_used': False,
        'human_masks_used_in_prediction_selection': False,
        'fasc_prediction_dir': str(FASC_PRED.relative_to(ROOT)),
        'apo_prediction_dir': str(APO_PRED.relative_to(ROOT)),
        'fascicle_geometry_sha256': sha256_file(ROOT / 'scripts/fascicle_geometry.py'),
        'candidate_validity_sha256': sha256_file(ROOT / 'scripts/candidate_validity.py'),
        'apo_baseline_module_sha256': sha256_file(ROOT / 'src/umud_baseline/apo_baseline.py'),
        'splits_sha256': sha256_file(ROOT / 'manifest/splits_v2.csv'),
        'manifest_sha256': sha256_file(ROOT / 'manifest/manifest.csv'),
        'script_sha256': sha256_file(Path(__file__)),
    }
    (out / 'verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False),
                                           encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
