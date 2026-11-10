"""Corrected Round25 reference evaluation (Stage A / Round26).

Reads only frozen predictions after their hashes and the frozen reference
identity have been verified. This script deliberately writes to a new output
directory: the historical Round25 v1 evaluation is never overwritten.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.round25_boundary_fit import (  # noqa: E402
    curve_errors, fit_curve,
)
from src.umud_baseline.apo_baseline import (  # noqa: E402
    correctness_support_valid, error_at_reference, support_valid_mask,
)

BASE = ROOT / 'runs/geometry_eval/round25_boundary_fit_v1'
CORRECTED_BASE = ROOT / 'runs/geometry_eval/round26_stage_a_from_round25_v1'
REF = ROOT / 'runs/geometry_eval/round24_apo_pair_reference_v1/inputs/reference_143.json'
REF_SOURCE = ROOT / 'manifest/apo_pair_selection_eval_subset_v1.json'
REF_SHA256 = '1d2fef9ea2529971b660a5957705f4497e82bc9be7f7597c6e415ab30e8e13bb'
REF_SOURCE_SHA256 = '6914441fbe23744215903cf36753895e8731ede493231471dc7c77a76149d545'
FOCUS = ('image_0871.tif', 'image_0873.tif', 'image_0875.tif',
         'image_0881.tif', 'image_0982.tif', 'image_0983.tif',
         'image_0987.tif')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def j(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=True),
                          encoding='utf-8')


def unique_rows(rows, label, expected_n=None):
    """Reject duplicate, missing, extra or malformed row identities."""
    if not isinstance(rows, list):
        raise ValueError(f'{label} is not a JSON list')
    out = {}
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get('image_id'), str):
            raise ValueError(f'{label} has a row without image_id')
        iid = r['image_id']
        if iid in out:
            raise ValueError(f'{label} duplicate image_id: {iid}')
        out[iid] = r
    if expected_n is not None and len(out) != expected_n:
        raise ValueError(f'{label} identity count {len(out)} != {expected_n}')
    return out


def validate_reference(path, details):
    """Validate the actual reference file, not merely a success flag."""
    if sha(path) != REF_SHA256:
        raise ValueError('frozen reference hash mismatch')
    if sha(REF_SOURCE) != REF_SOURCE_SHA256:
        raise ValueError('frozen reference source manifest hash mismatch')
    raw = j(path)
    refs = unique_rows(raw, 'reference', 143)
    if len(details) != 153:
        raise ValueError('fit_details is not 153 identities')
    if not set(refs).issubset(details):
        raise ValueError('reference has an identity absent from predictions')
    unclear = set(details) - set(refs)
    if len(unclear) != 10:
        raise ValueError(f'expected 10 target-unclear identities, got {len(unclear)}')
    for iid, r in refs.items():
        for key in ('filename', 'width', 'height', 'reference_x',
                    'reference_sup', 'reference_deep', 'human_mask_sha256'):
            if key not in r:
                raise ValueError(f'reference missing field {key}: {iid}')
        if r['filename'] != iid:
            raise ValueError(f'reference filename/image_id mismatch: {iid}')
        d = details[iid]
        if (int(r['width']), int(r['height'])) != (int(d['width']),
                                                   int(d['height'])):
            raise ValueError(f'reference dimension mismatch: {iid}')
        x = np.asarray(r['reference_x'], dtype=float)
        sup = np.asarray(r['reference_sup'], dtype=float)
        deep = np.asarray(r['reference_deep'], dtype=float)
        if len(x) == 0 or not (len(x) == len(sup) == len(deep)):
            raise ValueError(f'reference array length mismatch: {iid}')
        if not (np.isfinite(x).all() and np.isfinite(sup).all() and
                np.isfinite(deep).all()):
            raise ValueError(f'non-finite reference array: {iid}')
        if np.any(np.diff(x) <= 0):
            raise ValueError(f'reference_x not strictly increasing: {iid}')
        if x.min() < 0 or x.max() >= int(r['width']):
            raise ValueError(f'reference_x outside image width: {iid}')
        if not isinstance(r['human_mask_sha256'], str) or \
                len(r['human_mask_sha256']) != 64:
            raise ValueError(f'invalid reference mask identity: {iid}')
    if len(refs) + len(unclear) != 153:
        raise ValueError('reference/unclear partition != 153')
    return refs, unclear


def verify(base):
    p = base / 'predictions/manifest.json'
    s = base / 'predictions/manifest.sha256'
    if not p.exists() or not s.exists() or s.read_text().strip() != sha(p):
        raise ValueError('prediction manifest hash mismatch')
    m = j(p)
    if m.get('reference_values_read') is not False or \
            m.get('frozen_before_reference_evaluation') is not True:
        raise ValueError('predictions not reference-blind/frozen')
    required = {'predictions/F0_153.json', 'predictions/F1_153.json',
                'predictions/fit_details.json', 'inputs/common_points.json'}
    files = m.get('files')
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError('prediction manifest lacks required frozen files')
    for rel, want in files.items():
        if sha(base / rel) != want:
            raise ValueError(f'prediction changed: {rel}')
    if not (base / 'prediction_success.json').exists():
        raise ValueError('prediction success marker missing')
    cfg = ROOT / 'manifest/round25_boundary_fit_v1.json'
    if sha(cfg) != m['config_sha256']:
        raise ValueError('config changed after prediction')
    c = j(cfg)
    if c.get('F1', {}).get('knots_u') != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError('frozen F1 knots changed')
    if c.get('F1', {}).get('no_reference_or_manual_weights') is not True:
        raise ValueError('F1 reference/manual weights flag changed')
    if c.get('constraints', {}).get('reference_read_during_prediction') is not False:
        raise ValueError('prediction was not reference-blind')
    if sha(base / 'inputs/common_points.json') != c.get('common_points_sha256'):
        raise ValueError('common points changed after prediction')
    return m, cfg


def replay_f0(details, F0, common):
    """Refit every F0 candidate from its original run-median points.

    This is a true coefficient replay, not a residual threshold check.
    Returns a metadata-only fallback explanation.
    """
    roi_mismatches = []
    fallback = []
    n_candidates = 0
    for iid, d in details.items():
        r0 = F0[iid]
        if d.get('F0_fits') is None:
            if r0.get('status') == 'candidate' or r0.get('fits') is not None:
                raise ValueError(f'{iid}: rejected F0 has candidate output')
            if d.get('J_status') != r0.get('status'):
                raise ValueError(f'{iid}: rejected status mismatch')
            continue
        if r0.get('status') != 'candidate' or not isinstance(r0.get('fits'), dict):
            raise ValueError(f'{iid}: candidate F0 status/fits mismatch')
        n_candidates += 1
        fits = d['F0_fits']
        points = d.get('points')
        support = d.get('support')
        paired_x = d.get('paired_x')
        if not isinstance(points, list) or not points:
            raise ValueError(f'{iid}: no original run points for F0 replay')
        if not isinstance(support, list) or len(support) != 2:
            raise ValueError(f'{iid}: invalid F0 support')
        xs = np.asarray([p.get('x') for p in points], dtype=float)
        if not np.isfinite(xs).all() or len(set(xs.tolist())) < 2:
            raise ValueError(f'{iid}: invalid F0 point x values')
        if [int(v) for v in paired_x] != [int(v) for v in xs]:
            raise ValueError(f'{iid}: F0 paired_x does not match original points')
        if support != r0.get('support') or paired_x != r0.get('paired_x'):
            raise ValueError(f'{iid}: F0 support/paired_x changed')
        if np.any(xs < support[0]) or np.any(xs > support[1]):
            raise ValueError(f'{iid}: F0 point outside support')
        if d.get('max_sample_gap') != r0.get('max_sample_gap'):
            raise ValueError(f'{iid}: F0 max_sample_gap changed')
        for key, ykey in (('sup', 'sup_y'), ('deep', 'deep_y')):
            ys = np.asarray([p.get(ykey) for p in points], dtype=float)
            if not np.isfinite(ys).all():
                raise ValueError(f'{iid}: non-finite F0 {key} point')
            want = np.asarray(fits.get(key), dtype=float)
            got = np.polyfit(xs, ys, 1)
            if want.shape != (2,) or not np.isfinite(want).all():
                raise ValueError(f'{iid}: invalid stored F0 {key} coefficients')
            if not np.allclose(got, want, rtol=0.0, atol=1e-10):
                raise ValueError(f'{iid}: F0 {key} coefficient replay mismatch')
            if not np.allclose(np.asarray(r0['fits'][key], float), got,
                               rtol=0.0, atol=1e-10):
                raise ValueError(f'{iid}: F0 output coefficient replay mismatch')
        # The source fit metadata may say r_frozen/full_image while the
        # prediction's roi_source is inherited from the J row. Only this
        # non-scoring label is allowed to differ.
        cr = common.get(iid, {})
        for k in set(fits) | set(r0['fits']):
            if k in ('roi_source',):
                continue
            if fits.get(k) != r0['fits'].get(k):
                raise ValueError(f'{iid}: F0 fit metadata changed: {k}')
        a = fits.get('roi_source')
        b = cr.get('J_source')
        if a != b:
            roi_mismatches.append({'image_id': iid, 'J_source': b,
                                   'F0_fit_roi_source': a})
        if cr.get('J_fallback'):
            fallback.append({'image_id': iid,
                             'J_fallback': cr.get('J_fallback'),
                             'J_fallback_reason': cr.get('J_fallback_reason'),
                             'J_source': b, 'F0_fit_roi_source': a})
    if n_candidates != 149:
        raise ValueError(f'F0 candidate count {n_candidates} != 149')
    return {'F0_candidates_replayed': n_candidates,
            'exact_coefficient_refit': True,
            'fallback_metadata_only': 'roi_source',
            'roi_source_semantics': (
                'F0 fits carry the ROI source of the frozen selected-pair '
                'geometry (r_frozen/pair_selection_full_image); J_source '
                'carries the pair-selection route (joint_pair/baseline/etc). '
                'The labels are not expected to be equal and neither enters '
                'the F0 coefficients, support, points or scoring.'),
            'roi_source_label_not_equal_count': len(roi_mismatches),
            'roi_source_label_not_equal': roi_mismatches,
            'fallback_rows': fallback,
            'fallback_metadata_only_explanation': (
                'For fallback rows, J_fallback/J_fallback_reason and the '
                'ROI-source label are retained as provenance. The F0 '
                'coefficients were independently refit exactly from the '
                'original run-median points; no fallback row changes support, '
                'paired_x or scoring.')}


def replay_f1(details):
    """Refit F1 from the same frozen points; compare every knot ordinate."""
    n = 0
    for iid, d in details.items():
        curve = d.get('F1_curve')
        if curve is None:
            if d.get('F1_failure') is None and d.get('J_status') == 'candidate':
                raise ValueError(f'{iid}: F1 curve missing without failure')
            continue
        points = [(p['x'], p['sup_y'], p['deep_y']) for p in d['points']]
        refit, reason = fit_curve(points, d['support'], d['height'],
                                  d['max_sample_gap'])
        if refit is None:
            raise ValueError(f'{iid}: F1 replay failed: {reason}')
        for key in ('sup', 'deep'):
            a = np.asarray(refit[key]['knots_y'], float)
            b = np.asarray(curve[key]['knots_y'], float)
            if a.shape != b.shape or not np.allclose(a, b, rtol=0.0,
                                                     atol=1e-10):
                raise ValueError(f'{iid}: F1 knot replay mismatch')
        n += 1
    return {'F1_candidates_replayed': n, 'exact_knot_refit': True}


def f0_score(rec, ref):
    if rec.get('status') != 'candidate' or rec.get('fits') is None:
        return False, None, None
    err = error_at_reference(rec['fits'], ref['reference_x'],
                             ref['reference_sup'], ref['reference_deep'])
    cs = correctness_support_valid(err)
    return bool(cs['correct']), cs, err


def f1_score(rec, ref):
    if rec.get('status') != 'candidate' or rec.get('curve') is None:
        return False, None, None
    curve = rec['curve']
    err = curve_errors(curve, ref['reference_x'], ref['reference_sup'],
                       ref['reference_deep'])
    valid = support_valid_mask(curve, ref['reference_x'])
    n = len(ref['reference_x'])
    sup = float(np.sum((err['se'] <= 10.0) & valid) / n)
    deep = float(np.sum((err['de'] <= 10.0) & valid) / n)
    cs = {'correct': bool(sup >= 0.8 and deep >= 0.8),
          'sup_coverage': sup, 'deep_coverage': deep}
    return bool(cs['correct']), cs, {'se': err['se'], 'de': err['de'],
                                     'valid_mask': valid}


def dist(vals):
    vals = [float(v) for v in vals]
    if not vals:
        return {'n': 0, 'median': None, 'p90': None, 'max': None,
                'mean': None}
    return {'n': len(vals), 'median': float(np.median(vals)),
            'p90': float(np.percentile(vals, 90)),
            'max': float(max(vals)), 'mean': float(np.mean(vals))}


def mean_or_none(a):
    return float(np.mean(a)) if len(a) else None


def evaluate(base, out, ref_path=REF, canonical=False):
    for stale in ('success.json', 'validation_failed.json'):
        p = out / stale
        if p.exists():
            p.unlink()
    try:
        manifest, cfg_path = verify(base)
        config_before = sha(cfg_path)
        details = unique_rows(j(base / 'predictions/fit_details.json'),
                              'fit_details', 153)
        F0 = unique_rows(j(base / 'predictions/F0_153.json'), 'F0', 153)
        F1 = unique_rows(j(base / 'predictions/F1_153.json'), 'F1', 153)
        if set(F0) != set(details) or set(F1) != set(details):
            raise ValueError('F0/F1/details identity sets differ')
        common_raw = j(base / 'inputs/common_points.json')
        common = unique_rows(common_raw['records'], 'common_points', 153)
        if set(common) != set(details):
            raise ValueError('common input identity set differs from predictions')
        refs, unclear = validate_reference(ref_path, details)
        replay = replay_f0(details, F0, common)
        replay.update(replay_f1(details))
        # F0/F1 must be status-compatible with fit details.
        for iid, d in details.items():
            if d.get('J_status') not in ('candidate', 'rejected'):
                raise ValueError(f'{iid}: invalid J status')
            if d['J_status'] == 'candidate' and F0[iid].get('status') != 'candidate':
                raise ValueError(f'{iid}: F0 status does not match candidate')
            if d['J_status'] == 'rejected' and F0[iid].get('status') != 'rejected':
                raise ValueError(f'{iid}: F0 status does not match rejection')
            if F1[iid].get('status') not in ('candidate', 'rejected', 'failed'):
                raise ValueError(f'{iid}: invalid F1 status')
        state = {'F0': {}, 'F1': {}}
        for iid, ref in refs.items():
            ok0, cs0, e0 = f0_score(F0[iid], ref)
            ok1, cs1, e1 = f1_score(F1[iid], ref)
            state['F0'][iid] = (ok0, cs0, e0)
            state['F1'][iid] = (ok1, cs1, e1)
        four = {'both': [], 'only_F0': [], 'only_F1': [], 'neither': []}
        for iid in sorted(refs):
            a, b = state['F0'][iid][0], state['F1'][iid][0]
            key = ('both' if a and b else 'only_F0' if a else
                   'only_F1' if b else 'neither')
            four[key].append(iid)
        if sum(len(v) for v in four.values()) != 143:
            raise ValueError('four-grid sum != 143')
        if len(four['both']) + len(four['only_F0']) != 128:
            raise ValueError(f'F0 correct did not reproduce 128: {four}')
        image_equal = {}
        group_equal = {}
        groups = {}
        for st in ('F0', 'F1'):
            rows = [{'group_id': details[i]['group_id'],
                     'ok': bool(state[st][i][0])} for i in refs]
            image_equal[st] = float(np.mean([r['ok'] for r in rows]))
            gg = defaultdict(list)
            for r in rows:
                gg[r['group_id']].append(1.0 if r['ok'] else 0.0)
            group_equal[st] = float(np.mean([np.mean(v) for v in gg.values()]))
            groups[st] = len(gg)
        cov = {}
        image_equal_dist = {}
        group_equal_dist = {}
        per_image_delta = []
        for st in ('F0', 'F1'):
            vals_s = []; vals_d = []
            se_all = []; de_all = []
            image_s = []; image_d = []
            per_image = {}
            for iid in refs:
                cs = state[st][iid][1]
                if cs is None:
                    vals_s.append(0.0); vals_d.append(0.0)
                    continue
                vals_s.append(cs['sup_coverage'])
                vals_d.append(cs['deep_coverage'])
                err = state[st][iid][2]
                valid = np.asarray(err['valid_mask'], bool)
                se = np.asarray(err['se'])[valid]
                de = np.asarray(err['de'])[valid]
                se_all.extend(se.tolist()); de_all.extend(de.tolist())
                image_s.append(float(se.mean())); image_d.append(float(de.mean()))
                per_image[iid] = {'sup': float(se.mean()),
                                  'deep': float(de.mean())}
            gg_s = defaultdict(list); gg_d = defaultdict(list)
            for iid in refs:
                if iid in per_image:
                    gid = details[iid]['group_id']
                    gg_s[gid].append(per_image[iid]['sup'])
                    gg_d[gid].append(per_image[iid]['deep'])
            cov[st] = {
                'n_images': 143, 'n_groups': len({
                    details[i]['group_id'] for i in refs}),
                'sup_image_equal': float(np.mean(vals_s)),
                'deep_image_equal': float(np.mean(vals_d)),
                'deep_group_equal': None,  # replaced below from actual deep rows
                'sup_distance_px': dist(se_all),
                'deep_distance_px': dist(de_all),
                'failed_images_zero': True,
            }
            # Correct group-equal coverage is computed from the per-image
            # coverage values, never from the sup list for both boundaries.
            gs_cov = defaultdict(list); gd_cov = defaultdict(list)
            for iid in refs:
                v = state[st][iid][1]
                gs_cov[details[iid]['group_id']].append(
                    v['sup_coverage'] if v else 0.0)
                gd_cov[details[iid]['group_id']].append(
                    v['deep_coverage'] if v else 0.0)
            cov[st]['sup_group_equal'] = float(np.mean(
                [np.mean(v) for v in gs_cov.values()]))
            cov[st]['deep_group_equal'] = float(np.mean(
                [np.mean(v) for v in gd_cov.values()]))
            image_equal_dist[st] = {
                'sup_px': dist(image_s), 'deep_px': dist(image_d),
                'n_images': len(image_s)}
            group_equal_dist[st] = {
                'sup_px': dist([mean_or_none(v) for v in gg_s.values()]),
                'deep_px': dist([mean_or_none(v) for v in gg_d.values()]),
                'n_groups': len(gg_s)}
        # Restructure per-image arms cleanly: recompute F0 and F1 maps.
        image_maps = {'F0': {}, 'F1': {}}
        for st in ('F0', 'F1'):
            for iid in refs:
                cs = state[st][iid][1]
                if cs is None:
                    continue
                err = state[st][iid][2]
                valid = np.asarray(err['valid_mask'], bool)
                image_maps[st][iid] = {
                    'sup': float(np.asarray(err['se'])[valid].mean()),
                    'deep': float(np.asarray(err['de'])[valid].mean())}
        per_image_delta = []
        for iid in sorted(refs):
            if iid not in image_maps['F0'] or iid not in image_maps['F1']:
                continue
            a = image_maps['F0'][iid]; b = image_maps['F1'][iid]
            per_image_delta.append({
                'image_id': iid, 'group_id': details[iid]['group_id'],
                'sup_F0_px': a['sup'], 'sup_F1_px': b['sup'],
                'deep_F0_px': a['deep'], 'deep_F1_px': b['deep'],
                'delta_sup_px': b['sup'] - a['sup'],
                'delta_deep_px': b['deep'] - a['deep'],
                'improved_sup': b['sup'] < a['sup'],
                'improved_deep': b['deep'] < a['deep'],
                'degraded_sup': b['sup'] > a['sup'],
                'degraded_deep': b['deep'] > a['deep'],
            })
        by_group = {}
        for r in per_image_delta:
            by_group.setdefault(r['group_id'], []).append(r)
        group_delta = []
        for gid, rs in sorted(by_group.items()):
            n = len(rs)
            ds = float(np.mean([r['delta_sup_px'] for r in rs]))
            dd = float(np.mean([r['delta_deep_px'] for r in rs]))
            group_delta.append({
                'group_id': gid, 'n_images': n,
                'delta_sup_px': ds, 'delta_deep_px': dd,
                'improved_sup': ds < 0, 'improved_deep': dd < 0,
                'degraded_sup': ds > 0, 'degraded_deep': dd > 0,
                'image_ids': [r['image_id'] for r in rs]})
        # One-to-one common positions F0/F1, with both fits available.
        common_positions = []
        for iid, ref in refs.items():
            e0 = state['F0'][iid][2]
            e1 = state['F1'][iid][2]
            if e0 is None or e1 is None:
                continue
            v = np.asarray(e0['valid_mask']) & np.asarray(e1['valid_mask'])
            for k, x in enumerate(ref['reference_x']):
                if v[k]:
                    common_positions.append({
                        'image_id': iid, 'reference_x': int(x),
                        'F0_sup_error_px': float(e0['se'][k]),
                        'F0_deep_error_px': float(e0['de'][k]),
                        'F1_sup_error_px': float(e1['se'][k]),
                        'F1_deep_error_px': float(e1['de'][k])})
        if len({(r['image_id'], r['reference_x'])
                for r in common_positions}) != len(common_positions):
            raise ValueError('common positions are not one-to-one')
        common_doc = {
            'n_images': len({r['image_id'] for r in common_positions}),
            'n_unique_positions': len(common_positions),
            'n_total_arm_observations': 2 * len(common_positions),
            'F0_sup': dist([r['F0_sup_error_px'] for r in common_positions]),
            'F0_deep': dist([r['F0_deep_error_px'] for r in common_positions]),
            'F1_sup': dist([r['F1_sup_error_px'] for r in common_positions]),
            'F1_deep': dist([r['F1_deep_error_px'] for r in common_positions]),
            'delta_F1_minus_F0_sup': dist(
                [r['F1_sup_error_px'] - r['F0_sup_error_px']
                 for r in common_positions]),
            'delta_F1_minus_F0_deep': dist(
                [r['F1_deep_error_px'] - r['F0_deep_error_px']
                 for r in common_positions]),
            'position_improved_sup': sum(
                r['F1_sup_error_px'] < r['F0_sup_error_px']
                for r in common_positions),
            'position_degraded_sup': sum(
                r['F1_sup_error_px'] > r['F0_sup_error_px']
                for r in common_positions),
            'position_equal_sup': sum(
                r['F1_sup_error_px'] == r['F0_sup_error_px']
                for r in common_positions),
            'position_improved_deep': sum(
                r['F1_deep_error_px'] < r['F0_deep_error_px']
                for r in common_positions),
            'position_degraded_deep': sum(
                r['F1_deep_error_px'] > r['F0_deep_error_px']
                for r in common_positions),
            'position_equal_deep': sum(
                r['F1_deep_error_px'] == r['F0_deep_error_px']
                for r in common_positions),
            'rows': common_positions,
        }
        # 153 outputs and status.
        all_rows = []
        for iid in sorted(F0):
            r0 = F0[iid]; r1 = F1[iid]
            ref = refs.get(iid)
            all_rows.append({
                'image_id': iid, 'group_id': details[iid]['group_id'],
                'size': f"{details[iid]['width']}x{details[iid]['height']}",
                'reference_status': ('fixed_143' if ref else
                                     'target_unclear_10'),
                'F0_status': r0['status'], 'F1_status': r1['status'],
                'F1_failure': r1.get('failure'),
                'F0_correct': (None if ref is None else state['F0'][iid][0]),
                'F1_correct': (None if ref is None else state['F1'][iid][0]),
                'J_selected_pair_key': details[iid]['selected_pair_key'],
                'J_fallback': details[iid].get('J_fallback'),
            })
        focus = [r for r in all_rows if r['image_id'] in FOCUS]
        out_doc = {
            'version': 'round25_corrected_evaluation_v1',
            'prediction_manifest_sha256': sha(base / 'predictions/manifest.json'),
            'reference_values_read_after_freeze': True,
            'reference_sha256': sha(ref_path),
            'reference_source_sha256': sha(REF_SOURCE),
            'reference_identity_validation': {
                'n_143': len(refs), 'n_unclear_10': len(unclear),
                'unique_ids': True, 'dims_match_predictions': True,
                'arrays_finite_and_aligned': True,
                'source_manifest_hash_checked': True},
            'config_read_only': {'sha256_before': config_before,
                                 'sha256_after': sha(cfg_path),
                                 'unchanged': config_before == sha(cfg_path)},
            'F0_F1_replay': replay,
            'four_grid_143': {k: {'n': len(v), 'ids': sorted(v)}
                              for k, v in four.items()},
            'net_F1_minus_F0': len(four['only_F1']) - len(four['only_F0']),
            'image_equal': image_equal,
            'group_equal': {st: {'rate': group_equal[st],
                                 'n_groups': groups[st]}
                            for st in ('F0', 'F1')},
            'paired_group_difference_F1_minus_F0': (
                group_equal['F1'] - group_equal['F0']),
            'coverage_distance': cov,
            'image_equal_distance': image_equal_dist,
            'group_equal_distance': group_equal_dist,
            'per_image_distance_delta': {
                'rows': per_image_delta,
                'n_images_with_both_arms': len(per_image_delta),
                'n_improved_sup': sum(r['improved_sup'] for r in per_image_delta),
                'n_degraded_sup': sum(r['degraded_sup'] for r in per_image_delta),
                'n_improved_deep': sum(r['improved_deep'] for r in per_image_delta),
                'n_degraded_deep': sum(r['degraded_deep'] for r in per_image_delta),
                'improved_sup_ids': [r['image_id'] for r in per_image_delta
                                     if r['improved_sup']],
                'degraded_sup_ids': [r['image_id'] for r in per_image_delta
                                     if r['degraded_sup']],
                'improved_deep_ids': [r['image_id'] for r in per_image_delta
                                      if r['improved_deep']],
                'degraded_deep_ids': [r['image_id'] for r in per_image_delta
                                      if r['degraded_deep']],
            },
            'per_group_distance_delta': {'rows': group_delta,
                                         'n_groups': len(group_delta)},
            'common_positions': common_doc,
            'all_153': {
                'n': len(all_rows),
                'F0_candidate': sum(r['F0_status'] == 'candidate' for r in all_rows),
                'F1_candidate': sum(r['F1_status'] == 'candidate' for r in all_rows),
                'F0_rejected': sum(r['F0_status'] == 'rejected' for r in all_rows),
                'F1_rejected': sum(r['F1_status'] == 'rejected' for r in all_rows),
                'F1_failed': sum(r['F1_status'] == 'failed' for r in all_rows),
                'new_outputs': [r['image_id'] for r in all_rows
                                if r['F0_status'] != 'candidate' and
                                r['F1_status'] == 'candidate'],
                'lost_outputs': [r['image_id'] for r in all_rows
                                 if r['F0_status'] == 'candidate' and
                                 r['F1_status'] != 'candidate'],
                'unclear_10': [r['image_id'] for r in all_rows
                               if r['reference_status'] == 'target_unclear_10'],
                'rows': all_rows,
            },
            'focus_7': focus,
            'fit_failures': [r for r in all_rows if r['F1_status'] == 'failed'],
            'all_predictions_unchanged': True,
        }
        if out_doc['config_read_only']['sha256_after'] != config_before:
            raise ValueError('config changed during evaluation')
        write_json(out / 'evaluation/summary.json', out_doc)
        write_json(out / 'evaluation/four_grid_143.json', out_doc['four_grid_143'])
        write_json(out / 'evaluation/coverage_distance.json', cov)
        write_json(out / 'evaluation/image_group_equal_distance.json',
                   {'image_equal': image_equal_dist,
                    'group_equal': group_equal_dist})
        write_json(out / 'evaluation/per_image_distance_delta.json',
                   out_doc['per_image_distance_delta'])
        write_json(out / 'evaluation/per_group_distance_delta.json',
                   out_doc['per_group_distance_delta'])
        write_json(out / 'evaluation/common_positions.json', common_doc)
        write_json(out / 'evaluation/per_image_153.json', all_rows)
        write_json(out / 'success.json', {
            'version': 'round25_corrected_success_v1',
            'prediction_manifest_sha256': sha(base / 'predictions/manifest.json'),
            'reference_sha256': sha(ref_path),
            'reference_read_after_freeze': True,
            'n_153': 153, 'n_143': 143,
            'four_grid_total': sum(len(v) for v in four.values()),
        })
        return out_doc
    except Exception as e:
        write_json(out / 'validation_failed.json', {
            'version': 'round25_failure_v1', 'error': str(e),
            'success_written': False})
        if (out / 'success.json').exists():
            (out / 'success.json').unlink()
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(CORRECTED_BASE))
    args = ap.parse_args()
    base = BASE
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.resolve() != CORRECTED_BASE.resolve():
        raise ValueError('corrected evaluator has a fixed output path; '
                         'refusing arbitrary --out rewrite')
    out.mkdir(parents=True, exist_ok=True)
    doc = evaluate(base, out, canonical=True)
    print(json.dumps({
        'four_grid': {k: v['n'] for k, v in doc['four_grid_143'].items()},
        'net_F1_minus_F0': doc['net_F1_minus_F0'],
        'image_equal': doc['image_equal'],
        'group_equal': {k: v['rate'] for k, v in doc['group_equal'].items()},
        'coverage': {st: {'sup': doc['coverage_distance'][st]['sup_image_equal'],
                          'deep': doc['coverage_distance'][st]['deep_image_equal'],
                          'deep_group': doc['coverage_distance'][st]['deep_group_equal']}
                     for st in ('F0', 'F1')},
        'common': {'n': doc['common_positions']['n_unique_positions'],
                   'images': doc['common_positions']['n_images']},
    }, indent=2))


if __name__ == '__main__':
    main()
