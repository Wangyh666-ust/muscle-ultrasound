"""Round27 diagnostic-only geometry and candidate correspondence; no model code."""
import math
import numpy as np
from scripts.fascicle_pa_fl import fl_for_segment
from scripts.fl_acceptance_policy import policy_fields
from scripts.fascicle_geometry import _overlap_fraction

TOL = 1e-8


def roots_on_curve(seg, fit):
    """All intersections with finite piecewise segments. Never extend an end segment."""
    cx, cy = seg['centroid']
    th = math.radians(seg['direction_deg']); dx, dy = math.cos(th), math.sin(th)
    roots = []; overlaps = []
    xs, ys = fit['knots_x'], fit['knots_y']
    for k in range(len(xs) - 1):
        lo, hi = xs[k:k+2]
        slope = (ys[k+1] - ys[k]) / (hi - lo)
        intercept = ys[k] - slope * lo
        denom = dy - slope * dx
        rhs = slope * cx + intercept - cy
        if abs(denom) < 1e-12:
            if abs(rhs) < TOL: overlaps.append(k)
            continue
        t = rhs / denom; x = cx + t * dx
        if lo - TOL <= x <= hi + TOL:
            if not any(abs(t - r['t']) <= TOL for r in roots):
                roots.append({'t': float(t), 'point': [float(x), float(cy+t*dy)],
                              'piece': k, 'line': [float(slope), float(intercept)]})
    return {'roots': sorted(roots, key=lambda r: r['t']), 'coincident_pieces': overlaps}


def compact_fl(fl):
    return {**{k: fl.get(k) for k in ('fl_px_raw', 'visible_length_px',
        'extrapolated_length_px', 'extrapolation_ratio', 'intersection_sup',
        'intersection_deep', 'fl_failure_conditions', 'fl_local_support_missing')},
        'G0': policy_fields(fl, 'G0'), 'G1': policy_fields(fl, 'G1')}


def piecewise_fl(seg, curve, template, width, height):
    """Locate unique finite curve roots, then reuse the unchanged acceptance implementation."""
    roots = {}; reasons = []
    if curve is None:
        reasons.append('piecewise_fit_unavailable')
    else:
        for side in ('sup', 'deep'):
            roots[side] = roots_on_curve(seg, curve[side])
            if roots[side]['coincident_pieces']:
                reasons.append(side + '_coincident_with_boundary')
            elif len(roots[side]['roots']) == 0:
                reasons.append(side + '_no_intersection_in_defined_support')
            elif len(roots[side]['roots']) > 1:
                reasons.append(side + '_multiple_intersections')
    if reasons:
        empty = {'fl_px_raw': None, 'fl_failure_conditions_pure': reasons,
                 'fl_failure_conditions': reasons}
        return {**compact_fl(empty), 'definition_status': 'unavailable', 'roots': roots}
    # This changes only the boundary equations at the analytically located intersections.
    local = dict(template)
    for side in ('sup', 'deep'):
        local[side] = roots[side]['roots'][0]['line']
    result = fl_for_segment(seg, local, width, height, apo_ok=True)
    for side in ('sup', 'deep'):
        if result.get('intersection_' + side) is not None:
            if not np.allclose(result['intersection_' + side], roots[side]['roots'][0]['point'],
                               atol=TOL, rtol=1e-10):
                raise ValueError('piecewise / frozen FL intersection mismatch')
    return {**compact_fl(result), 'definition_status': 'defined', 'roots': roots}


def fixed_selection(candidates, labels, fit_arm, policy):
    """No reselection or partial averaging when one originally selected line fails."""
    by_id = {c['label_id']: c for c in candidates}
    if len(by_id) != len(candidates) or len(set(labels)) != len(labels):
        raise ValueError('duplicate candidate/selected identity')
    rows = [by_id[i][fit_arm] for i in labels]
    usable = bool(rows) and all(r[policy]['accept'] and r['fl_px_raw'] is not None for r in rows)
    raw_defined = bool(rows) and all(r['fl_px_raw'] is not None for r in rows)
    return {'selected_ids': labels, 'n_requested': len(labels),
            'n_accepted': sum(r[policy]['accept'] for r in rows),
            'value_px': float(np.mean([r['fl_px_raw'] for r in rows])) if usable else None,
            'raw_mean_px': float(np.mean([r['fl_px_raw'] for r in rows])) if raw_defined else None,
            'failed_ids': [i for i, r in zip(labels, rows) if not r[policy]['accept']]}


def correspond(a, b, perp_tol=10., angle_tol=10., overlap_min=.2):
    """Mutual unique best geometric match using frozen primary tolerances, never FL/GT."""
    edges = []
    for i, x in enumerate(a):
        for k, y in enumerate(b):
            ad = abs(x['direction_deg'] - y['direction_deg']) % 180
            ad = min(ad, 180-ad)
            delta = np.subtract(y['centroid'], x['centroid'])
            p = []
            for s in (x, y):
                t = math.radians(s['direction_deg'])
                p.append(abs(float(delta @ np.array([-math.sin(t), math.cos(t)]))))
            ov = [_overlap_fraction(x, y), _overlap_fraction(y, x)]
            if ad <= angle_tol and max(p) <= perp_tol and min(ov) >= overlap_min:
                edges.append({'i': i, 'j': k, 'angle_diff_deg': ad,
                              'max_perp_px': max(p), 'min_overlap': min(ov),
                              'score': max(p)/perp_tol + ad/angle_tol})
    def best(which, count):
        out = {}
        for i in range(count):
            es = sorted([e for e in edges if e[which] == i], key=lambda e: e['score'])
            if es and (len(es) == 1 or es[1]['score'] - es[0]['score'] > 1e-10):
                out[i] = es[0]
        return out
    aa, bb = best('i', len(a)), best('j', len(b))
    pairs = [e for i, e in aa.items() if e['j'] in bb and bb[e['j']]['i'] == i]
    return {'pairs': pairs, 'n_admissible_edges': len(edges),
            'unmatched_C0': sorted(set(range(len(a))) - {e['i'] for e in pairs}),
            'unmatched_C1': sorted(set(range(len(b))) - {e['j'] for e in pairs}),
            'interpretation': 'conditional geometric correspondence, not verified anatomical identity'}


def mean_change_decomposition(c0, c1, selected0, selected1, pairs):
    """Exact symmetric algebra; unresolved composition is not called selection causality."""
    if not selected0 or not selected1:
        return {'status': 'no_common_image_output'}
    a = {c['label_id']: c for c in c0}; b = {c['label_id']: c for c in c1}
    if len(a) != len(c0) or len(b) != len(c1): raise ValueError('duplicate label')
    if len(selected0) != len(set(selected0)) or len(selected1) != len(set(selected1)):
        raise ValueError('duplicate selection')
    if len({p['i'] for p in pairs}) != len(pairs) or len({p['j'] for p in pairs}) != len(pairs):
        raise ValueError('nonunique correspondence')
    w0 = {i: 1/len(selected0) for i in selected0}; w1 = {i: 1/len(selected1) for i in selected1}
    value = 0.; weight = 0.; used0 = set(); used1 = set(); terms = []
    for p in pairs:
        x, y = c0[p['i']], c1[p['j']]
        l0, l1 = x['F0']['fl_px_raw'], y['F0']['fl_px_raw']
        if l0 is None or l1 is None: continue
        id0, id1 = x['label_id'], y['label_id']
        u, v = w0.get(id0, 0), w1.get(id1, 0)
        dv = .5*(u+v)*(l1-l0); dw = .5*(l0+l1)*(v-u)
        value += dv; weight += dw; used0.add(id0); used1.add(id1)
        if u or v: terms.append({'C0_label': id0, 'C1_label': id1,
            'C0_px': l0, 'C1_px': l1, 'C0_weight': u, 'C1_weight': v,
            'value_term_px': dv, 'selection_weight_term_px': dw})
    missing0 = [i for i in selected0 if i not in used0]
    missing1 = [i for i in selected1 if i not in used1]
    unresolved = sum(w1[i]*b[i]['F0']['fl_px_raw'] for i in missing1) - sum(
        w0[i]*a[i]['F0']['fl_px_raw'] for i in missing0)
    m0 = sum(w0[i]*a[i]['F0']['fl_px_raw'] for i in selected0)
    m1 = sum(w1[i]*b[i]['F0']['fl_px_raw'] for i in selected1)
    residual = (m1-m0) - (value+weight+unresolved)
    if abs(residual) > 1e-7: raise ValueError('decomposition identity does not close')
    return {'status': 'ok', 'C0_px': m0, 'C1_px': m1, 'delta_px': m1-m0,
        'matched_value_term_px': value, 'matched_selection_weight_term_px': weight,
        'unmatched_composition_term_px': unresolved, 'closure_error_px': residual,
        'selected_without_finite_match_C0': missing0, 'selected_without_finite_match_C1': missing1,
        'all_selected_have_finite_matches': not missing0 and not missing1, 'terms': terms}


def aggregation_diagnostic(candidates, labels):
    rows = {c['label_id']: c['F0'] for c in candidates}
    vals = [rows[i]['fl_px_raw'] for i in labels]
    if not vals: return {'n': 0, 'mean_px': None, 'median_three_px': None, 'lines': []}
    m = float(np.mean(vals))
    return {'n': len(vals), 'mean_px': m,
        'median_three_px': float(np.median(vals)) if len(vals) == 3 else None,
        'range_px': float(max(vals)-min(vals)),
        'lines': [{'label_id': i, 'fl_px': v, 'contribution_to_mean_px': v/len(vals),
                   'leave_one_out_change_px': (m-v)/(len(vals)-1) if len(vals) > 1 else None}
                  for i, v in zip(labels, vals)]}
