"""Round25 fixed four-segment continuous piecewise-linear boundary fit.

F0 is the frozen Round24 J global straight-line fit on the same selected pair.
F1 uses the same selected pair, same original paired_x and same run-median
observations, with five fixed knots at u=0,.25,.5,.75,1 and OLS interpolation.
No reference arrays enter prediction.
"""
import math
import numpy as np

KNOTS_U = (0.0, 0.25, 0.50, 0.75, 1.0)
TOL = 1e-8


def points_from_run_identity(run_identity):
    """Return (x, sup_median, deep_median) from saved per-column run members."""
    out = []
    for g in run_identity or []:
        su = g.get('sup_ys') or []
        de = g.get('deep_ys') or []
        if not su or not de:
            raise ValueError('empty run member identity')
        out.append((int(g['x']), float(np.median(su)), float(np.median(de))))
    return out


def _design(x, lo, hi):
    u = (np.asarray(x, dtype=float) - float(lo)) / float(hi - lo)
    A = np.zeros((len(u), len(KNOTS_U)), dtype=float)
    for i, val in enumerate(u):
        if val < 0 or val > 1 or not np.isfinite(val):
            return None, None
        j = min(int(math.floor(val * 4)), 3)
        t = val * 4 - j
        A[i, j] += 1.0 - t
        A[i, j + 1] += t
    return A, u


def fit_piecewise(points, support, height, y_key='sup'):
    """Fit five knot values to original run median observations."""
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1 if y_key == 'sup' else 2] for p in points],
                    dtype=float)
    lo, hi = int(support[0]), int(support[1])
    if hi <= lo:
        return None, 'zero_support_span'
    A, u = _design(xs, lo, hi)
    if A is None:
        return None, 'point_outside_support'
    if not np.isfinite(ys).all():
        return None, 'nonfinite_observation'
    rank = np.linalg.matrix_rank(A)
    if rank < len(KNOTS_U):
        return None, f'rank_deficient_{rank}'
    try:
        knots, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    except np.linalg.LinAlgError:
        return None, 'least_squares_failure'
    if not np.isfinite(knots).all():
        return None, 'nonfinite_knots'
    res = np.abs(A @ knots - ys)
    p90 = float(np.quantile(res, 0.90)) if len(res) else None
    if p90 is None or p90 > max(3.0, 0.02 * float(height)):
        return None, f'residual_p90_{p90}'
    return {
        'knots_u': list(KNOTS_U),
        'knots_x': [lo + u0 * (hi - lo) for u0 in KNOTS_U],
        'knots_y': [float(v) for v in knots],
        'support': [lo, hi],
        'paired_x': [int(x) for x in xs],
        'max_sample_gap': None,
        'image_shape': [int(height), None],
        'residual_p90_px': p90,
        'residuals_px': [float(v) for v in res],
        'rank': int(rank), 'n_observations': len(xs),
    }, ''


def curve_value(fit, x):
    """Piecewise-linear y(x); None outside support."""
    if fit is None:
        return None
    lo, hi = fit['support']
    if x < lo or x > hi or hi <= lo:
        return None
    u = (float(x) - lo) / float(hi - lo)
    j = min(int(math.floor(u * 4)), 3)
    t = u * 4 - j
    return float(fit['knots_y'][j] * (1.0 - t) + fit['knots_y'][j + 1] * t)


def fit_curve(points, support, height, max_gap):
    sup, rs = fit_piecewise(points, support, height, 'sup')
    if sup is None:
        return None, f'sup:{rs}'
    deep, rd = fit_piecewise(points, support, height, 'deep')
    if deep is None:
        return None, f'deep:{rd}'
    if any(a >= b for a, b in zip(sup['knots_y'], deep['knots_y'])):
        return None, 'crossed_or_inverted_knots'
    sup.update({'max_sample_gap': float(max_gap),
                'image_shape': [int(height), None]})
    deep.update({'max_sample_gap': float(max_gap),
                 'image_shape': [int(height), None]})
    return {'sup': sup, 'deep': deep,
            'support': [int(support[0]), int(support[1])],
            'paired_x': [int(x) for x, _, _ in points],
            'max_sample_gap': float(max_gap),
            'knots_u': list(KNOTS_U)}, ''


def curve_errors(curve, ref_x, ref_sup, ref_deep):
    if curve is None:
        return None
    se = []
    de = []
    for x, ys, yd in zip(ref_x, ref_sup, ref_deep):
        py = curve_value(curve['sup'], x)
        pd = curve_value(curve['deep'], x)
        se.append(abs(py - float(ys)) if py is not None else math.inf)
        de.append(abs(pd - float(yd)) if pd is not None else math.inf)
    return {'se': np.asarray(se), 'de': np.asarray(de)}
