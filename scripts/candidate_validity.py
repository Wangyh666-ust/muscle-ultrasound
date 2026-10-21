"""Diagnostic validity of a fascicle candidate against the double apo region.

This module is intentionally NOT used by PA/FL selection. It checks:
  * whether the candidate lies between the superficial and deep aponeurosis
    fits (sampled along the segment), and
  * whether the apo fit has local paired-column support near the candidate
    (within the original geometry's max_sample_gap).

The result is reported separately from the step-1 wiring fix so that wiring
changes and candidate-validity diagnostics are not conflated.
"""
import numpy as np

DEFAULT_N_SAMPLES = 21
INSIDE_FRACTION_THRESHOLD = 0.5
LOCAL_SUPPORT_THRESHOLD = 0.5


def nearest_paired_distance(fits, x):
    if fits is None:
        return None
    paired = np.asarray(fits.get('paired_x', []), dtype=float)
    if len(paired) == 0:
        return float('inf')
    return float(np.min(np.abs(paired - float(x))))


_nearest_paired_distance = nearest_paired_distance


def check_candidate_region(seg, apo_fits, n_samples=DEFAULT_N_SAMPLES):
    """Return region/local-support diagnostics for one candidate segment."""
    if apo_fits is None:
        return {
            'region_status': 'unknown',
            'region_inside_fraction': None,
            'region_samples': 0,
            'local_support_status': 'unknown',
            'local_support_fraction': None,
            'candidate_validity_status': 'unknown',
            'candidate_validity_conditions': ['no_apo_boundaries'],
        }
    x0, y0 = map(float, seg['p0'])
    x1, y1 = map(float, seg['p1'])
    xs = np.linspace(x0, x1, n_samples)
    ys = np.linspace(y0, y1, n_samples)
    sup_y = np.polyval(apo_fits['sup'], xs)
    deep_y = np.polyval(apo_fits['deep'], xs)
    ordered = sup_y < deep_y
    inside = (ys >= sup_y) & (ys <= deep_y) & ordered
    inside_fraction = float(np.mean(inside))
    if inside_fraction >= 0.8:
        region_status = 'inside'
    elif inside_fraction > 0.0:
        region_status = 'partially_outside'
    else:
        region_status = 'outside'

    max_gap = float(apo_fits.get('max_sample_gap', 0.0))
    distances = [_nearest_paired_distance(apo_fits, x) for x in xs]
    local_ok = np.array([d is not None and d <= max_gap for d in distances])
    local_fraction = float(np.mean(local_ok)) if len(local_ok) else 0.0
    if local_fraction >= 0.8:
        local_status = 'supported'
    elif local_fraction > 0.0:
        local_status = 'partial'
    else:
        local_status = 'missing'

    conditions = []
    if inside_fraction < INSIDE_FRACTION_THRESHOLD:
        conditions.append('candidate_outside_band_region')
    if local_fraction < LOCAL_SUPPORT_THRESHOLD:
        conditions.append('candidate_local_support_gap')
    return {
        'region_status': region_status,
        'region_inside_fraction': inside_fraction,
        'region_samples': int(n_samples),
        'region_min_sup_y': float(np.min(sup_y)),
        'region_max_deep_y': float(np.max(deep_y)),
        'local_support_status': local_status,
        'local_support_fraction': local_fraction,
        'local_support_max_gap_px': max_gap,
        'local_support_min_distance_px': (float(np.nanmin(distances))
                                          if distances and np.isfinite(np.nanmin(distances))
                                          else None),
        'candidate_validity_status': 'valid' if not conditions else 'invalid',
        'candidate_validity_conditions': conditions,
    }
