"""Frozen G0/G1 FL acceptance policies.

G0 is the existing pure rule exactly. G1 changes only two conditions from hard
rejection to diagnostic status:

    intersection_outside_image
    intersection_outside_apo_support

G1 keeps the 0.80 extrapolation limit, apo/no-apo/parallel/zero-span conditions,
all other conditions, candidate extraction, ordering, selection and averaging.
Raw intersections/lengths are never recomputed or changed by these helpers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fascicle_pa_fl import select_candidates  # noqa: E402

G1_DIAGNOSTIC_ONLY = (
    'intersection_outside_image',
    'intersection_outside_apo_support',
)


def policy_failure_conditions(pure_conditions, policy):
    conditions = list(pure_conditions or [])
    if policy == 'G0':
        return conditions
    if policy == 'G1':
        return [c for c in conditions if c not in G1_DIAGNOSTIC_ONLY]
    raise ValueError(f'unknown FL policy: {policy}')


def policy_fields(fl, policy):
    """Return only the policy-specific acceptance fields for an FL result."""
    conditions = policy_failure_conditions(fl.get('fl_failure_conditions_pure'), policy)
    accept = not conditions
    return {
        'accept': accept,
        'status': 'ok' if accept else 'rejected',
        'conditions': conditions,
        'fl_px_pure': fl.get('fl_px_raw') if accept else None,
    }


def policy_from_candidate(candidate, policy):
    conditions = policy_failure_conditions(
        candidate.get('original_fl_failure_conditions_pure',
                      candidate.get('fl_failure_conditions_pure')), policy)
    accept = not conditions
    return {
        'accept': accept,
        'status': 'ok' if accept else 'rejected',
        'conditions': conditions,
        'fl_px_pure': candidate.get('fl_px_raw') if accept else None,
    }


def policy_equal(a, b, atol=1e-9, rtol=1e-12):
    def close(x, y):
        if x is None or y is None:
            return x is None and y is None
        return abs(float(x) - float(y)) <= atol + rtol * abs(float(y))
    return (bool(a['accept']) == bool(b['accept'])
            and a['status'] == b['status']
            and set(a['conditions'] or []) == set(b['conditions'] or [])
            and close(a['fl_px_pure'], b['fl_px_pure']))


def apply_policy_to_candidate(candidate, policy):
    """Copy one evaluated candidate and change only acceptance fields."""
    c = dict(candidate)
    old_conditions = list(candidate.get(
        'original_fl_failure_conditions_pure',
        candidate.get('fl_failure_conditions_pure')) or [])
    conditions = policy_failure_conditions(old_conditions, policy)
    accept = not conditions
    c['original_fl_failure_conditions_pure'] = old_conditions
    c['original_fl_accept_pure'] = bool(candidate.get(
        'original_fl_accept_pure', candidate.get('fl_accept_pure')))
    c['original_fl_status_pure'] = candidate.get(
        'original_fl_status_pure', candidate.get('fl_status_pure'))
    c['original_fl_px_pure'] = candidate.get(
        'original_fl_px_pure', candidate.get('fl_px_pure'))
    c['policy'] = policy
    c['policy_failure_conditions_pure'] = conditions
    c['policy_accept_pure'] = accept
    c['policy_status_pure'] = 'ok' if accept else 'rejected'
    c['policy_fl_px_pure'] = (
        candidate.get('fl_px_raw') if accept else None)
    # Selection calls the existing function, which reads these two fields.
    c['fl_accept_pure'] = accept
    c['fl_status_pure'] = c['policy_status_pure']
    c['fl_px_pure'] = c['policy_fl_px_pure']
    c['fl_failure_conditions_pure'] = conditions
    # Strict acceptance and all raw geometry fields are deliberately untouched.
    return c


def apply_policy(candidates, policy):
    return [apply_policy_to_candidate(c, policy) for c in candidates]


def select_with_policy(candidates, policy):
    """Use the existing selector unchanged, with policy-adjusted pure fields."""
    adjusted = apply_policy(candidates, policy)
    return select_candidates(adjusted), adjusted


def raw_geometry_equal(a, b):
    """Check that only acceptance fields differ between two policy copies."""
    fields = ('fl_px_raw', 'visible_length_px', 'extrapolated_length_px',
              'extrapolation_ratio', 'intersection_sup', 'intersection_deep',
              'fl_outside_image', 'fl_outside_apo_support',
              'fl_local_support_missing', 'fl_excessive_extrapolation')
    for k in fields:
        if a.get(k) != b.get(k):
            return False, k
    return True, None
