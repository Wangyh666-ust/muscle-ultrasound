"""Frozen apo MT baseline entry: baseline geometry + conservative pair selection.

This is the reusable version of the ROUND4 conservative integration:

    baseline candidate                -> keep baseline
    baseline reason ambiguous_multiple_bands -> use frozen pair selector
    any other baseline rejection      -> keep baseline rejection

The model, mask, ROI, threshold, near-run merging, fit residuals and scoring
protocol are unchanged. Predictions are consumed from the frozen dev cache;
this module never trains or re-thresholds.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.umud_baseline.geometry import extract_apo_bands, mt_at_positions  # noqa: E402
from scripts.pair_selection import measure_mt_selected  # noqa: E402

VERSIONED_RULE = ROOT / 'manifest/apo_pair_selection_rule_v1.json'
RUN_RULE = ROOT / 'runs/geometry_eval/apo_pair_selection_rule_v1/rule.json'
DEFAULT_RULE = VERSIONED_RULE if VERSIONED_RULE.exists() else RUN_RULE
FROZEN_RULE_VERSION = 'apo_pair_selection_v1'
PRIMARY_TOL_PX = 10
PRIMARY_COVERAGE_THRESHOLD = 0.8


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_rule(rule_path=DEFAULT_RULE):
    """Load and verify the train-frozen pair-selection rule."""
    rule_path = Path(rule_path)
    rule = json.loads(rule_path.read_text(encoding='utf-8'))
    if rule.get('version') != FROZEN_RULE_VERSION:
        raise ValueError(f'Unexpected rule version: {rule.get("version")}')
    if not rule.get('frozen_before_dev'):
        raise ValueError('Rule was not frozen before dev')
    hash_path = rule_path.with_suffix('.sha256')
    if not hash_path.exists():
        hash_path = rule_path.parent / 'rule.sha256'
    expected = hash_path.read_text(encoding='utf-8').strip()
    actual = sha256_file(rule_path)
    if actual != expected:
        raise ValueError('Frozen rule hash mismatch')
    return rule, actual


def baseline_result(pred_mask):
    """Original geometry: extract_apo_bands + quartile MT positions."""
    fits, reason = extract_apo_bands(pred_mask)
    if fits is None:
        return {'status': 'rejected', 'reason': reason}, None
    lo, hi = fits['support']
    xs = [lo + f * (hi - lo) for f in (.25, .5, .75)]
    mt, reason = mt_at_positions(fits, xs)
    if reason:
        return {'status': 'rejected', 'reason': reason}, fits
    return {'status': 'candidate', 'x_positions': list(map(float, xs)),
            'mt_px': mt, 'mt_mean_px': float(np.mean(mt)),
            'support': list(fits['support'])}, fits


def conservative_result(base_res, base_fits, new_res, new_fits):
    """Preserve baseline except for the single pre-registered ambiguous case."""
    if base_res['status'] == 'candidate':
        return base_res, base_fits, 'baseline'
    if base_res.get('reason') == 'ambiguous_multiple_bands':
        return new_res, new_fits, 'new_pair_selection'
    return base_res, base_fits, 'baseline_rejected'


def measure_conservative(pred_mask, rule):
    """Reusable prediction-mode entry. Human/reference data is not accepted."""
    base_res, base_fits = baseline_result(pred_mask)
    new_res, new_fits, new_meta = measure_mt_selected(pred_mask, rule)
    cons_res, cons_fits, source = conservative_result(
        base_res, base_fits, new_res, new_fits)
    return {
        'status': cons_res['status'],
        'reason': cons_res.get('reason', ''),
        'source': source,
        'result': cons_res,
        'fits': cons_fits,
        'baseline_status': base_res['status'],
        'baseline_reason': base_res.get('reason', ''),
        'new_status': new_res['status'],
        'new_reason': new_res.get('reason', ''),
        'new_selected_support': new_meta.get('selected_support'),
        'new_selected_spacing_median': new_meta.get('selected_spacing_median'),
        'new_spacing_prior_px': new_meta.get('spacing_prior_px'),
    }


def support_valid_mask(fits, ref_x):
    """Reference columns covered by the selected pair, fixed denominator."""
    if fits is None:
        return None
    xs = np.asarray(ref_x, dtype=float)
    paired = np.asarray(fits['paired_x'], dtype=float)
    lo, hi = fits['support']
    if len(paired):
        d = np.min(np.abs(paired[None, :] - xs[:, None]), axis=1)
    else:
        d = np.full(len(xs), np.inf)
    return (xs >= lo) & (xs <= hi) & (d <= float(fits['max_sample_gap']))


def error_at_reference(fits, ref_x, ref_sup, ref_deep):
    if fits is None:
        return None
    xs = np.asarray(ref_x, dtype=float)
    ys = np.polyval(fits['sup'], xs)
    yd = np.polyval(fits['deep'], xs)
    se = np.abs(ys - np.asarray(ref_sup, dtype=float))
    de = np.abs(yd - np.asarray(ref_deep, dtype=float))
    valid = support_valid_mask(fits, ref_x)
    sv = valid.any()
    return {
        'se': se, 'de': de, 'valid_mask': valid,
        'old_sup_error_mean_px': float(se.mean()),
        'old_deep_error_mean_px': float(de.mean()),
        'old_combined_error_mean_px': float(((se + de) / 2).mean()),
        'sv_sup_error_mean_px': float(se[valid].mean()) if sv else None,
        'sv_deep_error_mean_px': float(de[valid].mean()) if sv else None,
        'sv_combined_error_mean_px': float(((se + de)[valid] / 2).mean()) if sv else None,
        'support_valid_fraction': float(valid.mean()),
        'n_support_valid_columns': int(valid.sum()),
        'n_reference_columns': int(len(xs)),
    }


def correctness_old(err, tol=PRIMARY_TOL_PX,
                    coverage_threshold=PRIMARY_COVERAGE_THRESHOLD):
    if err is None:
        return {'correct': False, 'sup_coverage': None, 'deep_coverage': None}
    sup_cov = float(np.mean(err['se'] <= tol))
    deep_cov = float(np.mean(err['de'] <= tol))
    return {'correct': bool(sup_cov >= coverage_threshold and deep_cov >= coverage_threshold),
            'sup_coverage': sup_cov, 'deep_coverage': deep_cov}


def correctness_support_valid(err, tol=PRIMARY_TOL_PX,
                              coverage_threshold=PRIMARY_COVERAGE_THRESHOLD):
    """Outside-support / gap columns stay in the denominator as uncovered."""
    if err is None:
        return {'correct': False, 'sup_coverage': None, 'deep_coverage': None}
    valid = err['valid_mask']
    n = len(err['se'])
    sup_cov = float(np.sum((err['se'] <= tol) & valid) / n)
    deep_cov = float(np.sum((err['de'] <= tol) & valid) / n)
    return {'correct': bool(sup_cov >= coverage_threshold and deep_cov >= coverage_threshold),
            'sup_coverage': sup_cov, 'deep_coverage': deep_cov}


def scoring_protocol():
    """Canonical scoring protocol object; hash it for the version record."""
    return {
        'version': 'apo_support_valid_v1',
        'reference_subset': 'fixed 143 images, reference columns frozen before scoring',
        'tolerance_px': PRIMARY_TOL_PX,
        'coverage_threshold': PRIMARY_COVERAGE_THRESHOLD,
        'support_valid_primary': ('covered only inside selected-pair support and within '
                                  'max_sample_gap of a valid paired column; outside/'
                                  'gap columns count as uncovered; denominator fixed'),
        'old_metric_retained': 'all reference columns, including extrapolation',
        'pipeline_must_output_candidate': True,
        'failures_count_as_not_correct': True,
    }


def scoring_protocol_sha256():
    payload = json.dumps(scoring_protocol(), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def code_versions():
    """Hashes of every code/config component in the frozen decision path."""
    paths = {
        'geometry': ROOT / 'src/umud_baseline/geometry.py',
        'pair_selection': ROOT / 'scripts/pair_selection.py',
        'apo_baseline_module': ROOT / 'src/umud_baseline/apo_baseline.py',
        'freeze_entry_script': ROOT / 'scripts/freeze_apo_baseline.py',
        'splits': ROOT / 'manifest/splits_v2.csv',
        'manifest': ROOT / 'manifest/manifest.csv',
    }
    return {name: {'path': str(p.relative_to(ROOT)), 'sha256': sha256_file(p)}
            for name, p in paths.items() if p.exists()}
