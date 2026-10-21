"""Target band-pair selection experiment core.

This module keeps the existing per-column run generation and line-fitting
rules. It changes only which two bands are selected when a mask contains more
than two runs. It does not merge nearby runs, change the ROI, change the model
or threshold, or change the geometry fit/residual thresholds.

The rule parameters are frozen in a JSON file before any dev evaluation.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.umud_baseline.geometry import mt_at_positions  # noqa: E402


def run_centres_per_column(mask):
    """Same per-column run rule as geometry.extract_apo_bands."""
    h, w = mask.shape
    out = []
    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if not len(ys):
            out.append([])
            continue
        splits = np.flatnonzero(np.diff(ys) > max(2, round(h * .002))) + 1
        out.append([float(np.median(v)) for v in np.split(ys, splits)])
    return out


def build_tracks(runs, track_tol=8.0, max_gap=3):
    """Track run centres across columns; no run merging.

    Each track stores one point per column at most. A run is assigned to at
    most one track. Tracks may be unmatched for up to max_gap columns.
    """
    tracks = []  # list of {'points': [(x, y), ...]}
    for x, centres in enumerate(runs):
        if not centres:
            continue
        candidates = []
        for ti, track in enumerate(tracks):
            last_x, last_y = track['points'][-1]
            gap = x - last_x
            if gap <= 0 or gap > max_gap:
                continue
            slope = 0.0
            if len(track['points']) >= 2:
                x1, y1 = track['points'][-2]
                x2, y2 = track['points'][-1]
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
            pred = last_y + slope * gap
            for ri, y in enumerate(centres):
                d = abs(y - pred)
                if d <= track_tol * max(1, gap):
                    candidates.append((d, ti, ri))
        candidates.sort()
        used_tracks, used_runs = set(), set()
        for d, ti, ri in candidates:
            if ti in used_tracks or ri in used_runs:
                continue
            tracks[ti]['points'].append((x, centres[ri]))
            used_tracks.add(ti)
            used_runs.add(ri)
        for ri, y in enumerate(centres):
            if ri not in used_runs:
                tracks.append({'points': [(x, y)]})
    return tracks


def select_pair(tracks, width, height, rule):
    """Select two tracks by support first, then spacing prior.

    Candidate pairs must have common support >= max(20, 0.25*width). Among
    pairs whose support is within support_tolerance_fraction of the maximum,
    choose the spacing closest to spacing_prior_fraction * height.
    """
    min_cols = max(rule.get('min_track_columns_abs', 5),
                   rule.get('min_track_columns_fraction', 0.05) * width)
    usable = [t for t in tracks if len(t['points']) >= min_cols]
    if len(usable) < 2:
        # Keep the original geometry's rejection vocabulary; detail is in meta.
        return None, 'insufficient_paired_support', {
            'detail': 'insufficient_band_tracks',
            'n_tracks': len(tracks), 'n_usable': len(usable)}
    pairs = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            xi = {p[0]: p[1] for p in usable[i]['points']}
            xj = {p[0]: p[1] for p in usable[j]['points']}
            common = sorted(set(xi) & set(xj))
            if len(common) < max(20, 0.25 * width):
                continue
            sp = np.asarray([abs(xi[x] - xj[x]) for x in common], dtype=float)
            pairs.append({
                'i': i, 'j': j,
                'common_x': common,
                'support': len(common),
                'spacing_median': float(np.median(sp)),
            })
    if not pairs:
        return None, 'insufficient_paired_support', {
            'detail': 'insufficient_pair_support',
            'n_tracks': len(tracks), 'n_usable': len(usable)}
    max_support = max(p['support'] for p in pairs)
    support_tol = rule.get('support_tolerance_fraction', 0.10)
    near = [p for p in pairs if p['support'] >= max_support * (1 - support_tol)]
    prior = rule.get('spacing_prior_fraction', 0.358) * height
    near.sort(key=lambda p: abs(p['spacing_median'] - prior))
    chosen = near[0]
    # Order the two tracks as superficial (smaller median y) and deep.
    yi = np.median([p[1] for p in usable[chosen['i']]['points']])
    yj = np.median([p[1] for p in usable[chosen['j']]['points']])
    if yi <= yj:
        sup_track, deep_track = usable[chosen['i']], usable[chosen['j']]
    else:
        sup_track, deep_track = usable[chosen['j']], usable[chosen['i']]
    meta = {
        'n_tracks': len(tracks),
        'n_usable_tracks': len(usable),
        'n_pair_candidates': len(pairs),
        'max_support': max_support,
        'selected_support': chosen['support'],
        'selected_spacing_median': chosen['spacing_median'],
        'spacing_prior_px': prior,
        'selected_track_sizes': [len(sup_track['points']), len(deep_track['points'])],
    }
    return (sup_track, deep_track, chosen['common_x']), None, meta


def fit_selected_pair(sup_track, deep_track, common_x, width, height):
    """Use the same fit/residual/support checks as geometry.extract_apo_bands."""
    xs = np.asarray(common_x, dtype=int)
    sup_map = {p[0]: p[1] for p in sup_track['points']}
    deep_map = {p[0]: p[1] for p in deep_track['points']}
    sup = np.asarray([sup_map[x] for x in xs], dtype=float)
    deep = np.asarray([deep_map[x] for x in xs], dtype=float)
    # Same minimum band separation as geometry.extract_apo_bands:
    #   if b-a <= max(2, H*.005): continue
    # Apply it to the selected pair before support/residual checks so the new
    # implementation does not silently accept columns the original would skip.
    keep = (deep - sup) > max(2, .005 * height)
    xs, sup, deep = xs[keep], sup[keep], deep[keep]
    if len(xs) < max(8, .3 * width):
        return None, 'insufficient_paired_support'
    if xs[-1] - xs[0] < .3 * width:
        return None, 'short_common_support'
    fits = {'sup': np.polyfit(xs, sup, 1), 'deep': np.polyfit(xs, deep, 1)}
    residuals = [np.abs(np.polyval(fits[k], xs) - v) for k, v in [('sup', sup), ('deep', deep)]]
    if max(np.quantile(v, .9) for v in residuals) > max(3, .02 * height):
        return None, 'nonlinear_or_unstable_bands'
    gap = np.polyval(fits['deep'] - fits['sup'], [xs[0], xs[-1]])
    if (gap <= 0).any():
        return None, 'crossed_or_inverted_lines'
    fits.update(support=(int(xs[0]), int(xs[-1])), paired_x=xs,
                max_sample_gap=max(2, .015 * width), image_shape=(height, width),
                roi_source='pair_selection_full_image')
    return fits, None


def extract_apo_bands_selected(mask, rule):
    """Return (fits, reason, meta) with only pair selection changed."""
    runs = run_centres_per_column(mask)
    tracks = build_tracks(runs, track_tol=rule.get('track_tol_px', 8.0),
                          max_gap=rule.get('max_gap_columns', 3))
    selected, reason, meta = select_pair(tracks, mask.shape[1], mask.shape[0], rule)
    if reason:
        return None, reason, meta
    sup_track, deep_track, common_x = selected
    fits, reason = fit_selected_pair(sup_track, deep_track, common_x,
                                     mask.shape[1], mask.shape[0])
    meta['selected_support_sup'] = len(sup_track['points'])
    meta['selected_support_deep'] = len(deep_track['points'])
    return fits, reason, meta


def measure_mt_selected(mask, rule):
    """Candidate/rejected result for the selected pair, using the same default
    quartile positions and position checks as geometry.measure_mt."""
    fits, reason, meta = extract_apo_bands_selected(mask, rule)
    if reason:
        return {'status': 'rejected', 'reason': reason}, None, meta
    lo, hi = fits['support']
    xs = [lo + f * (hi - lo) for f in (.25, .5, .75)]
    mt, reason = mt_at_positions(fits, xs)
    if reason:
        return {'status': 'rejected', 'reason': reason}, fits, meta
    return {'status': 'candidate', 'x_positions': list(map(float, xs)),
            'mt_px': mt, 'mt_mean_px': float(np.mean(mt)),
            'support': list(fits['support']), 'position_rule': 'paired_support_quartiles',
            'roi_source': fits['roi_source']}, fits, meta
