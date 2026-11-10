"""Fascicle candidate extraction and human-reference matching.

Pre-registered minimal geometry:
  * Connected components (8-connectivity) on the binary fascicle mask.
  * Component PCA; direction = principal axis, normalized to [0,180).
  * Evaluable segment: area >= 20 px, length >= 30 px, width <= 8 px,
    linearity = lambda1/(lambda1+lambda2) >= 0.90, elongation >= 3.
  * Otherwise classified as short_segment / crossing_or_branched.
  * Fragmented neighbours: evaluable segments with angle diff <= 5 deg,
    perpendicular midpoint offset <= 5 px, along-direction gap <= 20 px.
    They are flagged, not merged; no human mask enters prediction extraction.

Matching predicted -> human:
  * angle difference <= angle_tol, perpendicular distance from predicted
    midpoint to the human infinite line <= perp_tol, and overlap along the
    human direction >= overlap_min.
  * Unmatched predicted segments are reported as extra_candidates, not
    automatically labelled false positives.
"""
import math

import numpy as np
from scipy import ndimage


def _norm_angle(deg):
    deg = deg % 180.0
    return deg


def _line_from_segment(seg):
    """Return (a,b,c) for ax+by+c=0 through the segment, unit normal."""
    x0, y0 = seg['p0']
    x1, y1 = seg['p1']
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    # Normal (-dy, dx)
    a, b = -dy / norm, dx / norm
    c = -(a * x0 + b * y0)
    return a, b, c


def extract_segments(mask, min_area=20, min_length=30, max_width=8,
                     min_linearity=0.90, min_elongation=3.0,
                     fragment_angle_tol=5.0, fragment_perp_tol=5.0,
                     fragment_gap=20.0):
    mask = np.asarray(mask, dtype=bool)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    segments = []
    rejected = []
    for lab in range(1, n + 1):
        ys, xs = np.nonzero(labels == lab)
        area = len(xs)
        if area == 0:
            continue
        pts = np.stack([xs, ys], axis=1).astype(float)  # (x,y)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = centered.T @ centered / max(area - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        principal = eigvecs[:, 0]
        normal = eigvecs[:, 1]
        proj = centered @ principal
        perp = centered @ normal
        length = float(proj.max() - proj.min())
        width = float(perp.max() - perp.min())
        i_min, i_max = int(np.argmin(proj)), int(np.argmax(proj))
        p0 = tuple(float(v) for v in pts[i_min])
        p1 = tuple(float(v) for v in pts[i_max])
        direction = _norm_angle(math.degrees(math.atan2(principal[1], principal[0])))
        lam1, lam2 = float(eigvals[0]), float(eigvals[1])
        linearity = lam1 / (lam1 + lam2) if (lam1 + lam2) > 0 else 1.0
        elongation = length / max(width, 1.0)
        rec = {
            'label_id': int(lab),
            'area_px': int(area),
            'centroid': [float(centroid[0]), float(centroid[1])],
            'p0': list(p0), 'p1': list(p1),
            'length_px': length,
            'width_px': width,
            'direction_deg': direction,
            'linearity': linearity,
            'elongation': elongation,
        }
        if area < min_area or length < min_length:
            rec['category'] = 'short_segment'
            rejected.append(rec)
        elif width > max_width or linearity < min_linearity or elongation < min_elongation:
            rec['category'] = 'crossing_or_branched'
            rejected.append(rec)
        else:
            rec['category'] = 'evaluable'
            rec['fragmented_neighbor'] = False
            segments.append(rec)
    # Flag fragmented neighbours among evaluable segments; do not merge.
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a, b = segments[i], segments[j]
            ad = abs(a['direction_deg'] - b['direction_deg'])
            ad = min(ad, 180.0 - ad)
            if ad > fragment_angle_tol:
                continue
            line = _line_from_segment(a)
            if line is None:
                continue
            ax, ay, ac = line
            bx, by = b['centroid']
            perp = abs(ax * bx + by * ay + ac)
            if perp > fragment_perp_tol:
                continue
            # Gap along a's direction between the nearest endpoints.
            d = np.array([a['p1'][0] - a['p0'][0], a['p1'][1] - a['p0'][1]])
            norm = np.linalg.norm(d)
            if norm == 0:
                continue
            d = d / norm
            b0 = np.array(b['p0']) - np.array(a['p0'])
            b1 = np.array(b['p1']) - np.array(a['p0'])
            t0, t1 = float(b0 @ d), float(b1 @ d)
            t_lo, t_hi = sorted([t0, t1])
            gap = max(0.0, t_lo - norm, -t_hi)
            if gap <= fragment_gap:
                a['fragmented_neighbor'] = True
                b['fragmented_neighbor'] = True
    return {'segments': segments, 'rejected': rejected,
            'n_components': int(n)}


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _fragments_related(a, b, angle_tol=5.0, perp_tol=5.0, gap_tol=20.0):
    ad = abs(a['direction_deg'] - b['direction_deg'])
    ad = min(ad, 180.0 - ad)
    if ad > angle_tol:
        return False
    line = _line_from_segment(a)
    if line is None:
        return False
    ax, ay, ac = line
    bx, by = b['centroid']
    perp = abs(ax * bx + by * ay + ac)
    if perp > perp_tol:
        return False
    d = np.array([a['p1'][0] - a['p0'][0], a['p1'][1] - a['p0'][1]], dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0:
        return False
    d /= norm
    b0 = np.array(b['p0']) - np.array(a['p0'])
    b1 = np.array(b['p1']) - np.array(a['p0'])
    t0, t1 = float(b0 @ d), float(b1 @ d)
    t_lo, t_hi = sorted([t0, t1])
    gap = max(0.0, t_lo - norm, -t_hi)
    return gap <= gap_tol


def build_tracks(segments, angle_tol=5.0, perp_tol=5.0, gap_tol=20.0):
    """Group collinear fragments into reference tracks; do not alter masks.

    A track keeps the original segment indices, its combined endpoints,
    direction, length and a `fragmented` flag. Tracks are used as the primary
    matching units so that a broken human annotation is not counted as many
    independent misses.
    """
    n = len(segments)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _fragments_related(segments[i], segments[j], angle_tol, perp_tol, gap_tol):
                uf.union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    tracks = []
    for gid, idxs in enumerate(sorted(groups.values(), key=lambda v: v[0])):
        segs = [segments[i] for i in idxs]
        # Circular mean direction (mod 180).
        ang = np.radians([s['direction_deg'] for s in segs])
        mean_dir = (math.degrees(math.atan2(np.mean(np.sin(2 * ang)),
                                            np.mean(np.cos(2 * ang)))) / 2.0) % 180.0
        d = np.array([math.cos(math.radians(mean_dir)),
                      math.sin(math.radians(mean_dir))])
        pts = []
        for s in segs:
            pts.append(np.array(s['p0'], dtype=float))
            pts.append(np.array(s['p1'], dtype=float))
        pts = np.stack(pts, axis=0)
        proj = pts @ d
        p0 = pts[int(np.argmin(proj))].tolist()
        p1 = pts[int(np.argmax(proj))].tolist()
        centroid = np.mean([s['centroid'] for s in segs], axis=0).tolist()
        tracks.append({
            'track_id': gid,
            'segment_indices': idxs,
            'n_fragments': len(idxs),
            'fragmented': len(idxs) > 1,
            'p0': p0, 'p1': p1,
            'centroid': centroid,
            'direction_deg': float(mean_dir),
            'length_px': float(proj.max() - proj.min()),
            'width_px': float(max(s['width_px'] for s in segs)),
            'area_px': int(sum(s['area_px'] for s in segs)),
        })
    return tracks


def _overlap_fraction(pred_seg, human_seg):
    """Overlap of the predicted segment projected on the human segment."""
    hx0, hy0 = human_seg['p0']
    hx1, hy1 = human_seg['p1']
    d = np.array([hx1 - hx0, hy1 - hy0], dtype=float)
    L = float(np.linalg.norm(d))
    if L == 0:
        return 0.0
    d = d / L
    def proj(pt):
        return float((np.array(pt, dtype=float) - np.array([hx0, hy0])) @ d)
    h_lo, h_hi = 0.0, L
    p_lo, p_hi = sorted([proj(pred_seg['p0']), proj(pred_seg['p1'])])
    overlap = max(0.0, min(h_hi, p_hi) - max(h_lo, p_lo))
    return overlap / L


def match_segments(pred_segments, human_segments, perp_tol=10.0,
                   angle_tol=10.0, overlap_min=0.2):
    matches = []
    used_pred = set()
    for hi, h in enumerate(human_segments):
        hx, hy = h['centroid']
        best = None
        for pi, p in enumerate(pred_segments):
            if pi in used_pred:
                continue
            line = _line_from_segment(h)
            if line is None:
                continue
            a, b, c = line
            px, py = p['centroid']
            perp = abs(a * px + b * py + c)
            ad = abs(p['direction_deg'] - h['direction_deg'])
            ad = min(ad, 180.0 - ad)
            overlap = _overlap_fraction(p, h)
            mid = math.hypot(px - hx, py - hy)
            if perp <= perp_tol and ad <= angle_tol and overlap >= overlap_min:
                score = perp / max(perp_tol, 1e-6) + ad / max(angle_tol, 1e-6)
                if best is None or score < best['score']:
                    best = {
                        'human_index': hi, 'pred_index': pi, 'score': score,
                        'perp_distance_px': perp, 'midpoint_distance_px': mid,
                        'angle_diff_deg': ad, 'overlap_fraction': overlap,
                    }
        if best is not None:
            used_pred.add(best['pred_index'])
            matches.append(best)
    matched_human = {m['human_index'] for m in matches}
    unmatched_human = [i for i in range(len(human_segments)) if i not in matched_human]
    unmatched_pred = [i for i in range(len(pred_segments)) if i not in used_pred]
    return {
        'matches': matches,
        'unmatched_human': unmatched_human,
        'unmatched_pred': unmatched_pred,
    }


def summarize_match(pred_segments, human_segments, match):
    n_h = len(human_segments)
    n_p = len(pred_segments)
    perps = [m['perp_distance_px'] for m in match['matches']]
    mids = [m['midpoint_distance_px'] for m in match['matches']]
    angles = [m['angle_diff_deg'] for m in match['matches']]
    return {
        'n_human_evaluable': n_h,
        'n_pred_evaluable': n_p,
        'n_matched': len(match['matches']),
        'coverage': (len(match['matches']) / n_h) if n_h else None,
        'failure_rate_human': (1 - len(match['matches']) / n_h) if n_h else None,
        'n_unmatched_human': len(match['unmatched_human']),
        'n_unmatched_pred_extra_candidates': len(match['unmatched_pred']),
        'extra_candidate_rate': (len(match['unmatched_pred']) / n_p) if n_p else None,
        'position_perp_mean_px': float(np.mean(perps)) if perps else None,
        'position_perp_median_px': float(np.median(perps)) if perps else None,
        'position_mid_mean_px': float(np.mean(mids)) if mids else None,
        'position_mid_median_px': float(np.median(mids)) if mids else None,
        'direction_mean_deg': float(np.mean(angles)) if angles else None,
        'direction_median_deg': float(np.median(angles)) if angles else None,
        'position_perp_p90_px': float(np.percentile(perps, 90)) if perps else None,
        'direction_p90_deg': float(np.percentile(angles, 90)) if angles else None,
        # Explicit direction-sample counts: direction is defined only on
        # matched predicted->human pairs; it is not imputed for unmatched data.
        'n_direction_pairs': len(angles),
        'direction_defined': bool(angles),
        'n_position_pairs': len(perps),
        'position_defined': bool(perps),
    }


def aggregate_variants(variant_summaries):
    """Mean variant metrics -> image-level metrics (only evaluable denominators)."""
    keys = [
        'coverage', 'failure_rate_human', 'position_perp_mean_px',
        'position_perp_median_px', 'position_mid_mean_px', 'position_mid_median_px',
        'direction_mean_deg', 'direction_median_deg', 'extra_candidate_rate',
        'position_perp_p90_px', 'direction_p90_deg',
    ]
    out = {}
    for k in keys:
        vals = [v[k] for v in variant_summaries if v.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    out['n_variants'] = len(variant_summaries)
    out['n_human_evaluable_sum_variants'] = int(sum(v['n_human_evaluable'] for v in variant_summaries))
    out['n_pred_evaluable_sum_variants'] = int(sum(v['n_pred_evaluable'] for v in variant_summaries))
    out['n_matched_sum_variants'] = int(sum(v['n_matched'] for v in variant_summaries))
    out['n_unmatched_pred_sum_variants'] = int(sum(v['n_unmatched_pred_extra_candidates'] for v in variant_summaries))
    out['n_direction_pairs_sum_variants'] = int(sum(
        v.get('n_direction_pairs', 0) for v in variant_summaries))
    out['n_variants_direction_defined'] = int(sum(
        bool(v.get('direction_defined')) for v in variant_summaries))
    return out
