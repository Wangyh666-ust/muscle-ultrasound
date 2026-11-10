"""Label-free boundary path reconstruction. Coordinates are native crop pixels.

Only image, probability, evidence domain, frozen C1 observations and an optional
verified pixel/mm scale enter this module. Manual boundaries are never accepted.
Original paths survive every fusion; bridges and extrapolation are not observations.
"""
from copy import deepcopy
from itertools import product

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, map_coordinates
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from scripts.round63_common import defaults


def sample(field, x, y, radius=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not radius:
        return map_coordinates(field, [y, x], order=1, mode='constant', cval=0.)
    return np.max([map_coordinates(field, [y+d, x], order=1, mode='constant', cval=0.)
                   for d in range(-radius, radius+1)], axis=0)


def image_evidence(image, width):
    """Local bright-ridge contrast, used only inside qualified geometric gaps."""
    a = np.asarray(image, np.float32)
    fine = gaussian_filter(a, (max(.6, .001*width), max(.6, .001*width)))
    wide = gaussian_filter(a, (max(2., .012*width), max(2., .006*width)))
    var = np.maximum(gaussian_filter(a*a, max(2., .01*width))-wide*wide, 4.)
    return np.clip(.5+(fine-wide)/(2*np.sqrt(var)), 0., 1.)


def tangent(path, right, width, cfg):
    x, y = path['x'], path['y']
    at = -1 if right else 0
    use = abs(x-x[at]) <= max(5., cfg['endpoint_window']*width)
    if use.sum() < 3:
        use = np.zeros(len(x), bool)
        use[-min(5, len(x)):] = True if right else False
        if not right:
            use[:min(5, len(x))] = True
    dx, yy = x[use]-x[at], y[use]
    if len(dx) < 2 or np.ptp(dx) < 1:
        return 0.
    a = np.column_stack([dx, np.ones(len(dx))])
    weights = np.sqrt(np.maximum(path['confidence'][use], .05))
    coef = np.linalg.lstsq(a*weights[:, None], yy*weights, rcond=None)[0]
    for _ in range(3):
        residual = yy-a@coef
        mad = max(.5*width/512., 1.4826*np.median(abs(residual-np.median(residual))))
        w = weights*np.sqrt(np.minimum(1., 1.5*mad/np.maximum(abs(residual), 1e-6)))
        coef = np.linalg.lstsq(a*w[:, None], yy*w, rcond=None)[0]
    return float(np.clip(coef[0], -2., 2.))


def hermite(x, x0, y0, m0, x1, y1, m1):
    """Position/tangent-continuous bridge; derivatives are with respect to x."""
    length = x1-x0
    t = (np.asarray(x)-x0)/length
    y = (2*t**3-3*t**2+1)*y0 + (t**3-2*t**2+t)*length*m0
    y += (-2*t**3+3*t**2)*y1 + (t**3-t**2)*length*m1
    d = (6*t*t-6*t)*y0/length + (3*t*t-4*t+1)*m0
    d += (-6*t*t+6*t)*y1/length + (3*t*t-2*t)*m1
    dd = (12*t-6)*y0/length**2 + (6*t-4)*m0/length
    dd += (-12*t+6)*y1/length**2 + (6*t-2)*m1/length
    return y, d, dd


def refine_bridge(x, y, dy, ddy, probability, gray, width, cfg):
    """Follow nearby evidence with a correction whose value/tangent vanish at ends."""
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import gaussian_filter1d
    length = x[-1]-x[0]
    t = (x-x[0])/length
    radius = max(2, round(min(.012, cfg['near_distance']/2)*width))
    offsets = np.arange(-radius, radius+1, dtype=float)
    heights = y[:, None]+offsets[None]
    xx = np.repeat(x[:, None], len(offsets), axis=1)
    q = sample(probability, xx.ravel(), heights.ravel()).reshape(heights.shape)
    g = sample(gray, xx.ravel(), heights.ravel()).reshape(heights.shape)
    score = q+.35*g-.02*(offsets[None]/radius)**2
    best = score.argmax(1)
    signal = q+.35*g
    targets = offsets[best].copy()
    # Centre one local ridge rather than snap to the nearest edge of a flat band.
    # Distinct peaks are never averaged across the intervening valley.
    for i, peak in enumerate(best):
        threshold = signal[i].min()+.5*np.ptp(signal[i])
        lo = hi = int(peak)
        while lo > 0 and signal[i, lo-1] > threshold:
            lo -= 1
        while hi < len(offsets)-1 and signal[i, hi+1] > threshold:
            hi += 1
        weights = np.maximum(signal[i, lo:hi+1]-signal[i].min(), 0.)
        if weights.sum() > 0:
            targets[i] = float(weights@offsets[lo:hi+1]/weights.sum())
    informative = (np.ptp(q+.35*g, axis=1) >= .04) & (t >= .08) & (t <= .92)
    if informative.sum() < 5:
        return y, dy, ddy, False
    positions = np.linspace(.125, .875, 7)
    centers = gaussian_filter1d(y+targets, sigma=max(.7, len(x)/70), mode='nearest')
    nodes_x = np.r_[x[0], x[0]+positions*length, x[-1]]
    nodes_y = np.r_[y[0], np.interp(nodes_x[1:-1], x[informative], centers[informative]), y[-1]]
    spline = CubicSpline(nodes_x, nodes_y, bc_type=((1, dy[0]), (1, dy[-1])), extrapolate=False)
    revised = spline(x)
    delta = revised-y
    if abs(delta).max() > radius:
        return y, dy, ddy, False
    old = sample(probability, x, y)+.35*sample(gray, x, y)
    new = sample(probability, x, revised)+.35*sample(gray, x, revised)
    if float(np.mean(new-old)) < -.002:
        return y, dy, ddy, False
    return revised, spline(x, 1), spline(x, 2), True


def pack_path(x, y, confidence, observed, ids, kind=None, events=None):
    order = np.argsort(x, kind='stable')
    x, y, confidence, observed = [np.asarray(a)[order] for a in (x, y, confidence, observed)]
    assert len(x) > 1 and np.all(np.diff(x) > 0)
    return dict(x=x.astype(float), y=y.astype(float), confidence=confidence.astype(float),
                observed=observed.astype(bool), kind=(np.where(observed, 0, 1) if kind is None else np.asarray(kind)[order]).astype(int),
                ids=tuple(sorted(ids)), events=list(events or []))


def extract_paths(prob, valid, width, cfg, hard=False):
    """Track per-column ridges one-to-one. No connected-component branch averaging."""
    floor = .5 if hard else cfg['low']
    q = gaussian_filter(np.asarray(prob, float), (max(.45, width/1046.), 0.))
    q *= valid
    tracks, active = [], []
    max_step = max(1.25, cfg['point_step']*width)
    for x in np.flatnonzero(valid.any(0)):
        yy, _ = find_peaks(q[:, x], height=floor, prominence=.025,
                           distance=max(1, round(.003*width)))
        yy = yy[valid[yy, x]]
        if len(yy) > 20:
            yy = yy[np.argsort(-q[yy, x], kind='stable')[:20]]
        active = [i for i in active if x-tracks[i][-1][0] <= cfg['max_column_gap']+1]
        used = set()
        if active and len(yy):
            cost = np.full((len(active), len(yy)), 1e6)
            for k, i in enumerate(active):
                points = tracks[i]
                last_x, last_y, _ = points[-1]
                slope = 0. if len(points) < 4 else np.clip((last_y-points[-4][1])/(last_x-points[-4][0]), -1.5, 1.5)
                pred = last_y+slope*(x-last_x)
                distance = abs(yy-pred)
                cost[k] = np.where(distance <= max_step*(x-last_x), distance/max_step-.05*q[yy, x], 1e6)
            a, b = linear_sum_assignment(cost)
            for k, n in zip(a, b):
                if cost[k, n] < 1e5:
                    tracks[active[k]].append((x, float(yy[n]), float(prob[yy[n], x])))
                    used.add(n)
        for n, y in enumerate(yy):
            if n not in used:
                tracks.append([(x, float(y), float(prob[y, x]))])
                active.append(len(tracks)-1)
    paths = []
    for i, points in enumerate(tracks):
        if len(points) < max(5, round(cfg['minimum_track']*width)):
            continue
        x, y, p = np.asarray(points).T
        if np.count_nonzero(p >= cfg['high']) < max(2, round(cfg['minimum_anchor']*width)):
            continue
        # Small robust smoothing of ridge quantization, never a global polynomial.
        size = max(3, 2*round(.003*width)+1)
        smooth = median_filter(y, size=size, mode='nearest')
        y = np.clip(smooth, y-width/1046., y+width/1046.)
        grid = np.arange(x[0], x[-1]+1)
        paths.append(pack_path(grid, np.interp(grid, x, y), np.interp(grid, x, p),
                               np.isin(grid, x), [i]))
    return sorted(paths, key=lambda p: (-len(p['x'])*float(p['confidence'].mean()), p['ids']))[:cfg['max_paths']]


def bridge(a, b, probability, gray, valid, width, cfg):
    """Return a qualified hypothesis; never mutate either observed source path."""
    if set(a['ids']) & set(b['ids']) or len(set(a['ids']+b['ids'])) > cfg['max_chain']:
        return None, 'reused_source'
    if b['x'][0] <= a['x'][0] or b['x'][-1] <= a['x'][-1]+.005*width:
        return None, 'not_a_continuation'
    gap = b['x'][0]-a['x'][-1]
    limit = cfg['near_distance']*width
    events = a['events']+b['events']
    if gap <= 0:
        lo, hi = b['x'][0], a['x'][-1]
        xx = np.arange(np.ceil(lo), np.floor(hi)+1)
        if len(xx) < max(3, .01*width):
            return None, 'insufficient_overlap'
        ya, yb = [np.interp(xx, p['x'], p['y']) for p in (a, b)]
        separation = abs(ya-yb)
        if separation.max() > limit:
            return None, 'different_layer_distance'
        da, db = [np.gradient(y) for y in (ya, yb)]
        if np.quantile(abs(np.arctan(da)-np.arctan(db)), .9) > np.deg2rad(25):
            return None, 'overlap_different_trend'
        xs = np.unique(np.r_[a['x'], b['x']])
        ys, ps, obs = np.zeros(len(xs)), np.zeros(len(xs)), np.zeros(len(xs), bool)
        den = np.zeros(len(xs))
        for p in (a, b):
            inside = (xs >= p['x'][0]) & (xs <= p['x'][-1])
            px = xs[inside]
            w = np.interp(px, p['x'], p['confidence'])
            ys[inside] += np.interp(px, p['x'], p['y'])*np.maximum(w, .05)
            den[inside] += np.maximum(w, .05)
            ps[inside] = np.maximum(ps[inside], w)
            indices = np.searchsorted(xs, p['x'][p['observed']])
            obs[indices] = True
        ys /= den
        # Averages in empty space are not measured evidence or automatically safe.
        support = sample(probability, xs, ys, max(1, round(width/1046.)))
        original = np.maximum(np.interp(xs, a['x'], a['confidence'], left=0, right=0),
                              np.interp(xs, b['x'], b['confidence'], left=0, right=0))
        overlap = (xs >= lo) & (xs <= hi)
        if np.mean(support[overlap] >= .35*original[overlap]) < .8:
            return None, 'fusion_between_probability_ridges'
        obs &= support >= cfg['low']
        events.append(dict(type='near_overlap', maximum_separation=float(separation.max()/width),
                           interval=[float(lo), float(hi)]))
        return pack_path(xs, ys, support, obs, a['ids']+b['ids'], events=events), None
    if gap > cfg['max_gap']*width:
        return None, 'gap_too_long'
    if gap < 1:
        return None, 'subpixel_gap'
    x0, x1 = a['x'][-1], b['x'][0]
    y0, y1 = a['y'][-1], b['y'][0]
    m0, m1 = tangent(a, True, width, cfg), tangent(b, False, width, cfg)
    # This is a distance to the endpoint trends, not an equal-angle requirement.
    normal_error = max(abs(y1-y0-m0*gap)/np.sqrt(1+m0*m0),
                       abs(y0-y1+m1*gap)/np.sqrt(1+m1*m1))
    if normal_error > limit:
        return None, 'different_layer_distance'
    xx = np.linspace(x0, x1, max(9, int(np.ceil(gap))+1))
    base_y, base_dy, base_ddy = hermite(xx, x0, y0, m0, x1, y1, m1)
    refined_y, refined_dy, refined_ddy, changed = refine_bridge(xx, base_y, base_dy, base_ddy, probability, gray, width, cfg)
    chord = y0+(y1-y0)*(xx-x0)/gap
    # Account for bounded endpoint pixel quantization, which scales as 1/gap^2.
    # The uncertainty shrinks with image scale; this is not a larger global angle gate.
    curvature_budget = cfg['max_curvature']+12*width*(2*width/1046.)/gap**2
    variants = []
    for alpha in ((0., .25, .5, 1.) if changed else (0.,)):
        yy = base_y+alpha*(refined_y-base_y)
        dy = base_dy+alpha*(refined_dy-base_dy)
        ddy = base_ddy+alpha*(refined_ddy-base_ddy)
        curvature = width*abs(ddy)/(1+dy*dy)**1.5
        if curvature.max() > curvature_budget or np.ptp(np.arctan(dy)) > np.deg2rad(cfg['max_turn_deg']):
            continue
        if abs(yy-chord).max() > cfg['maximum_bulge']*width:
            continue
        quality = float(np.mean(sample(probability, xx, yy)+.35*sample(gray, xx, yy)))
        # On a flat-topped ridge prefer its locally estimated centre, not its edge.
        variants.append((quality+.001*alpha, yy, dy, ddy, curvature, alpha))
    if not variants:
        return None, 'abrupt_bridge'
    _, yy, dy, ddy, curvature, refined = max(variants, key=lambda z: z[0])
    if np.any(sample(valid.astype(float), xx, yy) < .99):
        return None, 'outside_evidence'
    radius = max(1, round(.003*width))
    interior = (xx >= x0+.15*gap) & (xx <= x1-.15*gap)
    evidence = sample(probability, xx[interior], yy[interior], radius)
    photometric = sample(gray, xx[interior], yy[interior], radius)
    hits = float(np.mean(evidence >= cfg['corridor_probability']))
    bright = float(np.mean(photometric >= .60))
    if hits < cfg['corridor_hit_fraction'] and bright < cfg['image_hit_fraction']:
        return None, 'no_bridge_evidence'
    integer_x = np.arange(int(np.floor(x0))+1, int(np.ceil(x1)))
    integer_y = np.interp(integer_x, xx, yy)
    confidence = sample(probability, integer_x, integer_y, radius)
    events.append(dict(type='hermite_gap', interval=[float(x0), float(x1)],
                       normalized_curvature=float(curvature.max()), probability_hits=hits,
                       image_hits=bright, endpoint_slopes=[m0, m1], evidence_refined=refined,
                       curvature_budget=float(curvature_budget)))
    return pack_path(np.r_[a['x'], integer_x, b['x']], np.r_[a['y'], integer_y, b['y']],
                     np.r_[a['confidence'], confidence, b['confidence']],
                     np.r_[a['observed'], np.zeros(len(integer_x), bool), b['observed']],
                     a['ids']+b['ids'], events=events), None


def path_quality(path, width):
    observed = float(path['observed'].sum()/width)
    span = float(np.ptp(path['x'])/width)
    probability = float(np.mean(path['confidence']))
    imputed = float((~path['observed']).sum()/len(path['x']))
    return .45*probability+.30*observed+.25*span-.10*imputed


def rebuild(paths, probability, gray, valid, width, cfg):
    pool = list(paths)
    seen = {p['ids'] for p in paths}
    audit = dict(accepted=[], rejected={})
    frontier = list(paths)
    for _ in range(cfg['max_chain']-1):
        added = []
        for a, b in product(frontier, paths):
            if tuple(sorted(a['ids']+b['ids'])) in seen:
                continue
            p, reason = bridge(a, b, probability, gray, valid, width, cfg)
            if p is None:
                audit['rejected'][reason] = audit['rejected'].get(reason, 0)+1
            else:
                seen.add(p['ids'])
                added.append(p)
                audit['accepted'].append(dict(ids=p['ids'], events=p['events']))
        if not added:
            break
        frontier = sorted(added, key=lambda p: -path_quality(p, width))[:cfg['max_paths']]
        pool.extend(frontier)
    # Keep original candidates even when a smoother/longer hypothesis exists.
    return paths+sorted(pool[len(paths):], key=lambda p: -path_quality(p, width))[:cfg['max_paths']], audit


def containment(upper, lower, segments, bounds):
    votes = [[], [], []]
    left, right = bounds
    for s in segments:
        a, b = np.asarray(s['p0']), np.asarray(s['p1'])
        xy = a[None]+np.linspace(0, 1, 7)[:, None]*(b-a)[None]
        common = (xy[:, 0] >= max(upper['x'][0], lower['x'][0])) & (xy[:, 0] <= min(upper['x'][-1], lower['x'][-1]))
        if common.sum() < 3:
            continue
        x, y = xy[common].T
        uy, ly = [np.interp(x, p['x'], p['y']) for p in (upper, lower)]
        vote = float(np.mean((y >= uy) & (y <= ly)))
        region = min(2, max(0, int(3*(s['centroid'][0]-left)/max(right-left, 1))))
        votes[region].append(vote)
    active = [i for i, v in enumerate(votes) if v]
    n = sum(map(len, votes))
    if n < 3 or len(active) < 2:
        return 0., False, n
    weights = np.array([.25, .5, .25])[active]
    means = [np.mean(sorted(votes[i], reverse=True)[:max(1, int(np.ceil(.8*len(votes[i]))))]) for i in active]
    return float(weights@means/weights.sum()), True, n


def choose_pair(paths, width, cfg, segments=(), px_per_mm=None, use_c1=False, use_prior=False):
    pairs = []
    for u, d in product(*paths):
        lo, hi = max(u['x'][0], d['x'][0]), min(u['x'][-1], d['x'][-1])
        if hi-lo < cfg['min_pair_span']*width:
            continue
        x = np.linspace(lo, hi, max(25, round((hi-lo)/2)))
        y0, y1 = [np.interp(x, p['x'], p['y']) for p in (u, d)]
        if np.any(y1-y0 <= cfg['min_thickness']*width):
            continue
        mt = float(np.mean(y1-y0))
        score = .5*(path_quality(u, width)+path_quality(d, width))+.10*(hi-lo)/width
        c1, c1_active, count = containment(u, d, segments, [lo, hi]) if use_c1 else (0., False, 0)
        if c1_active:
            score += .20*(c1-.5)
        penalty, mt_mm = 0., None
        if px_per_mm is not None and np.isfinite(px_per_mm) and px_per_mm > 0:
            mt_mm = mt/px_per_mm
            if use_prior:
                low, high = cfg['prior_flat_mm']
                distance = max(low-mt_mm, mt_mm-high, 0.)
                penalty = cfg['prior_maximum_penalty']*(1-np.exp(-.5*(distance/cfg['prior_shoulder_mm'])**2))
                score -= penalty
        pairs.append(dict(upper=u, lower=d, score=float(score), interval=[float(lo), float(hi)],
                          MT_px=mt, MT_mm=mt_mm, c1=c1, c1_active=c1_active,
                          c1_tracks=count, prior_penalty=float(penalty), prior_enabled=bool(use_prior and mt_mm is not None)))
    pairs.sort(key=lambda p: (-p['score'], -np.ptp(p['upper']['x'])-np.ptp(p['lower']['x']),
                              p['upper']['ids'], p['lower']['ids']))
    if not pairs:
        return None, []
    selected = deepcopy(pairs[0])
    different = next((p for p in pairs[1:] if abs(p['MT_px']-selected['MT_px']) > .02*width), None)
    selected['margin'] = float(selected['score']-different['score']) if different else None
    selected['ambiguous'] = bool(different and selected['margin'] < cfg['ambiguity_margin'])
    return selected, pairs


def extend(path, probability, gray, valid, width, cfg):
    """Finite continuation with damped curvature; no unbounded polynomial rays."""
    result = deepcopy(path)
    for right in (False, True):
        p = result
        at = -1 if right else 0
        x0, y0 = p['x'][at], p['y'][at]
        direction = 1 if right else -1
        n = int(cfg['extension']*width)
        if n < 1:
            continue
        dx = direction*np.arange(1, n+1, dtype=float)
        slope = tangent(p, right, width, cfg)
        use = abs(p['x']-x0) <= max(6, cfg['endpoint_window']*width)
        curvature = 0.
        if use.sum() >= 6:
            coeff = np.polyfit(p['x'][use]-x0, p['y'][use], 2, w=np.maximum(.1, p['confidence'][use]))
            curvature = float(np.clip(2*coeff[0], -cfg['max_curvature']/width, cfg['max_curvature']/width))
        y = y0+slope*dx+.5*curvature*dx**2/(1+abs(dx)/max(.04*width, 1))
        x = x0+dx
        ok = sample(valid.astype(float), x, y) > .99
        q = sample(probability, x, y, max(1, round(.003*width)))
        g = sample(gray, x, y, max(1, round(.003*width)))
        run, stop = 0, len(x)
        for i in range(len(x)):
            if not ok[i]:
                stop = i
                break
            run = run+1 if q[i] < cfg['corridor_probability'] and g[i] < .60 else 0
            if run > max(1, cfg['extension_blind_run']*width):
                stop = i-run+1
                break
        if stop <= 0:
            continue
        x, y, q = x[:stop], y[:stop], q[:stop]
        result = pack_path(np.r_[p['x'], x], np.r_[p['y'], y], np.r_[p['confidence'], q],
                           np.r_[p['observed'], np.zeros(len(x), bool)], p['ids'],
                           kind=np.r_[p['kind'], np.full(len(x), 2)],
                           events=p['events']+[dict(type='extension', interval=[float(x.min()), float(x.max())])])
    return result


def decode(probability, image, evidence, width, cfg=None, segments=(), px_per_mm=None):
    cfg = cfg or defaults()
    assert probability.shape == (3, *image.shape) and evidence.shape == image.shape
    assert width > 0 and np.isfinite(probability).all()
    valid = evidence.astype(bool)
    # Union helps identify boundary evidence but cannot replace the role-specific head.
    fields = [probability[i]*(cfg['union_floor']+(1-cfg['union_floor'])*probability[0]) for i in (1, 2)]
    hard = [extract_paths(probability[i]*valid, valid, width, cfg, True) for i in (1, 2)]
    weak = [extract_paths(p*valid, valid, width, cfg) for p in fields]
    gray = image_evidence(image, width)
    rebuilt, audits = [], []
    for paths, p in zip(weak, fields):
        q, a = rebuild(paths, p, gray, valid, width, cfg)
        rebuilt.append(q)
        audits.append(a)
    arms, candidates = {}, {}
    for arm, paths, c1, prior in [('T', hard, False, False), ('U', weak, False, False),
                                  ('H', rebuilt, False, False), ('M', rebuilt, True, False),
                                  ('P', rebuilt, True, True)]:
        arms[arm], candidates[arm] = choose_pair(paths, width, cfg, segments, px_per_mm, c1, prior)
    arms['E'] = deepcopy(arms['P'])
    if arms['E']:
        for side, p in zip(('upper', 'lower'), fields):
            arms['E'][side] = extend(arms['E'][side], p, gray, valid, width, cfg)
        u, d = arms['E']['upper'], arms['E']['lower']
        lo, hi = max(u['x'][0], d['x'][0]), min(u['x'][-1], d['x'][-1])
        x = np.linspace(lo, hi, max(25, round(hi-lo)))
        if np.any(np.interp(x, d['x'], d['y'])-np.interp(x, u['x'], u['y']) <= cfg['min_thickness']*width):
            arms['E'] = deepcopy(arms['P'])
            arms['E']['extension_rejected'] = 'order_or_spacing'
        else:
            arms['E']['interval'] = [float(lo), float(hi)]
    return dict(arms=arms, candidates=candidates, tracks=dict(hard=hard, weak=weak, rebuilt=rebuilt), audit=audits)


def compact_path(path):
    return {k: path[k] for k in ('x', 'y', 'observed', 'kind', 'ids', 'events')}


def compact_pair(pair):
    if pair is None:
        return None
    return {**{k: v for k, v in pair.items() if k not in ('upper', 'lower')},
            'upper': compact_path(pair['upper']), 'lower': compact_path(pair['lower'])}
