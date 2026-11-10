"""Shared geometry diagnostics and reference scoring for the frozen C1 population.

These measurements isolate the new boundary mechanism. They do not impersonate
the default Round40 qualification/selection pipeline or a Kaggle submission.
Finite-range FL and exploratory tangent-ray FL are explicitly separate.
"""
import time
import cv2
import numpy as np
from scripts.round63_common import *
from scripts.round63_paths import decode, compact_pair


def mapped_segments(metadata, rec):
    a = np.asarray(rec['matrix'], float)
    box = np.asarray(rec['input_box'][:2])
    result = []
    for s in metadata['segments']:
        q = dict(s)
        for key in ('p0', 'p1', 'centroid'):
            q[key] = a[:, :2]@np.asarray(s[key])+a[:, 2]-box
        delta = q['p1']-q['p0']
        q['direction_deg'] = float(np.rad2deg(np.arctan2(delta[1], delta[0])) % 180)
        result.append(q)
    return result


def original_pair(pair, rec):
    """Undo both crop translation and native image affine before measuring."""
    if pair is None:
        return None
    q = dict(pair)
    sx, sy = rec['matrix'][0][0], rec['matrix'][1][1]
    tx, ty = rec['matrix'][0][2], rec['matrix'][1][2]
    a, b = rec['input_box'][:2]
    for side in SIDES:
        p = dict(pair[side])
        p['x'] = (p['x']+a-tx)/sx
        p['y'] = (p['y']+b-ty)/sy
        q[side] = p
    q['interval'] = [(x+a-tx)/sx for x in pair['interval']]
    return q


def local_slope(path, at, width):
    x, y = path['x'], path['y']
    end = at <= x[0] or at >= x[-1]
    use = abs(x-at) <= max(4., (.05 if end else .025)*width)
    if use.sum() < 3:
        nearest = np.argsort(abs(x-at))[:min(5, len(x))]
        use = np.zeros(len(x), bool); use[nearest] = True
    a = np.column_stack([x[use]-at, np.ones(use.sum())])
    weights = np.ones(use.sum())
    coefficients = np.linalg.lstsq(a, y[use], rcond=None)[0]
    for _ in range(3):
        residual = y[use]-a@coefficients
        sigma = max(.5*width/1046., 1.4826*np.median(abs(residual-np.median(residual))))
        weights = np.sqrt(np.minimum(1., 1.5*sigma/np.maximum(abs(residual), 1e-8)))
        coefficients = np.linalg.lstsq(a*weights[:, None], y[use]*weights, rcond=None)[0]
    return float(coefficients[0])


def roots(path, m, b, rays=False, width=None):
    x, y = path['x'], path['y']
    residual = y-m*x-b
    crossings = np.flatnonzero(residual[:-1]*residual[1:] <= 0)
    points = []
    for i in crossings:
        slope = (y[i+1]-y[i])/(x[i+1]-x[i])
        if abs(slope-m) < 1e-8:
            continue
        rx = (b-y[i]+slope*x[i])/(slope-m)
        if not any(abs(rx-p[0]) < 1e-5 for p in points):
            kind = int(max(path['kind'][i], path['kind'][i+1]))
            points.append((float(rx), float(m*rx+b), kind))
    if rays:
        for right, at in [(False, 0), (True, -1)]:
            slope = local_slope(path, x[at], width or np.ptp(x))
            if abs(slope-m) < 1e-6:
                continue
            rx = (b-y[at]+slope*x[at])/(slope-m)
            if (right and rx > x[-1]) or (not right and rx < x[0]):
                points.append((float(rx), float(m*rx+b), 3))
    return points


def aggregate(candidates, field):
    active = [i for i in range(3) if any(c['region'] == i and c[field] is not None for c in candidates)]
    if not active:
        return None, {}
    region_weights = np.array([.25, .5, .25])[active]
    region_weights /= region_weights.sum()
    value, weights = 0., {}
    for region, rw in zip(active, region_weights):
        pool = [c for c in candidates if c['region'] == region and c[field] is not None]
        # One vote per frozen observed segment; no change of direction or fragment merging.
        w = rw/len(pool)
        for c in pool:
            value += w*c[field]
            weights[str(c['id'])] = float(w)
    return float(value), weights


def geometry(pair, segments, scale, width=None):
    if pair is None:
        return dict(MT=None, PA=None, FL_finite=None, FL_ray=None, contributors=[], weights={})
    u, d = pair['upper'], pair['lower']
    lo, hi = pair['interval']
    width = width or hi-lo
    xx = lo+np.array([.25, .5, .75])*(hi-lo)
    mt = float(np.mean(np.interp(xx, d['x'], d['y'])-np.interp(xx, u['x'], u['y'])))
    candidates = []
    for s in segments:
        cx, cy = s['centroid']
        if cx < lo or cx > hi:
            continue
        xy = np.asarray(s['p0'])[None]+np.linspace(0, 1, 7)[:, None]*(np.asarray(s['p1'])-s['p0'])[None]
        yu, yd = [np.interp(xy[:, 0], p['x'], p['y']) for p in (u, d)]
        inside = (xy[:, 0] >= lo) & (xy[:, 0] <= hi) & (xy[:, 1] >= yu) & (xy[:, 1] <= yd)
        if inside.mean() < .8:
            continue
        theta = np.deg2rad(s['direction_deg'])
        if abs(np.cos(theta)) < .01:
            continue
        m, b = np.tan(theta), cy-np.tan(theta)*cx
        slope = local_slope(d, cx, width)
        pa = abs((s['direction_deg']-np.rad2deg(np.arctan(slope))+90) % 180-90)
        q = dict(id=s['label_id'], centroid=s['centroid'], PA=float(pa), FL_finite=None,
                 FL_ray=None, ray_support=None, finite_support=None,
                 region=min(2, int(3*(cx-lo)/max(hi-lo, 1))))
        for mode in ('finite', 'ray'):
            rs = [roots(p, m, b, mode == 'ray', width) for p in (u, d)]
            if all(len(r) == 1 for r in rs):
                points = [r[0] for r in rs]
                length = float(np.linalg.norm(np.subtract(points[0][:2], points[1][:2])))
                if np.isfinite(length) and length < 1e7:
                    q['FL_'+mode] = length/(scale or 1.)
                    q[mode+'_support'] = [p[2] for p in points]
                    q[mode+'_points'] = [p[:2] for p in points]
        candidates.append(q)
    out = dict(MT=mt/(scale or 1.), contributors=candidates, weights={}, unit='mm' if scale else 'px')
    for field in ('PA', 'FL_finite', 'FL_ray'):
        out[field], out['weights'][field] = aggregate(candidates, field)
    out['ray_unobserved_mass'] = float(sum(out['weights']['FL_ray'].get(str(c['id']), 0.)
                                          for c in candidates if c['ray_support'] and max(c['ray_support']) > 0))
    return out


def reference_score(pair, z):
    result = {}
    x = z['x'].astype(float)
    radius = int(z['radius'])
    for i, side in enumerate(SIDES):
        y = z['ys'][i]
        supported = np.zeros(len(x), bool)
        predicted = np.full(len(x), np.nan)
        observed, interpolated, extended = [np.zeros(len(x), bool) for _ in range(3)]
        if pair:
            path = pair[side]
            supported = (x >= path['x'][0]) & (x <= path['x'][-1])
            predicted[supported] = np.interp(x[supported], path['x'], path['y'])
            nearest = np.clip(np.searchsorted(path['x'], x[supported]), 0, len(path['x'])-1)
            types = path['kind'][nearest]
            observed[supported], interpolated[supported], extended[supported] = [types == k for k in (0, 1, 2)]
        error = abs(predicted-y)
        good = supported & (error <= radius)
        cuts = np.flatnonzero(np.diff(np.r_[False, ~good, False]))
        result[side] = dict(columns=len(x), radius=radius, coverage=float(good.mean()),
                            supported=float(supported.mean()), gap=int(np.max(cuts[1::2]-cuts[::2], initial=0)),
                            mean_error=float(error[supported].mean()) if supported.any() else None,
                            p95_error=float(np.quantile(error[supported], .95)) if supported.any() else None,
                            normalized_error=float(error[supported].mean()/float(z['reference_width'])) if supported.any() else None,
                            observed_fraction=float(observed.mean()), bridged_fraction=float(interpolated.mean()),
                            extended_fraction=float(extended.mean()),
                            imputed_correct=int((good & ~observed).sum()), imputed_wrong=int((supported & ~observed & ~good).sum()),
                            hit_bits=np.packbits(good).tolist())
    if pair:
        common = (x >= pair['interval'][0]) & (x <= pair['interval'][1])
        if common.any():
            pred_mt = np.interp(x[common], pair['lower']['x'], pair['lower']['y'])-np.interp(x[common], pair['upper']['x'], pair['upper']['y'])
            true_mt = z['ys'][1, common]-z['ys'][0, common]
            result['MT_error_px'] = float(np.mean(abs(pred_mt-true_mt)))
        else:
            result['MT_error_px'] = None
    else:
        result['MT_error_px'] = None
    return result


def display_pair(pair):
    q = compact_pair(pair)
    if q:
        for side in SIDES:
            for key in ('x', 'y'):
                q[side][key] = np.round(q[side][key], 3)
    return q


def process(rec, probability, cfg, metadata, binding):
    cv2.setNumThreads(1)
    tic = time.monotonic()
    dest = OUT/'records'/rec['dataset']/Path(rec['image_id']).stem/(rec['view']+'.json.gz')
    source = ROOT/rec['path']
    assert sha(source) == rec['sha256']
    with np.load(source) as z:
        image, evidence, width = z['image'], z['evidence'].astype(bool), float(z['reference_width'])
        segs = mapped_segments(metadata, rec)
        scale = metadata['px_per_mm']*rec['matrix'][1][1] if metadata['px_per_mm'] else None
        result = decode(probability, image, evidence, width, cfg, segs, scale)
        record = dict(key=rec['key'], image_id=rec['image_id'], dataset=rec['dataset'], view=rec['view'],
                      group=metadata['group'], manual=rec['manual'], input=rec, binding=binding,
                      probability_sha256=array_sha(probability), arms={}, audit=result['audit'],
                      track_counts={k: list(map(len, v)) for k, v in result['tracks'].items()},
                      segments=segs, scale=scale, width=width, shape=image.shape)
        for arm, pair in result['arms'].items():
            q = dict(pair=display_pair(pair), geometry=geometry(original_pair(pair, rec), metadata['segments'], metadata['px_per_mm'], width/rec['matrix'][0][0]))
            if rec['manual']:
                q['reference'] = reference_score(pair, z)
                q['referenceLines'] = [dict(x=z['x'], y=y) for y in z['ys']]
            q['pair_count'] = len(result['candidates'].get(arm, result['candidates']['P']))
            record['arms'][arm] = q
        # Images are display artifacts only; all computation above uses untouched arrays.
        folder = OUT/'gallery/assets'/rec['dataset']/Path(rec['image_id']).stem
        folder.mkdir(parents=True, exist_ok=True)
        ratio = min(1., 1000/max(image.shape))
        small = cv2.resize(image.astype(np.uint8), None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder/(rec['view']+'.jpg')), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        panels = []
        for p in probability:
            a = cv2.resize(p, (256, 256), interpolation=cv2.INTER_AREA)
            panels.append(cv2.applyColorMap(np.rint(a*255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA))
        cv2.imwrite(str(folder/(rec['view']+'_probability.jpg')), np.concatenate(panels, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 90])
    record['seconds'] = time.monotonic()-tic
    write(dest, record)
    return dict(key=rec['key'], view=rec['view'], seconds=record['seconds'], destination=str(dest))
