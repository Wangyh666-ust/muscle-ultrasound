"""Fit one observed boundary at a time, with the frozen four-piece estimator."""
from scripts.round55_replay import SIDES, np, owned_columns
from scripts.round36_core import support
from scripts.round39_core import _design
from scripts.round29_core import value_at


def fit_side(points, width, height):
    xy = np.asarray(points, dtype=float)
    if len(xy) < 8 or np.any(np.diff(xy[:, 0]) <= 0):
        return None
    xs, ys = xy.T
    lo, hi = xs[[0, -1]]
    design, _ = _design(xs, lo, hi)
    if design is None or np.linalg.matrix_rank(design) < 5:
        return None
    knots = np.linalg.lstsq(design, ys, rcond=None)[0]
    residual = float(np.quantile(abs(design @ knots - ys), .9))
    if residual > max(3 * width / 1046, .02 * height):
        return None
    rays = {}
    for i, name in ((0, 'left_ray'), (-1, 'right_ray')):
        x = xs[i]
        weights = np.exp(-.25 * ((xs - x) / (.25 * width)) ** 2)
        line = np.column_stack((xs - x, np.ones(len(xs))))
        slope = float(np.linalg.lstsq(line * weights[:, None], ys * weights, rcond=None)[0][0])
        rays[name] = [slope, float(knots[i] - slope * x)]
    return dict(knots_u=[0,.25,.5,.75,1], knots_x=np.linspace(lo,hi,5).tolist(), knots_y=knots.tolist(),
                support=[float(lo),float(hi)], paired_x=xs.tolist(), max_sample_gap=max(2,.015*1046)*width/1046,
                image_shape=[height,width], residual_p90_px=residual, rank=5, n_observations=len(xs), **rays)


def ordered(curve):
    lo, hi = curve['support']
    xs = sorted({lo,hi,*(x for side in SIDES for x in curve[side]['knots_x'] if lo<=x<=hi)})
    return all(value_at(curve['sup'],x) < value_at(curve['deep'],x) for x in xs)


def retained_pair(pair, maps, factor, offset, width, height, sides=SIDES):
    """Retain a uniquely associated side's own observations beyond shared columns."""
    if pair is None:
        return None
    owned = owned_columns(pair,maps,factor,offset)
    old = pair['curve']
    curve = dict(old)
    for side in sides:
        evidence = owned[side]
        if evidence['association'] != 'single_track':
            continue
        xs = evidence['columns']
        if len(xs) <= len(old['paired_x']):
            continue
        fit = fit_side(np.column_stack((xs,evidence['ordinates'])),width,height)
        if fit is None:
            continue
        check_x = np.asarray(old['paired_x'])
        delta = abs(np.interp(check_x,fit['knots_x'],fit['knots_y']) - np.interp(check_x,old[side]['knots_x'],old[side]['knots_y']))
        # Preserve the already qualified geometry over its shared observations.
        if np.quantile(delta,.95) <= max(1,4*width/1046):
            curve[side] = fit
    if all(curve[s] is old[s] for s in SIDES) or not ordered(curve):
        return pair
    return {**pair,'curve':curve}


def values_at(fit, xs):
    return np.interp(xs, fit['knots_x'], fit['knots_y'])


def match_fit(fit, guide, tol, column_tol):
    xs, ys = guide.T
    observed = support(fit,xs)
    cols = np.asarray(fit['paired_x'])
    idx = np.searchsorted(cols,xs)
    dist = np.minimum(abs(xs-cols[np.clip(idx,0,len(cols)-1)]),abs(xs-cols[np.clip(idx-1,0,len(cols)-1)]))
    good = observed & (dist<=column_tol) & (abs(values_at(fit,xs)-ys)<=tol)
    return good


def longest_gap(good):
    cuts = np.flatnonzero(np.diff(np.r_[False,~good,False]))
    return int(np.max(cuts[1::2]-cuts[::2],initial=0))


def controls():
    upper = np.column_stack((np.arange(0,161),20+.1*np.arange(0,161)))
    lower = np.column_stack((np.arange(40,201),55+.1*np.arange(40,201)))
    fit = fit_side(upper,200,100)
    assert fit['support']==[0,160] and len(fit['paired_x'])==161
    assert np.max(abs(values_at(fit,upper[:,0])-upper[:,1]))<1e-9
    cut = upper[~np.isin(upper[:,0],np.arange(63,83))]
    assert len(fit_side(cut,200,100)['paired_x'])==141
    bend = upper.copy()
    bend[:,1] += np.maximum(0,bend[:,0]-80)*.15
    assert np.max(abs(values_at(fit_side(bend,200,100),bend[:,0])-bend[:,1]))<1e-9
    pts = np.column_stack((np.arange(40,161),upper[40:,1],lower[:121,1]))
    base = dict(sup=fit_side(pts[:,[0,1]],200,100),deep=fit_side(pts[:,[0,2]],200,100),
                support=[40,160],paired_x=pts[:,0].tolist(),max_sample_gap=3)
    pair = dict(pair_id=0,curve=base,observations=pts.tolist(),source_decoder='D1')
    ambiguous = dict(upper)
    ambiguous[12] += 12
    maps = [[],[dict(upper)],[dict(lower)]]
    both = retained_pair(pair,maps,1,(0,0),200,100)
    assert both['curve']['sup']['support']==[0,160] and both['curve']['deep']['support']==[40,200]
    assert both['curve']['paired_x']==base['paired_x']
    one = retained_pair(pair,maps,1,(0,0),200,100,('deep',))
    assert one['curve']['sup']==base['sup'] and one['curve']['deep']['support']==[40,200]
    maps[1].append(ambiguous)
    assert retained_pair(pair,maps,1,(0,0),200,100,('sup',)) is pair
    return dict(staggered=True, preserves_joint=True, one_side=True, ambiguous=True, internal_gap=True, bend=True)
