"""Audit the residual Round52 severe FL events without changing the pipeline."""
import copy
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache

from scripts.round52_layer_replay import (
    DEST as R52, ROOT, CFG, R40, EVENTS, KINDS, SEEDS, Path, check_equal, error,
    gzread, gzwrite, initialize, j, key, measure, np, readcached, rows, sha, source_path, write,
    load_fascicles,
)
from scripts.round32_common import inverse_spec, map_curve
from scripts.round39_core import coverage
from scripts.round46_decoder import POOL46, SOURCE46, core, rays46
from scripts.round25_evaluate import REF, f1_score, validate_reference
from scripts.round51_fl_attribution import DEST as R51
from scripts.round36_core import support


DEST = ROOT/'runs/round53_boundary_stage_audit_v1'
SIDES = ('sup', 'deep')
TOLERANCES = (.05, .10, .15)
THRESHOLD = .20


def trace_pool():
    anchor = '    return merged,audit'
    assert SOURCE46.count(anchor) == 1
    source = SOURCE46.replace(anchor, '    return merged,pools,maps,factor,(x0,y0)')
    ns = dict(POOL46.__globals__)
    exec(compile(source, '<round53_read_only_pool_trace>', 'exec'), ns)
    return ns['normalized_pool_from_columns']


TRACED = trace_pool()


def compare(guide, candidate, ref_width):
    """Match layer geometry only where both have support, independently of its width."""
    xs = np.asarray(guide['paired_x'])
    present = support(candidate, xs)
    if present.sum() < max(8, .10 * ref_width):
        return dict(comparable=False, overlap=float(present.mean()))
    xs = xs[present]
    a = [np.interp(xs, guide[s]['knots_x'], guide[s]['knots_y']) for s in SIDES]
    b = [np.interp(xs, candidate[s]['knots_x'], candidate[s]['knots_y']) for s in SIDES]
    depth = np.median(a[1] - a[0])
    delta = np.abs(np.asarray(a) - b)
    samples = {str(t): float(np.min(np.mean(delta <= max(4 * ref_width / 1046, t * depth), axis=1)))
               for t in TOLERANCES}
    p90 = [float(x) for x in np.quantile(delta, .9, axis=1)]
    all_x = np.unique(np.concatenate([guide['paired_x'], candidate['paired_x']]))
    xlo = min(guide['support'][0], candidate['support'][0])
    xhi = max(guide['support'][1], candidate['support'][1])
    grid = np.arange(np.ceil(xlo), xhi + 1e-7)
    grid = grid if len(grid) else all_x
    ma, mb = support(guide, grid), support(candidate, grid)
    return dict(comparable=True, overlap=float(present.mean()), support_iou=float((ma & mb).sum()/max((ma | mb).sum(),1)),
                matching=samples, error_p90_px=p90, original_depth=float(depth))


def near(row, tol):
    return row['comparable'] and row['matching'][str(tol)] >= .8


def corridor_coverage(guide, width, raw, bounded, probability, box, input_box):
    """Thresholded and (when cached) continuous responses, side by side."""
    _, h, w = raw.shape
    l, top, right, bottom = box
    lo, hi = guide['support']
    xs = np.arange(max(0, int(np.ceil(lo))), min(w, int(np.floor(hi)) + 1))
    xs = xs[support(guide, xs)]
    rad = max(1, round(4 * width / 1046))
    out = {}
    for label, radius in (('r4',rad), ('r8',2*rad)):
        out[label] = {}
        for field, masks in (('raw',raw), ('bounded',bounded), ('prob35',probability)):
            if masks is None:
                continue
            vals = []
            for role, side in enumerate(SIDES, start=1):
                yy = np.rint(np.interp(xs, guide[side]['knots_x'], guide[side]['knots_y'])).astype(int)
                yprobe = np.clip(yy[:, None] + np.arange(-radius,radius+1), 0, h-1)
                q = masks[[0, role]][:, yprobe, xs[:, None]]
                found = (q > (.35 if field == 'prob35' else .5)).any(axis=(0, 2))
                valid = (yy >= 0) & (yy < h)
                if field == 'prob35':
                    valid &= np.isfinite(q).any(axis=(0,2))
                vals.append(found & valid)
            n = max(len(xs), 1)
            out[label][field] = dict(upper=float(vals[0].sum()/n), lower=float(vals[1].sum()/n),
                                     joint=float((vals[0]&vals[1]).sum()/n))
    yy = [np.interp(xs, guide[side]['knots_x'], guide[side]['knots_y']) for side in SIDES]
    x0, y0, x1, y1 = input_box
    out['inside_evidence_box'] = float(np.mean((xs>=l)&(xs<right)&(yy[0]>=top)&(yy[1]<bottom)))
    out['inside_network_input'] = float(np.mean((xs>=x0)&(xs<x1)&(yy[0]>=y0)&(yy[1]<y1)))
    out['sample_count'] = len(xs)
    return out


def track_coverage(guide, maps, factor, offset):
    x0,y0 = offset
    denominator = len(guide['paired_x'])
    def columns(track, side):
        pts = np.asarray(list(track.items()))/factor + [x0,y0]
        xs, ys = pts.T
        yfit = np.interp(xs,guide[side]['knots_x'],guide[side]['knots_y'])
        return {round(float(x)) for x in xs[support(guide,xs) & (abs(ys-yfit)<=4/factor)]}

    out = {}
    for name, (upper,lower) in (('D0',(maps[0],maps[0])),('D1',(maps[1],maps[2]))):
        a,b = ([columns(track,s) for track in tracks] for s,tracks in zip(SIDES,(upper,lower)))
        joint = max((len(x&y) for x in a for y in b),default=0)
        out[name] = dict(upper=max(map(len,a),default=0)/denominator,
                         lower=max(map(len,b),default=0)/denominator,joint=joint/denominator)
    return out


def stage(traces, tol):
    pairs = traces['merged']
    selected = traces['selected']
    if near(selected,tol):
        return 'same_layer_lower_support' if selected['support_iou'] < .8 else 'same_layer_similar_support'
    if any(near(p,tol) and p['allowed'] for p in pairs):
        return 'matching_candidate_allowed_not_selected'
    if any(near(p,tol) for p in pairs):
        return 'matching_candidate_rejected'
    if any(near(p,tol) for p in traces['premerge']):
        return 'matching_candidate_lost_in_dedup'
    evidence = traces['corridor']['r4']
    if evidence['bounded']['joint'] >= .30:
        return 'mask_support_but_no_matching_fit'
    if evidence['raw']['joint'] >= .30:
        return 'evidence_box_reduces_support'
    return 'weak_or_disjoint_mask_support'


def mix_support(geometry, evidence):
    out = copy.deepcopy(geometry)
    for part in (None, *SIDES):
        a, b = (out, evidence) if part is None else (out[part], evidence[part])
        for f in ('support','paired_x','max_sample_gap'):
            a[f] = b[f]
    return out


@cache
def settings():
    return j(CFG), j(R40/'calibration.json')


def audit_image(r, events, ref):
    cfg, cal = settings()
    sources = {s:gzread(source_path(r, s)) for s in {int(e['condition'].split('|')[0]) for e in events}}
    traces, fascicles = {}, {}
    results = []
    for e in events:
        seed, kind = e['condition'].split('|')
        seed, tr = int(seed), e['transform']
        old = sources[seed]
        vo, v = [old['variants'][name] for name in ('identity', tr)]
        a, b = [z['arms'][kind+'|ID'] for z in (vo, v)]
        roi, region, t = v['measurement_roi'], v['region'], v['transform']
        width = region['reference_width']
        guide = map_curve(a['curve'], t)
        current = map_curve(b['curve'], t)
        k = seed,tr
        if k not in traces:
            raw, meta = readcached(r, tr, 'boundary_'+str(seed), 'I')
            assert meta['sha256'] == old['sources'][tr]['boundary_I']
            l, top, right, bottom = region['evidence_box']
            bounded = np.zeros_like(raw)
            bounded[:,top:bottom,l:right] = raw[:,top:bottom,l:right]
            cols = [list(enumerate(core.run_centres_per_column(m))) for m in bounded]
            pool, before, maps, factor, (x0,y0) = TRACED(cols, roi, reference_width=width)
            fitted = [map_curve(p['curve'], dict(sx=1/factor,sy=1/factor,tx=x0,ty=y0))
                      for group in before.values() for p in group if p['curve'] is not None]
            with np.load(ROOT/meta['data_path']) as ar:
                prob = ar['probability'] if 'probability' in ar else None
            if prob is not None:
                assert np.array_equal(raw, prob > .5)
            traces[k] = (pool, fitted, raw, bounded, prob, meta, maps, factor, (x0,y0))
        pool, fitted, raw, bounded, prob, meta, maps, factor, offset = traces[k]
        gates = {g['pair_id']:g for g in b['gates']}
        assert {p['pair_id'] for p in pool} == set(gates)
        record = dict(selected=compare(guide,current,width),
                      merged=[{**compare(guide,p['curve'],width), 'id':p['pair_id'],
                               'joint_observed_fraction':coverage(p['curve']['paired_x'],1.)/width,
                               'allowed':gates[p['pair_id']]['structural_allowed'],
                               'reasons':gates[p['pair_id']]['structural_reasons']} for p in pool],
                      premerge=[compare(guide,cv,width) for cv in fitted],
                      corridor=corridor_coverage(guide,width,raw,bounded,prob,region['evidence_box'],meta['input_box']),
                      tracks=track_coverage(guide,maps,factor,offset))
        stages = {str(tol):stage(record,tol) for tol in TOLERANCES}
        if ref is not None:
            def matches(cv):
                return f1_score(dict(status='candidate', curve=cv), ref)[0]

            true_ids = [p['pair_id'] for p in pool if matches(map_curve(p['curve'], inverse_spec(t)))]
            record['reference'] = dict(original=bool(matches(a['curve'])), factual=bool(matches(b['curve'])),
                moved_matching_pool=true_ids, moved_matching_allowed=[i for i in true_ids if gates[i]['structural_allowed']])
        if (tr,kind) not in fascicles:
            _, raw_c1, kept = load_fascicles(r,tr,kind,old,roi,cfg)
            fascicles[tr,kind] = len(raw_c1), kept
        n, kept = fascicles[tr,kind]
        def fl(cv, id):
            return measure(cv,id,kept,n,roi,t,old,cfg,cal)['values']['FL']

        x = a['values']['FL']
        selected = b['selected_pair_id']
        check_equal(fl(current,selected), b['values']['FL'], 'factual FL')
        check_equal(error('FL',x,fl(guide,a['selected_pair_id'])), e['differences']['FB'], 'freeze boundary FL')
        values = {label:fl(cv,selected) for label,cv in
                  (('geometry_only',mix_support(guide,current)), ('support_only',mix_support(current,guide)))}
        errors = {label:error('FL',x,y) for label,y in values.items()}
        alts = {p['id']:p for p in record['merged'] if p['id'] != selected and near(p,.10)}
        if alts:
            alternatives = [p for p in rays46(pool,roi,width) if p['pair_id'] in alts]
            for p in alternatives:
                i = p['pair_id']
                mode = 'allowed_alternatives' if alts[i]['allowed'] else 'rejected_alternatives'
                errors.setdefault(mode,{})[str(i)] = error('FL',x,fl(p['curve'],i))
        results.append(dict(**e, stages=stages, audit=record, intervention=errors))
    return results


def main():
    p = R52/'conservative/summary.json'
    surviving = j(p)['events']
    def event_id(r):
        return r['key'], r['condition'], r['transform']

    expected = {event_id(e):e for e in surviving['severe'] if e['arms']['separator']['FL'] > THRESHOLD}
    assert len(expected) == 102
    pool = {key(r):r for r in rows()}
    refs, _ = validate_reference(REF,{r['image_id']:r for r in pool.values() if r['dataset']=='apo153'})
    tasks = defaultdict(list)
    for f in (R51/'fl_replay').glob('*/*.json.gz'):
        for e in gzread(f):
            if e['cohort'] == 'controls' or event_id(e) in expected:
                if event_id(e) in expected:
                    check_equal(e['differences']['N'],expected[event_id(e)]['arms']['separator']['FL'],'unchanged Round52 residual')
                tasks[e['key']].append({k:e[k] for k in ('key','condition','transform','cohort','category','differences','boundary')})
    assert sum(map(len,tasks.values())) == 140
    out = []
    with ProcessPoolExecutor(max_workers=10, initializer=initialize) as proc:
        fs = [proc.submit(audit_image,pool[k],es,refs.get(pool[k]['image_id']) if pool[k]['dataset']=='apo153' else None) for k,es in tasks.items()]
        for f in as_completed(fs):
            out.extend(f.result())
    assert {event_id(e) for e in out if e['cohort']=='severe'} == set(expected)
    out.sort(key=event_id)
    gzwrite(DEST/'events.json.gz',out)
    write(DEST/'sources.json',dict(round52_sha256=sha(p), code_sha256=sha(Path(__file__)), reference_sha256=sha(REF),
                                  role_match_tolerance=TOLERANCES, support_iou=.8, support_mask_joint=.30))
    print('Boundary attribution audit complete',len(out),flush=True)


if __name__ == '__main__':
    main()
