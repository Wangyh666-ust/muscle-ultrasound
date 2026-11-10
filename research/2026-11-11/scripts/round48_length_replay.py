"""Isolated Round48 replay. The selected boundary never sees the length intervention."""
from scripts.round47_fast_v2 import initialize as fast_initialize
from scripts.round47_common import (
    ROOT, R46, R40, R41, CFG, KINDS, SEEDS, SCOPES, SPLIT, key, rows, j, sha,
    write, gzread, gzwrite, Path, np, cv2,
)
from scripts.round46_measure import readcached
from scripts.round41_measure import finish
from scripts.round32_common import map_curve, map_segment, inverse_spec
from scripts.round38_core import base_measure, qualified, aggregate, raw_components
from scripts.round37_core import filter_components
from scripts.round39_core import modify_fl
from scripts.round40_core import apply_u
from scripts.round39_verify import check_equal
import argparse
import concurrent.futures as cf
import time

DEST = ROOT / 'runs/round48_length_attribution_v1'
LENGTH = 20.0
LENGTH_FLAG = 'length_lt_30_reference'
PROB_TRANSFORMS = ('translate-2_1', 'r41_scale1015', 'brightness105')
CASE_KEYS = (
    'dev282|image_2404.tif', 'dev282|image_0566.tif', 'apo153|image_0643.tif',
    'apo153|image_1009.tif', 'apo153|image_1011.tif', 'external34|im_12_arch',
)


def probability_keys():
    import random
    rng = random.Random(2026110548)
    selected = []
    for ds, n in [('dev282', 16), ('apo153', 6), ('external34', 4)]:
        pool = sorted(key(r) for r in rows('development')
                      if r['dataset'] == ds and key(r) not in CASE_KEYS)
        selected.extend(rng.sample(pool, min(n, len(pool))))
    return selected + list(CASE_KEYS)


def register():
    p = DEST / 'protocol.json'
    if p.exists():
        return j(p)
    old_sources = j(ROOT / 'runs/round47_boundary_selection_v1/protocol.json')['protected']
    protocol = dict(
        training=False, default_changed=False, subagents=False,
        protected=old_sources, length_ref=LENGTH, baseline_length_ref=30,
        boundaries=list(SEEDS), c1=list(KINDS), scopes={k: SCOPES[k] for k in ('combined16', 'all26')},
        split_sha256=sha(R41/'partitions.json'),
        plan_sha256=sha(ROOT/'thinking/ROUND48_TRACK_STABILITY_PLAN.txt'),
        source='Round46 I masks, exact Round46 ID/Round47 R boundary and measurement',
        decision='20 reference pixels is the single prespecified conservative candidate. '
                 'Round42 removed the length floor and found both useful and unstable short lines.',
        shared='Identical C1 masks, ROI, selected boundary and extrapolation, deduplication and weighting rules. '
               'Within each view fixed between arms. Probability panel also anchors both views to the original boundary.',
        probability=dict(keys=probability_keys(), transforms=PROB_TRANSFORMS, thresholds=[.5, .35],
                         corridor_reference_px=2.5, coverage_loss=.2, min_base_coverage=.6),
        primary='331 historical confirmation images, all26 and old16, per C1 seed with equal boundary conditions. '
                'PA and FL pass, missing and severe tails, original reference support, and natural pair differences. '
                'Missing stays in denominator. Candidate should improve both seeds without worse coverage, '
                'severe tails, or corroborating deterioration on strict pairs.',
    )
    write(p, protocol)
    return protocol


def source_path(r, seed):
    return R46 / f"measurements/{seed}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"


def output_path(r):
    return DEST / f"replay/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"


def read_mask(r, tr, kind, variant):
    data, meta = readcached(r, tr, kind, 'I')
    assert all(meta['transform'][k] == variant['transform'][k]
               for k in ('sx', 'sy', 'tx', 'ty', 'width', 'height'))
    mask = np.zeros_like(data[0])
    left, top, right, bottom = variant['region']['evidence_box']
    mask[top:bottom, left:right] = data[0, top:bottom, left:right]
    return mask, meta


def extract(mask, roi, cfg, length=LENGTH):
    _, raw = raw_components(mask)
    kept, audit = filter_components(raw, roi, cfg['reference_filter'], 'scale')
    keep_ids = {s['label_id'] for s in kept}
    added = []
    for segment, gate in zip(raw, audit):
        if gate['flags'] == [LENGTH_FLAG] and gate['reference_length'] >= length - 1e-9:
            added.append({**segment, 'filter_quality': gate['quality'], 'filter_evidence': gate})
    relaxed = sorted(kept + added, key=lambda s: s['label_id'])
    assert len(keep_ids) + len(added) == len(relaxed)
    return raw, kept, relaxed, added, audit


def measure(curve, pair_id, segments, raw_count, roi, transform, row, cfg, cal):
    native = dict(width=transform['width'], height=transform['height'], px_per_mm=None)
    # base_measure on a one-element pool can only pick this pair.
    pool = [] if curve is None else [dict(pair_id=pair_id, curve=curve, old_quality=1., quality=1.)]
    # Supply empty selection evidence. With a single frozen pair its score cannot affect selection.
    evidence = np.zeros((1, 1), bool)
    st = dict(segments=segments, raw_count=raw_count, roi=roi, evidence=evidence)
    base, cs = base_measure(pool, st, native)
    assert base['curve'] == curve
    result = aggregate(qualified(base, cs), cs, native, cfg)
    result = modify_fl(result, cfg, cal, True, True)
    result['_native_dimensions'] = native
    native_output = apply_u(result, cfg, cal)
    return finish(native_output, transform, row)


def reduced(a, added=(), detailed=False):
    weights = {m: a[m+'_weights'] for m in ('PA', 'FL')}
    added_ids = {str(s['label_id']) for s in added}
    risk = a.get('boundary_sensitivity') or a.get('display', {}).get('risk') or {}
    active_ids = {sid for q in weights.values() for sid, w in q.items() if w > 0}
    out = dict(
        values=a['values'], raw=a['raw_components'], filtered=a['filter_count'],
        count={m: sum(w > 0 for w in q.values()) for m, q in weights.items()},
        max_weight={m: max(q.values(), default=0.) for m, q in weights.items()},
        added_used={m: sum(i in added_ids and w > 0 for i, w in q.items()) for m, q in weights.items()},
        added_weight={m: sum(w for i, w in q.items() if i in added_ids) for m, q in weights.items()},
        added_risk={i: v for i, v in risk.get('tracks', {}).items() if i in added_ids and i in active_ids},
        boundary=a['curve'] is not None,
    )
    if detailed:
        out['weights'] = weights
        candidates = a.get('candidates')
        out['segments'] = ([dict(c['segment'], sup=c['intersection_sup'], deep=c['intersection_deep'],
                                 PA=c['PA'], FL_px=c['raw_FL']) for c in candidates]
                           if candidates is not None else a['display']['segments'])
    return out


def initialize():
    fast_initialize()
    cv2.setNumThreads(1)


def replay(r, pilot=False):
    cfg, cal = j(CFG), j(R40/'calibration.json')
    dst = DEST/'pilot'/output_path(r).relative_to(DEST) if pilot else output_path(r)
    if dst.exists():
        return gzread(dst)['verification']
    old = {s: gzread(source_path(r, s)) for s in SEEDS}
    row = old[SEEDS[0]]
    names = list(dict.fromkeys(['identity', *SCOPES['all26']]))
    names = [tr for tr in names if tr in row['variants']]
    if pilot:
        names = [tr for tr in names if tr in ('identity', *PROB_TRANSFORMS)]
    variants, original_tracks = {}, {}
    parity = 0
    changed = 0
    for tr in names:
        v = row['variants'][tr]
        t, roi = v['transform'], v['measurement_roi']
        measured = {}
        for kind in KINDS:
            mask, meta = read_mask(r, tr, kind, v)
            raw, kept, relaxed, added, audit = extract(mask, roi, cfg)
            if tr == 'identity':
                original_tracks[kind] = dict(base=kept, added=added)
            for seed in SEEDS:
                vb = old[seed]['variants'][tr]
                assert vb['transform'] == t and vb['measurement_roi'] == roi
                assert old[seed]['sources'][tr][kind+'_I'] == meta['sha256']
                baseline = vb['arms'][kind+'|ID']
                assert (len(raw), len(kept), int(mask.sum())) == (
                    baseline['raw_components'], baseline['filter_count'],
                    baseline['C1_evidence']['consumed_pixels'])
                curve = map_curve(baseline['curve'], t)
                should_check = curve is not None and (pilot or bool(added))
                if should_check:
                    check = measure(curve, baseline['selected_pair_id'], kept, len(raw), roi, t, row, cfg, cal)
                    for field in ('values', 'PA_weights', 'FL_weights'):
                        check_equal(check[field], baseline[field], f"{key(r)} {tr} {seed} {kind} {field}")
                    parity += 1
                detailed = tr == 'identity' or (bool(added) and curve is not None)
                base = reduced(baseline, detailed=detailed)
                if added and curve is not None:
                    trial = measure(curve, baseline['selected_pair_id'], relaxed, len(raw), roi, t, row, cfg, cal)
                    check_equal(trial['values']['MT'], baseline['values']['MT'], 'frozen MT')
                    check_equal(trial['curve'], baseline['curve'], 'frozen boundary')
                    trial = reduced(trial, added, detailed=detailed)
                else:
                    trial = {**base, 'filtered': len(relaxed)}
                changed += trial['values'] != base['values']
                measured[f'{seed}|{kind}'] = dict(base=base, length=trial, added=len(added))
        variants[tr] = measured
    result = dict(key=key(r), dataset=r['dataset'], image_id=r['image_id'],
                  group_id=r['group_id'], partition=SPLIT['assignment'][key(r)],
                  roi_width=row['variants']['identity']['measurement_roi']['box'][2]
                            - row['variants']['identity']['measurement_roi']['box'][0],
                  scale=row['px_per_mm'], variants=variants, original_tracks=original_tracks,
                  verification=dict(variants=len(names), exact_baseline_replays=parity, changed_outputs=changed),
                  source_sha256={str(s): sha(source_path(r, s)) for s in SEEDS})
    gzwrite(dst, result)
    return result['verification']


def main(pilot, workers):
    protocol = register()
    if pilot:
        selected = set(protocol['probability']['keys']) - set(CASE_KEYS)
        batch = [r for r in rows('development') if key(r) in selected][:12]
    else:
        lock = j(DEST/'lock.json')
        for path, digest in lock['sources'].items():
            assert sha(ROOT/path) == digest, path
        batch = rows()
    t0 = time.monotonic()
    records = []
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        tasks = {pool.submit(replay, r, pilot): key(r) for r in batch}
        for task in cf.as_completed(tasks):
            rec = dict(key=tasks[task], **task.result())
            records.append(rec)
            if len(records) % 15 == 0 or len(records) == len(batch):
                print('Round48 replay', len(records), '/', len(batch), 'seconds', round(time.monotonic()-t0), flush=True)
    write(DEST/('pilot_success.json' if pilot else 'replay_success.json'),
          dict(rows=len(records), seconds=time.monotonic()-t0, records=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    main(args.pilot, args.workers)
