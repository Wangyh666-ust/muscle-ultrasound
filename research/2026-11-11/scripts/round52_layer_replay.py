"""Replay the inner-track ranking change on frozen masks and measurements."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
import math
import time

from scripts.round48_length_replay import (
    ROOT, R40, R41, CFG, KINDS, SEEDS, SPLIT, Path, check_equal, extract,
    gzread, gzwrite, initialize, j, key, measure, np, readcached, rows, sha,
    source_path, write,
)
from scripts.round39_core import coverage
from scripts.round40_core import evidence_gates
from scripts.round46_decoder import rays46
from scripts.round48_evaluate import error, status, summary, paired_ci
from scripts.round52_layer_scoring import rank_layers, scored_pool, separator_evidence


DEST = ROOT/'runs/round52_inner_separator_v1'
EVENTS = ROOT/'runs/round51_dual_fascicle_v1/fl_event_manifest.json'
NATURAL = ROOT/'runs/round47_boundary_selection_v1/evaluation/natural_pairs.json.gz'
PANEL = ('translate-2_1', 'r41_scale1015', 'brightness105')
METRICS = ('MT', 'PA', 'FL')
ARMS = ('legacy', 'separator')


def controls():
    points = [(float(x), 10., 50.) for x in range(100)]
    fixtures = {
        'no_divider': ([], 0.),
        'one_short_line': ([dict.fromkeys(range(20, 40), 30)], .2),
        'unrelated_short_lines': ([dict.fromkeys(range(x, x + 20), y)
                                   for x, y in zip(range(0, 100, 20), range(20, 45, 5))], .2),
        'sparse_same_track': ([dict.fromkeys(range(0, 100, 5), 30)], .01),
        'near_edge': ([dict.fromkeys(range(100), 12)], 0.),
        'outside_band': ([dict.fromkeys(range(100), 55)], 0.),
        'mostly_continuous': ([dict.fromkeys([x for x in range(5, 95) if x != 49], 30)], .89),
        'complete_divider': ([dict.fromkeys(range(100), 30)], 1.),
    }
    out = {}
    for name, (tracks, expected) in fixtures.items():
        out[name] = separator_evidence(points, tracks, 1., coverage)
        assert math.isclose(out[name]['fraction'], expected), (name, out[name])
        expected_total = .2 if name == 'sparse_same_track' else expected
        assert math.isclose(out[name]['strongest_fraction'], expected_total), (name, out[name])
    return out


def freeze(quality_field):
    import json

    dependencies = (Path(__file__), ROOT/'scripts/round52_layer_scoring.py',
                    ROOT/'scripts/round46_decoder.py', CFG, R40/'calibration.json',
                    R41/'partitions.json', EVENTS, NATURAL)
    specification = dict(
        bindings={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies},
        formula={'quality':'shared ROI fraction times (1 - longest same-track interior observed fraction squared)',
                 'single_track_quality':'(shared coverage - strongest same-track observed interior coverage) / ROI width'}[quality_field],
        stage='follow-up after first replay; shared historical cohort is not independent' if quality_field == 'single_track_quality' else 'first replay',
        gap_reference_pixels=2, edge_clearance_reference_pixels=3,
        fixed=['candidate tracks', 'four-segment fits', 'deduplication', 'support gates',
               'C1 containment factors', 'rays', 'measurement and aggregation'],
        training=False, inference=False, default_changed=False, panel=PANEL,
        cohorts='all 469 originals and fixed panel; Round51 selected severe/control events separate',
        controls=controls(),
    )
    p = DEST/'protocol.json'
    if p.exists():
        assert j(p) == json.loads(json.dumps(specification)), 'protocol changed'
    else:
        write(p, specification)


@cache
def settings():
    return j(CFG), j(R40/'calibration.json')


def replay(r, seed, transforms, quality_field):
    cfg, cal = settings()
    old = gzread(source_path(r, seed))
    result = {}
    for tr in transforms:
        variant = old['variants'][tr]
        native = {k:variant['transform'][k] for k in ('height', 'width')}
        roi = variant['measurement_roi']
        region = variant['region']
        masks, meta = readcached(r, tr, 'boundary_'+str(seed), 'I')
        assert old['sources'][tr]['boundary_I'] == meta['sha256']
        left, top, right, bottom = region['evidence_box']
        bounded = np.zeros_like(masks)
        bounded[:, top:bottom, left:right] = masks[:, top:bottom, left:right]
        pool = scored_pool(bounded, roi, region['reference_width'])
        assert len(pool) == variant['boundary_audit']['ID']['candidate_count']
        result[tr] = {}
        rays = None
        for kind in KINDS:
            frozen = variant['arms'][kind+'|ID']
            gates = [dict(g) for g in frozen['gates']]
            assert [p['pair_id'] for p in pool] == [g['pair_id'] for g in gates]
            fascicles = None
            if any(p['old_quality'] == 0 for p in pool):
                fascicles = load_fascicles(r, tr, kind, old, roi, cfg)
                mask, raw, kept = fascicles
                st = dict(raw_mask=mask, roi=roi, segments=kept)
                rebuilt = evidence_gates(pool, st, native, cfg, cal,
                                         [{'reasons': g['structural_reasons']} for g in gates])
                for g, q in zip(gates, rebuilt):
                    check_equal(g['soft_score'], q['soft_score'], f'C1 replay {key(r)} {seed} {tr} {kind}')
                    g['soft_factor'] = q['soft_factor']
            else:
                for p, g in zip(pool, gates):
                    g['soft_factor'] = g['soft_score']/p['old_quality']
                    assert 0 < g['soft_factor'] <= 1 + 1e-8
            old_pair = rank_layers(pool, gates, native)
            new_pair = rank_layers(pool, gates, native, quality_field=quality_field)
            before = old_pair['pair_id'] if old_pair is not None else None
            after = new_pair['pair_id'] if new_pair is not None else None
            assert before == frozen['selected_pair_id'], (key(r), seed, tr, kind, before, frozen['selected_pair_id'])
            if before == after:
                updated = frozen
            else:
                if rays is None:
                    rays = {p['pair_id']:p['curve'] for p in rays46(pool, roi, region['reference_width'])}
                _, raw, kept = fascicles or load_fascicles(r, tr, kind, old, roi, cfg)
                args = (kept, len(raw), roi, variant['transform'], old, cfg, cal)
                check_equal(measure(rays[before], before, *args)['values'], frozen['values'],
                            f'changed-pair baseline replay {key(r)} {seed} {tr} {kind}')
                updated = measure(rays[after], after, *args)
            result[tr][kind] = dict(
                legacy={k:frozen[k] for k in ('values', 'selected_pair_id')},
                separator={k:updated[k] for k in ('values', 'selected_pair_id')},
            )
            rec = result[tr][kind]
            if tr == 'identity' and r['dataset'] == 'apo153':
                for arm, measured in zip(ARMS, (frozen, updated)):
                    rec[arm]['curve'] = measured['curve']
            if before != after or r['image_id'] == 'image_2576.tif':
                rec['candidates'] = [dict(pair_id=p['pair_id'], decoder=p['source_decoder'],
                    legacy_quality=p['old_quality'], eligible=g['structural_allowed'],
                    C1_factor=g['soft_factor'], legacy_score=g['soft_score'],
                    separator_score=p['separator'][quality_field]*g['soft_factor'], **p['separator'])
                    for p, g in zip(pool, gates)]
    return dict(key=key(r), seed=seed, width=r['width'],
                roi_width=old['variants']['identity']['region']['reference_width'],
                scale=old['px_per_mm'], variants=result)


def load_fascicles(r, tr, kind, old, roi, cfg):
    c1, meta = readcached(r, tr, kind, 'I')
    assert old['sources'][tr][kind+'_I'] == meta['sha256']
    raw, kept, _, _, _ = extract(c1[0], roi, cfg)
    return c1[0], raw, kept


def summarize_stability(records):
    scope = {}
    for label in ('all', 'confirmation'):
        for kind in KINDS:
            selected = [r for r in records if r['kind'] == kind and
                        (label == 'all' or r['partition'] == label)]
            stats = {arm:{m:{f:sum(r[arm][m][f] for r in selected)
                             for f in ('passed', 'missing', 'severe')} for m in METRICS}
                     for arm in ARMS}
            for m in METRICS:
                common = [r for r in selected if all(not r[a][m]['missing'] for a in ARMS)]
                for arm in ARMS:
                    stats[arm][m]['common_worst'] = summary([r[arm][m]['worst'] for r in common])
            scope[label+'|'+kind] = dict(n=len(selected), stats=stats,
                fl_pass_delta=paired_ci([(r['component'], int(r['separator']['FL']['passed']) -
                                          int(r['legacy']['FL']['passed'])) for r in selected]))
    return scope


def comparisons(all_results, manifest):
    from scripts.round25_evaluate import REF, f1_score, validate_reference
    from scripts.round36_evaluate import measurement_errors

    by_key = {(r['key'], r['seed']):r for r in all_results}
    dataset = rows()
    refs, unclear = validate_reference(REF, {r['image_id']:r for r in dataset if r['dataset'] == 'apo153'})
    evaluations, changes, stability = {}, [], []
    targets = j(CFG)['targets']
    for seed in SEEDS:
        expert = j(ROOT/f'runs/round31_fascicle_guided_pair_v1/evaluation/{seed}/external34.json')
        expert = {x['image_id']:x['reference'] for x in expert['rows']}
        for kind in KINDS:
            cohort = defaultdict(list)
            for r in dataset:
                k = key(r)
                replayed = by_key[(k, seed)]
                original = replayed['variants']['identity'][kind]
                cohort[r['dataset']].append(dict(image_id=r['image_id'], group_id=r['group_id'],
                    arms=original, reference=expert.get(r['image_id'])))
                if r['dataset'] == 'apo153' and r['image_id'] in refs:
                    ref = refs[r['image_id']]
                    for arm in ARMS:
                        cv = original[arm]['curve']
                        original[arm]['correct'] = bool(f1_score(dict(status='candidate' if cv else 'rejected', curve=cv), ref)[0])
                    if original['legacy']['correct'] != original['separator']['correct']:
                        changes.append(dict(key=k, seed=seed, kind=kind,
                                            **{a:original[a]['correct'] for a in ARMS}))
                span = replayed['roi_width']/(replayed['scale'] or 1)
                rec = dict(key=k, seed=seed, kind=kind, partition=SPLIT['assignment'][k],
                           component=SPLIT['component_ids'][k])
                for arm in ARMS:
                    views = [(tr, replayed['variants'][tr][kind][arm]) for tr in PANEL]
                    rec[arm] = {m:status(m, original[arm], views, span, targets) for m in METRICS}
                stability.append(rec)
            evaluations[f'{seed}|{kind}'] = {
                ds:{arm:dict(n=len(rr), complete=sum(all(r['arms'][arm]['values'][m] is not None for m in METRICS) for r in rr),
                    pair_changes=sum(r['arms']['legacy']['selected_pair_id'] != r['arms']['separator']['selected_pair_id'] for r in rr),
                    **(dict(correct=sum(r['arms'][arm].get('correct', False) for r in rr), annotated=len(refs)) if ds == 'apo153' else {}),
                    **(dict(errors=measurement_errors(rr, arm)) if ds == 'external34' else {})) for arm in ARMS}
                for ds, rr in cohort.items()}

    selected = {}
    for category in ('severe', 'controls'):
        selected[category] = []
        for e in manifest[category]:
            seed, kind = e['condition'].split('|')
            replayed = by_key[(e['key'], int(seed))]['variants']
            orig, moved = replayed['identity'][kind], replayed[e['transform']][kind]
            rec = {**e, 'arms':{arm:{m:error(m, orig[arm]['values'][m], moved[arm]['values'][m])
                                    for m in METRICS} for arm in ARMS}}
            check_equal(rec['arms']['legacy']['FL'], e['original_difference'], f"Round51 event {e['key']}")
            selected[category].append(rec)
    pairs = []
    for pair in gzread(NATURAL):
        if not pair['arm'].endswith('|R'):
            continue
        seed, kind = pair['boundary_seed'], pair['arm'].split('|')[0]
        a, b = [by_key[(pair[k], seed)]['variants']['identity'][kind] for k in ('a', 'b')]
        pairs.append(dict(a=pair['a'], b=pair['b'], strict=pair['strict'], kind=kind,
                          **{arm:{m:error(m, a[arm]['values'][m], b[arm]['values'][m])
                                  for m in METRICS} for arm in ARMS}))
    assert len(pairs) == 117 * len(KINDS) * len(SEEDS)
    natural = {label:{kind:{m:{arm:summary([p[arm][m] for p in pairs if p['kind'] == kind
                    and (label == 'all' or p['strict']) and p[arm][m] is not None]) for arm in ARMS}
                    for m in METRICS} for kind in KINDS} for label in ('all', 'strict')}
    return dict(original=evaluations, reference_unclear=len(unclear), reference_transitions=changes,
                fixed_panel=summarize_stability(stability), natural=natural, events=selected)


def main(workers, pilot, quality_field):
    freeze(quality_field)
    manifest = j(EVENTS)
    extra = defaultdict(set)
    for e in manifest['severe'] + manifest['controls']:
        extra[(e['key'], int(e['condition'].split('|')[0]))].add(e['transform'])
    dataset = rows()
    if pilot:
        dataset = [r for r in dataset if r['image_id'] in ('image_2576.tif', 'image_0174.tif', 'im_06_arch')]
    tasks = [(r, seed, ('identity', *sorted(set(PANEL) | extra[(key(r), seed)])), quality_field)
             for r in dataset for seed in SEEDS]
    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        for f in as_completed([pool.submit(replay, *args) for args in tasks]):
            results.append(f.result())
            if len(results) % 25 == 0:
                print(f'Round52 {len(results)}/{len(tasks)} {time.time()-start:.0f}s', flush=True)
    results.sort(key=lambda r:(r['key'], r['seed']))
    gzwrite(DEST/('pilot.json.gz' if pilot else 'replay.json.gz'), results)
    if not pilot:
        result = comparisons(results, manifest)
        result['replay_sha256'] = sha(DEST/'replay.json.gz')
        write(DEST/'summary.json', result)
    print(f'Round52 complete: {len(results)} image-seed replays in {time.time()-start:.0f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--conservative', action='store_true')
    args = parser.parse_args()
    quality_field = 'single_track_quality' if args.conservative else 'quality'
    if args.conservative:
        DEST = DEST/'conservative'
    if args.evaluate_only:
        replay_file = DEST/'replay.json.gz'
        result = comparisons(gzread(replay_file), j(EVENTS))
        result['replay_sha256'] = sha(replay_file)
        write(DEST/'summary.json', result)
    else:
        main(args.workers, args.pilot, quality_field)
