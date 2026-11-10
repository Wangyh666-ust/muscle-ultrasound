"""Experimental boundary ranking using evidence from a single inner track."""
from scripts.round36_core import choose
from scripts.round46_decoder import POOL46, SOURCE46, core


def separator_evidence(points, tracks, pitch, coverage):
    """Use actual observations on one track, with the existing small-gap tolerance."""
    xs = [x for x, _, _ in points]
    shared = coverage(xs, pitch)
    longest = 0.
    strongest = 0.
    for track in tracks:
        current = 0.
        previous = None
        observed = []
        for x, upper, lower in points:
            y = track.get(x)
            if y is None or not upper + 3 < y < lower - 3:
                continue
            if previous is None or x - previous - pitch > 2 + 1e-8:
                current = pitch
            else:
                current += min(x - previous, pitch)
            previous = x
            longest = max(longest, current)
            observed.append(x)
        strongest = max(strongest, coverage(observed, pitch))
    fraction = min(longest / shared, 1.) if shared else 0.
    return dict(shared=shared, longest=longest, fraction=fraction, penalty=fraction ** 2,
                strongest=strongest, strongest_fraction=min(strongest / shared, 1.) if shared else 0.)


def _build():
    # Keep Round46 generation, qualification and dedup representatives frozen.
    source = SOURCE46
    edits = (
        (
            "        pools[d].append(dict(pair_id=len(pools[d]),curve=cv,quality=qual,points=pts,own_observations=own))",
            "        evidence=separator_evidence(pts,maps[0] if d=='D0' else maps[1]+maps[2],pitch,coverage)\n"
            "        evidence.update(legacy_inside=coverage(inside,pitch),quality=(evidence['shared']/w)*(1-evidence['penalty']))\n"
            "        evidence['single_track_quality']=(evidence['shared']-evidence['strongest'])/w\n"
            "        pools[d].append(dict(pair_id=len(pools[d]),curve=cv,quality=qual,points=pts,own_observations=own,separator=evidence))",
        ),
        (
            "        p['ranking_height']=y1-y0",
            "        p['ranking_height']=y1-y0\n"
            "        p['separator']=pools[p['source_decoder']][p['source_pair_id']]['separator']",
        ),
    )
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    ns = {**POOL46.__globals__, 'separator_evidence': separator_evidence}
    exec(compile(source, '<round52_layer_scoring>', 'exec'), ns)
    return ns['normalized_pool_from_columns']


_pool = _build()


def scored_pool(masks, roi, width):
    columns = [list(enumerate(core.run_centres_per_column(mask))) for mask in masks]
    return _pool(columns, roi, independent=False, reference_width=width)[0]


def rank_layers(pool, gates, native, *, quality_field=None):
    by_id = {p['pair_id']: p for p in pool}
    scores = []
    for gate in gates:
        if not gate['structural_allowed']:
            continue
        pair_id = gate['pair_id']
        score = (by_id[pair_id]['separator'][quality_field] * gate['soft_factor']
                 if quality_field is not None else gate['soft_score'])
        scores.append(dict(pair_id=pair_id, old_score=score))
    return choose(pool, scores, {**native, 'height': pool[0]['ranking_height']}) if scores else None
