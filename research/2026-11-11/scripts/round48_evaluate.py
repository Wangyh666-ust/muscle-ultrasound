"""Paired evaluation and independent reference checks for Round48."""
from collections import Counter, defaultdict
import argparse
from scripts.round48_length_replay import (
    DEST, ROOT, CFG, R46, R40, R41, SEEDS, KINDS, SCOPES, CASE_KEYS, SPLIT, rows, key, j, np,
    source_path, output_path, read_mask, extract, measure, reduced, initialize, write, gzread, gzwrite, sha, Path,
)
from scripts.round32_common import map_curve
from scripts.round38_evaluate import relative
from scripts.fascicle_geometry import extract_segments, match_segments
from src.umud_baseline.data import UmudTrainData, read_mask as load_label, align_mask_to_image

R47 = ROOT/'runs/round47_boundary_selection_v1'
METRICS = ('PA', 'FL')
ARMS = ('base', 'length')


def lock():
    p = DEST/'lock.json'
    protocol = j(DEST/'protocol.json')
    for path, digest in protocol['protected'].items():
        assert sha(ROOT/path) == digest, path
    pilot = j(DEST/'pilot_success.json')
    prob = j(DEST/'probability_audit.json')
    assert pilot['rows'] == 12 and prob['total'] == len(protocol['probability']['keys'])*6
    assert prob['exact'] >= .90*prob['total'], prob['exact']
    files = [*sorted((ROOT/'scripts').glob('round48_*.py')), CFG, R40/'calibration.json',
             R41/'partitions.json', R46/'lock.json', R47/'lock.json',
             R47/'evaluation/natural_pairs.json.gz', DEST/'protocol.json',
             DEST/'pilot_success.json', DEST/'probability_audit.json', DEST/'probability_traces.json.gz']
    record = dict(length_ref=protocol['length_ref'], baseline_replays=sum(r['exact_baseline_replays'] for r in pilot['records']),
                  probability_pairs=prob['exact'], probability_total=prob['total'],
                  sources={f.relative_to(ROOT).as_posix():sha(f) for f in files})
    if p.exists():
        assert j(p) == record
    else:
        write(p, record)
    print('Round48 locked',record['baseline_replays'],record['probability_pairs'],flush=True)


def summary(values):
    v = np.asarray([x for x in values if x is not None], float)
    return dict(n=len(v), median=float(np.median(v)) if len(v) else None,
                mean=float(v.mean()) if len(v) else None, p95=float(np.quantile(v,.95)) if len(v) else None)


def paired_ci(records):
    grouped = defaultdict(list)
    for component, diff in records:
        if diff is not None:
            grouped[component].append(diff)
    if not grouped:
        return dict(delta=None, ci95=None, components=0)
    totals = np.array([[sum(v),len(v)] for v in grouped.values()],float)
    rng = np.random.default_rng(2026110548)
    draw = totals[rng.integers(0,len(totals),size=(5000,len(totals)))].sum(1)
    return dict(delta=float(totals[:,0].sum()/totals[:,1].sum()),components=len(totals),
                ci95=np.quantile(draw[:,0]/draw[:,1],[.025,.975]).tolist())


def error(metric, a, b):
    if a is None or b is None:
        return None
    return abs(a-b) if metric == 'PA' else relative(a,b)


def status(metric, base, variants, span, targets):
    values = [(tr,v['values'][metric]) for tr,v in variants]
    x = base['values'][metric]
    diffs = [(tr,d) for tr,y in values if (d := error(metric,x,y)) is not None]
    worst_at, worst = max(diffs,key=lambda q:q[1],default=(None,None))
    absolute = max((abs(y-x)/span for tr,y in values if y is not None and x is not None),default=None)
    missing = x is None or any(y is None for tr,y in values)
    limit = targets['PA_deg'] if metric=='PA' else targets['FL_relative']
    severe = targets['severe_PA_deg'] if metric=='PA' else targets['severe_relative']
    return dict(missing=missing, identity_missing=x is None, worst=worst, worst_at=worst_at,
                severe=worst is not None and worst>severe,
                passed=not missing and worst is not None and worst<=limit
                and (metric=='PA' or absolute<=targets['FL_ROI_absolute']))


def per_image(r, scope, names, targets):
    variants=r['variants']
    span = r['roi_width']/(r['scale'] or 1)
    result=[]
    for cond, orig in variants['identity'].items():
        turns = [(t,variants[t][cond]) for t in names if t in variants]
        row = dict(key=r['key'], partition=r['partition'], condition=cond, scope=scope,
                   component=SPLIT['component_ids'][r['key']], transforms=len(turns), arms={})
        for arm in ARMS:
            selected = [(t,q[arm]) for t,q in turns]
            row['arms'][arm] = {m:status(m,orig[arm],selected,span,targets) for m in METRICS}
            row['arms'][arm]['joint'] = all(row['arms'][arm][m]['passed'] for m in METRICS)
        result.append(row)
    return result


def scope_report(data):
    out={}
    for cohort in ('all','development','confirmation'):
        out[cohort]={}
        for scope in ('combined16','all26'):
            out[cohort][scope]={}
            for kind in KINDS:
                rr=[r for r in data if (cohort=='all' or r['partition']==cohort)
                    and r['scope']==scope and r['condition'].endswith('|'+kind)]
                q=dict(n=len(rr),arms={})
                for arm in ARMS:
                    q['arms'][arm]={m:{f:sum(r['arms'][arm][m][f] for r in rr)
                                       for f in ('passed','missing','identity_missing','severe')} for m in METRICS}
                    q['arms'][arm]['joint']=sum(r['arms'][arm]['joint'] for r in rr)
                q['joint_ci']=paired_ci([(r['component'],int(r['arms']['length']['joint'])-int(r['arms']['base']['joint'])) for r in rr])
                for m in METRICS:
                    common=[r for r in rr if all(not r['arms'][a][m]['missing'] for a in ARMS)]
                    q[m+'_common_worst']={a:summary([r['arms'][a][m]['worst'] for r in common]) for a in ARMS}
                    q[m+'_pass_ci']=paired_ci([(r['component'],int(r['arms']['length'][m]['passed'])-int(r['arms']['base'][m]['passed'])) for r in rr])
                out[cohort][scope][kind]=q
    return out


def natural_report(loaded):
    pairs = [r for r in gzread(R47/'evaluation/natural_pairs.json.gz') if r['arm'].endswith('|R')]
    assert len(pairs)==117*len(KINDS)*len(SEEDS)
    data=[]
    for r in pairs:
        kind=r['arm'].split('|')[0]
        cond=f"{r['boundary_seed']}|{kind}"
        left,right=[loaded[r[k]]['variants']['identity'][cond] for k in ('a','b')]
        rec={k:r[k] for k in ('a','b','group','component','strict','partition')}
        rec.update(condition=cond,kind=kind,arms={})
        for arm in ARMS:
            aa,bb=left[arm],right[arm]
            rec['arms'][arm]={m:error(m,aa['values'][m],bb['values'][m]) for m in METRICS}
            rec['arms'][arm]['count_flip']=aa['count']['FL'] != bb['count']['FL']
        data.append(rec)
    gzwrite(DEST/'natural_pairs.json.gz',data)
    result={}
    for name,scope in [('all',data),('strict',[r for r in data if r['strict']]),
                       ('confirmation',[r for r in data if r['partition']=='confirmation'])]:
        result[name]={}
        for kind in KINDS:
            rr=[r for r in scope if r['kind']==kind]
            out=dict(n=len(rr),arms={a:{'FL_count_flip':sum(r['arms'][a]['count_flip'] for r in rr)} for a in ARMS})
            for m in METRICS:
                common=[r for r in rr if all(r['arms'][a][m] is not None for a in ARMS)]
                for arm in ARMS:
                    out['arms'][arm][m]=dict(
                        missing=sum(r['arms'][arm][m] is None for r in rr),
                        common=summary([r['arms'][arm][m] for r in common]),
                        severe=sum(r['arms'][arm][m] is not None and r['arms'][arm][m]>(3 if m=='PA' else .2) for r in rr))
                out[m+'_ci']=paired_ci([(r['component'],r['arms']['length'][m]-r['arms']['base'][m]) for r in common])
            result[name][kind]=out
    return result


def reference_report(loaded, manifest):
    wanted={r['image_id']:r for r in manifest if r['dataset']=='dev282'}
    samples=UmudTrainData('fasc','dev',include_filenames=wanted).samples
    assert len(samples)==282
    per_image=[]
    for s in samples:
        r=wanted[s['image']['filename']]
        raw=loaded[key(r)]
        for ann in s['masks']:
            mask=align_mask_to_image(load_label(ROOT/ann['path']),(r['width'],r['height']))
            truth=extract_segments(mask)['segments']
            for kind in KINDS:
                pred=raw['original_tracks'][kind]
                base,added=pred['base'],pred['added']
                added_ids={p['label_id'] for p in added}
                opts=dict(perp_tol=10.,angle_tol=10.,overlap_min=.2)
                arms={}
                for arm,segments in [('base',base),('length',base+added)]:
                    matches=match_segments(segments,truth,**opts)['matches']
                    arms[arm]=dict(matched=len(matches),truth=len(truth),
                                   angle=[m['angle_diff_deg'] for m in matches],
                                   added_match=sum(segments[m['pred_index']]['label_id'] in added_ids for m in matches))
                per_image.append(dict(key=key(r),kind=kind,added=len(added),
                                      partition=raw['partition'],annotation=ann['path'],arms=arms))
    gzwrite(DEST/'reference_matches.json.gz',per_image)
    out={}
    for scope in ('all','confirmation'):
        out[scope]={}
        for kind in KINDS:
            subset=[r for r in per_image if r['kind']==kind and (scope=='all' or r['partition']==scope)]
            out[scope][kind]={arm:dict(
                rows=len(subset), total_reference=sum(r['arms'][arm]['truth'] for r in subset),
                matched=sum(r['arms'][arm]['matched'] for r in subset),
                mean_variant_coverage=summary([r['arms'][arm]['matched']/r['arms'][arm]['truth'] for r in subset if r['arms'][arm]['truth']])['mean'],
                added=sum(r['added'] for r in subset), added_matched=sum(r['arms'][arm]['added_match'] for r in subset),
                direction=summary([v for r in subset for v in r['arms'][arm]['angle']])) for arm in ARMS}
    return out


def counts(loaded):
    out={}
    for kind in KINDS:
        for scope in ('identity','all26'):
            rr=[q for r in loaded.values() for tr,v in r['variants'].items()
                if tr=='identity' and scope=='identity' or tr!='identity' and scope=='all26'
                for cond,q in v.items() if cond.endswith('|'+kind)]
            risks=[v for q in rr for v in q['length']['added_risk'].values()]
            out[kind+'|'+scope]=dict(n=len(rr),added_shape=sum(q['added'] for q in rr),
                arms={a:{m:dict(n=sum(q[a]['count'][m] for q in rr),
                               zero=sum(q[a]['count'][m]==0 for q in rr),
                               low=sum(q[a]['count'][m]<=2 for q in rr),
                               median=summary([q[a]['count'][m] for q in rr])['median'])
                         for m in METRICS} for a in ARMS},
                n_changed=sum(any(abs((q['base']['values'][m] or 0)-(q['length']['values'][m] or 0))>1e-8 for m in METRICS) for q in rr),
                n_with_added_FL=sum(q['length']['added_used']['FL']>0 for q in rr),
                added_FL_weight=summary([q['length']['added_weight']['FL'] for q in rr if q['length']['added_used']['FL']]),
                added_FL_risk_probed=len(risks), added_FL_stable_under_probe=sum(x['reliable'] for x in risks))
    return out


def fixed_boundary_panel():
    initialize()
    cfg,cal=j(CFG),j(R40/'calibration.json')
    protocol=j(DEST/'protocol.json')['probability']
    panel=[]
    for r in rows():
        if key(r) not in protocol['keys']:
            continue
        originals={s:gzread(source_path(r,s)) for s in SEEDS}
        row=originals[SEEDS[0]]
        cases={}
        for tr in ('identity',*protocol['transforms']):
            v=row['variants'][tr];t=v['transform'];roi=v['measurement_roi']
            cases[tr]={}
            for kind in KINDS:
                mask,_=read_mask(r,tr,kind,v)
                raw,kept,relaxed,added,_=extract(mask,roi,cfg)
                for seed in SEEDS:
                    orig=originals[seed]['variants']['identity']['arms'][kind+'|ID']
                    curve=map_curve(orig['curve'],t)
                    base=measure(curve,orig['selected_pair_id'],kept,len(raw),roi,t,row,cfg,cal)
                    trial=measure(curve,orig['selected_pair_id'],relaxed,len(raw),roi,t,row,cfg,cal) if added and curve else base
                    cases[tr][f'{seed}|{kind}']=dict(base=reduced(base,detailed=key(r) in CASE_KEYS),
                                                   length=reduced(trial,added,detailed=key(r) in CASE_KEYS))
        rr=dict(key=key(r),partition=SPLIT['assignment'][key(r)],variants=cases,scale=row['px_per_mm'],
                roi_width=row['variants']['identity']['measurement_roi']['box'][2]
                          -row['variants']['identity']['measurement_roi']['box'][0])
        panel.append(rr)
    gzwrite(DEST/'anchored_boundary_panel.json.gz',panel)
    data=[v for r in panel for v in per_image(r,'all26',protocol['transforms'],cfg['targets'])]
    result=scope_report(data)['all']['all26']
    write(DEST/'anchored_boundary_summary.json',result)
    return result


def main():
    manifest=rows()
    assert j(DEST/'replay_success.json')['rows']==len(manifest)==469
    loaded={key(r):gzread(output_path(r)) for r in manifest}
    cfg=j(CFG)
    data=[v for r in loaded.values() for scope in ('combined16','all26')
          for v in per_image(r,scope,SCOPES[scope],cfg['targets'])]
    assert all(v['transforms']==(16 if v['scope']=='combined16' else 26) for v in data)
    gzwrite(DEST/'worst_by_image.json.gz',data)
    stability=scope_report(data)
    write(DEST/'stability.json',stability)
    write(DEST/'natural_summary.json',natural_report(loaded))
    write(DEST/'candidate_summary.json',counts(loaded))
    write(DEST/'reference_summary.json',reference_report(loaded,manifest))
    fixed_boundary_panel()
    write(DEST/'evaluation_success.json',dict(images=len(loaded),measurements=469*27*4*2,
          transforms=27, historical_images=331, development_images=138,
          fixed_boundary_panel=len(j(DEST/'protocol.json')['probability']['keys'])))
    print('Round48 evaluation complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--lock',action='store_true')
    args=parser.parse_args()
    lock() if args.lock else main()
