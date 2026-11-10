"""Evaluation of the locked endpoint rules, with recovery denominators explicit."""
from collections import Counter, defaultdict

from scripts.round54_fl_replay import (
    DEST, R53, ROOT, POLICIES, TRANSFORMS, DIAG_IMAGES, KINDS, SEEDS, SPLIT, CFG, np,
    audit_path, source, rows, key, j, gzread, gzwrite, write, sha, error, policy_values, ref_index,
)
from scripts.round53_experiment_evaluate import stats, external, natural
from scripts.round48_evaluate import status, summary


METRICS = ('MT','PA','FL')


def expert_refs():
    return {s:{r['image_id']:r['reference']['FL'] for r in j(ROOT/f'runs/round31_fascicle_guided_pair_v1/evaluation/{s}/external34.json')['rows']} for s in SEEDS}


def recovery_accuracy(records, groups):
    out = {}
    for g in groups:
        eligible = [r for r in records if r['values'][g] is not None and r['ref'] is not None]
        by = defaultdict(list)
        for r in eligible:
            by[r['key']].append(r)
        diff = [r['values'][g]-r['ref']['mean'] for r in eligible]
        out[g] = dict(total_opportunities=len(records),opportunity_images=len({r['key'] for r in records}),
             output=sum(r['values'][g] is not None for r in records),measured_events=len(diff),independent_images=len(by),
             MAE=float(np.mean(np.abs(diff))) if diff else None,bias=float(np.mean(diff)) if diff else None,
             image_equal_MAE=float(np.mean([np.mean([abs(r['values'][g]-r['ref']['mean']) for r in z]) for z in by.values()])) if by else None,
             worst=float(max(abs(x) for x in diff)) if diff else None,within_12mm=sum(abs(x)<=12 for x in diff),
             within_expert_range=sum(r['ref']['min']<=r['values'][g]<=r['ref']['max'] for r in eligible),
             median_reference_MAE=float(np.mean([abs(r['values'][g]-r['ref']['median']) for r in eligible])) if eligible else None)
    return out


def evidence_summary(tracks):
    pairs = ('sup','deep')
    result = {'n':len(tracks)}
    for e in (.02,.05):
        mass = {s:sum(t['weight'] for t in tracks if t['endpoints'][s]['distance'] is not None and t['endpoints'][s]['distance']<=e) for s in pairs}
        mass['both'] = sum(t['weight'] for t in tracks if all(t['endpoints'][s]['distance'] is not None and t['endpoints'][s]['distance']<=e for s in pairs))
        result[str(e)] = mass
    result['weight_any_ray'] = sum(t['weight'] for t in tracks if any(t['endpoints'][s]['ray'] for s in pairs))
    result['weight_over80'] = sum(t['weight'] for t in tracks if t['extrap']>.80)
    result['weight_both_risk'] = sum(t['weight'] for t in tracks if t['risk']['reliable'] and t['risk']['root_switches']==0)
    result['weighted_along_extrapolation'] = sum(t['weight']*t['extrap'] for t in tracks)
    return result


def evaluate(lock):
    dataset = rows()
    index = {key(r):r for r in dataset}
    experts = expert_refs()
    annotated = ref_index()
    conditions, recoveries, all_changes, references = [], [], [], []
    external_recoveries = []
    originals = {}
    reg = {(e['key'],e['seed'],e['kind'],e['transform']):e['cohort'] for e in gzread(R53/'full_events.json.gz')}
    checked = {}
    counts = defaultdict(Counter)
    target = j(CFG)['targets']
    for r in dataset:
        k = key(r)
        part = SPLIT['assignment'][k]
        for seed in SEEDS:
            frozen = gzread(source(r,seed))
            observed = gzread(audit_path(r,seed))['variants']
            vv = frozen['variants']
            assert tuple(vv)==TRANSFORMS
            for kind in KINDS:
                views = {}
                ref = {g:False for g in POLICIES} if r['dataset']=='apo153' and r['image_id'] in annotated else None
                for tr in TRANSFORMS:
                    a,b = [vv[tr][kind][g] for g in ('A','Qlocal')]
                    ev = observed.get(tr,{}).get(kind)
                    fl = policy_values(a,b,ev,lock)
                    assert fl['A']==a['values']['FL'] and fl['B']==b['values']['FL']
                    outs = {}
                    for g in POLICIES:
                        q = {**(a if g=='A' else b)['values'],'FL':fl[g]}
                        outs[g] = dict(values=q)
                        ct = counts[g]
                        ct['views'] += 1
                        ct['recover_FL'] += fl['A'] is None and fl[g] is not None
                        ct['lose_FL'] += fl['A'] is not None and fl[g] is None
                        ct['replace_FL'] += fl['A'] is not None and fl[g] is not None and fl['A']!=fl[g]
                        ct['exploratory_only'] += g in ('S','SR','X') and fl['A'] is None and fl['B'] is not None and fl[g] is None
                        ct['mixed_boundary_sources'] += g not in ('A','B') and fl[g] is not None and fl[g]==fl['A'] and a['selected_pair_id']!=b['selected_pair_id']
                    views[tr] = outs
                    ident = k,seed,kind,tr
                    if tr=='identity' and ev is not None and ref is not None:
                        flags = {g:z and z.get('reference_correct') for g,z in ev.items() if g in ('A','Qlocal')}
                        if any(x is not None for x in flags.values()):
                            ref = {g:bool(flags.get('A' if g=='A' else 'Qlocal')) for g in POLICIES}
                    changed = fl['A']!=fl['B']
                    if changed or ident in reg or r['image_id'] in DIAG_IMAGES and tr in ('identity','r41_scale1015','translate-3_-2','scale095'):
                        info = {g:evidence_summary(ev[g]['tracks']) for g in ('A','Qlocal') if ev and ev[g] and ev[g]['tracks']}
                        z = dict(key=k,seed=seed,kind=kind,transform=tr,partition=part,values=fl,
                                 pair_ids={g:x['selected_pair_id'] for g,x in (('A',a),('Qlocal',b))},evidence=info,
                                 ref=experts[seed].get(r['image_id']) if r['dataset']=='external34' else None)
                        if changed:
                            all_changes.append(z)
                        if fl['A'] is None and fl['B'] is not None:
                            assert ev is not None and 'Qlocal' in info
                            recoveries.append(z)
                            if z['ref'] is not None:
                                external_recoveries.append(z)
                        if ident in reg:
                            checked[ident] = {**z,'cohort':reg[ident]}
                    if ev:
                        for g in ('A','Qlocal'):
                            if ev[g] and ev[g].get('reference_endpoints'):
                                references.append(dict(key=k,seed=seed,kind=kind,transform=tr,arm=g,correct=ev[g]['reference_correct'],endpoints=ev[g]['reference_endpoints']))
                anchor = views['identity']
                originals[k,seed,kind] = {g:anchor[g]['values'] for g in POLICIES}
                span = frozen['roi_width']/(frozen['scale'] or 1.)
                rec = dict(key=k,seed=seed,kind=kind,partition=part,component=SPLIT['component_ids'][k],reference=ref,arms={})
                for g in POLICIES:
                    pairs = [(tr,views[tr][g]) for tr in TRANSFORMS[1:]]
                    rec['arms'][g] = {m:status(m,anchor[g],pairs,span,target) for m in METRICS}
                conditions.append(rec)
    assert len(conditions)==1876 and all(q['views']==50652 for q in counts.values())
    assert len(recoveries)==186 and len(external_recoveries)==8 and len(checked)==140
    for g in ('P','S','SR','F80','X'):
        assert counts[g]['lose_FL']==0
    for g in ('P','S','SR','F80'):
        assert counts[g]['replace_FL']==0
    out = dict(counts=dict(counts),cohorts={},external=external(originals,index,POLICIES),
               natural=natural(originals,POLICIES),external_recovery=recovery_accuracy(external_recoveries,POLICIES),
               audited_human_endpoints=len(references),lock=sha(DEST/'lock.json'))
    for label in ('all','confirmation','development'):
        rr = [r for r in conditions if label=='all' or r['partition']==label]
        out['cohorts'][label] = dict(pooled=stats(rr,POLICIES),
            **{str(s):stats([r for r in rr if r['seed']==s],POLICIES) for s in SEEDS},
            **{f'{s}|{k}':stats([r for r in rr if r['seed']==s and r['kind']==k],POLICIES) for s in SEEDS for k in KINDS})
    out['case_counts'] = {}
    for label in ('severe','controls'):
        cc = [r for r in checked.values() if r['cohort']==label]
        out['case_counts'][label] = {}
        for g in POLICIES:
            diffs = [error('FL',originals[r['key'],r['seed'],r['kind']][g]['FL'],r['values'][g]) for r in cc]
            out['case_counts'][label][g] = dict(n=len(cc),severe=sum(d is not None and d>.20 for d in diffs),missing=sum(d is None for d in diffs))
    def summarize_support(records):
        metrics = [r['evidence']['Qlocal'] for r in records if 'Qlocal' in r['evidence']]
        return {f:summary([m[f] for m in metrics]) for f in ('weight_any_ray','weight_over80','weighted_along_extrapolation','weight_both_risk')} | {
            f'weight_both_{z}':summary([m[z]['both'] for m in metrics]) for z in ('0.02','0.05') if metrics and z in metrics[0]}
    out['recovered_support'] = summarize_support(recoveries)
    write(DEST/'summary.json',out)
    gzwrite(DEST/'conditions.json.gz',conditions)
    gzwrite(DEST/'recoveries.json.gz',recoveries)
    gzwrite(DEST/'changed_outputs.json.gz',all_changes)
    gzwrite(DEST/'reference_endpoints.json.gz',references)
    gzwrite(DEST/'historical_cases.json.gz',list(checked.values()))
    return out
