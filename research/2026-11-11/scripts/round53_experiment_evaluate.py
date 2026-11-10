"""Freeze development decisions, then score the locked all-view comparison."""
from collections import Counter, defaultdict

from scripts.round53_experiment import DEST, GROUPS, KINDS, SEEDS, SPLIT, NATURAL, METRICS, TRANSFORMS
from scripts.round52_layer_replay import ROOT, CFG, gzread, gzwrite, j, key, rows, sha, write
from scripts.round48_evaluate import error, status, paired_ci, summary
from scripts.round25_evaluate import REF, f1_score, validate_reference
from scripts.round36_evaluate import measurement_errors


def case_report(records):
    out = {}
    for label in ('severe','controls'):
        rr = [r for r in records if r['cohort']==label]
        modes = sorted(set(g for r in rr for g in r['differences']))
        out[label] = {g:dict(n=len(v),severe=sum(x is not None and x>.2 for x in v),
                             missing=sum(x is None for x in v))
                      for g in modes if (v:=[r['differences'][g] for r in rr if g in r['differences']])}
    return out


def stats(records,groups):
    out = {}
    for g in groups:
        q = [r for r in records if g in r['arms']]
        out[g] = dict(n=len(q),
            metrics={m:dict(**{f:sum(r['arms'][g][m][f] for r in q) for f in ('passed','missing','severe')},
                    new_missing=sum(r['arms'][g][m]['missing'] and not r['arms']['A'][m]['missing'] for r in q),
                    recover_missing=sum(not r['arms'][g][m]['missing'] and r['arms']['A'][m]['missing'] for r in q),
                    rescued_severe=sum(r['arms']['A'][m]['severe'] and not r['arms'][g][m]['severe'] and not r['arms'][g][m]['missing'] for r in q),
                    new_severe=sum(r['arms'][g][m]['severe'] and not r['arms']['A'][m]['severe'] for r in q),
                    common_p95=summary([r['arms'][g][m]['worst'] for r in q if not r['arms'][g][m]['missing'] and not r['arms']['A'][m]['missing']])['p95']) for m in METRICS},
            reference_n=sum(r['reference'] is not None for r in q),
            reference_correct=sum(bool(r['reference'] and r['reference'].get(g,False)) for r in q),
            reference_lost=sum(bool(r['reference'] and r['reference'].get('A') and not r['reference'].get(g)) for r in q),
            reference_gained=sum(bool(r['reference'] and not r['reference'].get('A') and r['reference'].get(g)) for r in q),
            fl_bad_delta=paired_ci([(r['component'],int(r['arms'][g]['FL']['missing'] or r['arms'][g]['FL']['severe'])-
                                   int(r['arms']['A']['FL']['missing'] or r['arms']['A']['FL']['severe'])) for r in q]))
    return out


def evaluate(paths,groups,phase):
    index = {key(r):r for r in rows()}
    refs,_ = validate_reference(REF,{r['image_id']:r for r in index.values() if r['dataset']=='apo153'})
    target = j(CFG)['targets']
    severe_ids = gzread(DEST/'events.json.gz')
    desired = defaultdict(list)
    for e in severe_ids:
        seed,kind = e['condition'].split('|')
        desired[e['key'],int(seed),kind].append(e)
    outcomes = {}
    identities = {}
    selected_cases = {}
    for p in paths:
        r = gzread(p)
        k,seed,v = r['key'],r['seed'],r['variants']
        assert tuple(v)==TRANSFORMS
        anchor = v['identity']
        info = index[k]
        gt = refs.get(info['image_id']) if info['dataset']=='apo153' else None
        span = r['roi_width']/(r['scale'] or 1.)
        for kind in KINDS:
            ident = k,seed,kind
            aa = anchor[kind]
            active = [g for g in groups if g in aa]
            records = outcomes.setdefault(ident,dict(key=k,seed=seed,kind=kind,component=SPLIT['component_ids'][k],
                                       partition=SPLIT['assignment'][k],arms={},reference={} if gt is not None else None))
            identities.setdefault(ident,{}).update({g:aa[g]['values'] for g in active})
            for g in active:
                turns = [(tr,v[tr][kind][g]) for tr in TRANSFORMS[1:]]
                records['arms'][g] = {m:status(m,aa[g],turns,span,target) for m in METRICS}
                if gt is not None:
                    cv = aa[g]['curve']
                    records['reference'][g] = bool(f1_score(dict(status='candidate' if cv else 'rejected',curve=cv),gt)[0])
            for e in desired.get(ident,()):
                tr = e['transform']
                rec = selected_cases.setdefault((*ident,tr),dict(key=k,seed=seed,kind=kind,transform=tr,cohort=e['cohort'],differences={}))
                rec['differences'].update({g:error('FL',aa[g]['values']['FL'],v[tr][kind][g]['values']['FL']) for g in active})
    rr = list(outcomes.values())
    assert len(rr)==(138 if phase=='development' else 469)*len(SEEDS)*len(KINDS)
    assert all(set(r['arms'])==set(groups) for r in rr)
    scopes = {'development':rr} if phase=='development' else {
        'all':rr,'confirmation':[r for r in rr if r['partition']=='confirmation']}
    result = dict(cohorts={}, cases=case_report(list(selected_cases.values())))
    for name,cohort in scopes.items():
        breakdown = {'pooled':cohort}
        breakdown.update({str(s):[r for r in cohort if r['seed']==s] for s in SEEDS})
        breakdown.update({f'{s}|{k}':[r for r in cohort if r['seed']==s and r['kind']==k] for s in SEEDS for k in KINDS})
        result['cohorts'][name] = {k:stats(v,groups) for k,v in breakdown.items()}
    if phase=='full':
        assert len(selected_cases)==140
        assert result['cases']['severe']['A']==dict(n=102,severe=102,missing=0)
        result['external'] = external(identities,index,groups)
        result['natural'] = natural(identities,groups)
    write(DEST/f'{phase}_summary.json',result)
    gzwrite(DEST/f'{phase}_conditions.json.gz',rr)
    gzwrite(DEST/f'{phase}_events.json.gz',list(selected_cases.values()))
    return result


def external(identities,index,groups):
    out = {}
    for seed in SEEDS:
        ep = ROOT/f'runs/round31_fascicle_guided_pair_v1/evaluation/{seed}/external34.json'
        experts = {r['image_id']:r['reference'] for r in j(ep)['rows']}
        for kind in KINDS:
            subset = [dict(image_id=r['image_id'],group_id=r['group_id'],reference=experts[r['image_id']],
                arms={g:dict(values=v) for g,v in identities[(k,seed,kind)].items()}) for k,r in index.items() if r['dataset']=='external34']
            assert len(subset)==34
            out[f'{seed}|{kind}'] = {g:measurement_errors(subset,g) for g in groups}
    return out


def natural(identities,groups):
    data = defaultdict(list)
    for pair in gzread(NATURAL):
        if not pair['arm'].endswith('|R'):
            continue
        seed,kind = pair['boundary_seed'],pair['arm'].split('|')[0]
        a,b = [identities[(pair[k],seed,kind)] for k in ('a','b')]
        rec = {g:{m:error(m,a[g][m],b[g][m]) for m in METRICS} for g in groups}
        data[seed,kind].append((pair['strict'],rec))
    return {f'{s}|{k}':{label:{g:{m:dict(**summary(v),missing=sum(z is None for z in v),
                   severe=sum(z is not None and z>(3 if m=='PA' else .2) for z in v))
                   for m in METRICS if (v:=[r[g][m] for strict,r in q if strict or label=='all'])}
                   for g in groups} for label in ('all','strict')} for (s,k),q in data.items()}


def lock_development(digest,protocol,path):
    data = digest['cohorts']['development']
    baseline = data['pooled']['A']
    checks = {}
    for g in GROUPS[1:]:
        q = data['pooled'][g]
        fl = q['metrics']['FL']
        checks[g] = dict(improves_severe=fl['severe']<baseline['metrics']['FL']['severe'],
                         each_seed_safe=all(data[str(s)][g]['metrics']['FL']['severe']<=data[str(s)]['A']['metrics']['FL']['severe'] for s in SEEDS),
                         no_new_missing=fl['new_missing']==0,
                         reference_safe=q['reference_correct']>=baseline['reference_correct'])
    qualified = [g for g in ('Q25','Q20','Qlocal') if all(checks[g].values())]
    selected = min(qualified,key=lambda g:(data['pooled'][g]['metrics']['FL']['severe'],GROUPS.index(g))) if qualified else None
    rank_pass = all(checks['R'].values())
    combine = bool(selected and rank_pass)
    result = dict(protocol_sha256=sha(protocol),development_sha256=sha(DEST/'development_summary.json'),
                  checks=checks,qualification=selected,ranking_pass=rank_pass,combination_allowed=combine,
                  full_groups=list(GROUPS)+(['QR'] if combine else []))
    if path.exists():
        assert j(path)==result
    else:
        write(path,result)
    print('R53 lock',result,flush=True)
