"""Evaluate the frozen Round55 comparisons and lock development choices."""
from collections import Counter, defaultdict

from scripts.round55_replay import (
    DEST, ROOT, KINDS, SEEDS, SPLIT, R53, MUSCLE, BOUNDARY, SIDES, CASES,
    rows, key, ref_index, settings, j, sha, write, gzread, gzwrite, np,
)
from scripts.round53_experiment_evaluate import stats, external, natural
from scripts.round54_fl_evaluate import expert_refs, recovery_accuracy
from scripts.round48_evaluate import status, summary, error
from scripts.round25_evaluate import f1_score


def boundary_reference(curve,ref):
    ok,q,err = f1_score(dict(status='candidate' if curve else 'rejected',curve=curve),ref)
    out = {'correct':bool(ok)}
    if err is not None:
        x = np.asarray(ref['reference_x'])
        lo,hi = curve['support']
        terminal = ((x<=lo+.15*(hi-lo))|(x>=hi-.15*(hi-lo)))&err['valid_mask']
        for s,k in (('sup','se'),('deep','de')):
            out[s] = dict(coverage=q[s+'_coverage'],end_error=float(np.mean(err[k][terminal])) if np.any(terminal) else None)
    return out


def evaluate(paths,phase,groups,selected):
    index = {key(r):r for r in rows()}
    refs = ref_index()
    experts = expert_refs()
    targets = settings()[0]['targets']
    labels = (*groups,*MUSCLE) if phase in ('muscle','full') else groups
    conditions, diagnostics, ref_rows, cases, recovered = [],[],[],[],[]
    identities = {}
    counts = defaultdict(Counter)
    pair_cases = {(r['key'],r['seed'],r['kind'],r['transform']):r['cohort'] for r in gzread(R53/'full_events.json.gz')}
    for p in sorted(paths):
        r = gzread(p)
        k,seed,v = r['key'],r['seed'],r['variants']
        row = index[k]
        part = SPLIT['assignment'][k]
        ref = refs.get(row['image_id']) if row['dataset']=='apo153' else None
        span = r['roi_width']/(r['scale'] or 1.)
        for kind in KINDS:
            aa = v['identity'][kind]
            identities[k,seed,kind] = {g:aa[g]['values'] for g in labels}
            record = dict(key=k,seed=seed,kind=kind,partition=part,component=SPLIT['component_ids'][k],arms={},reference=None)
            if ref is not None:
                records = {g:boundary_reference(aa[selected if g in MUSCLE else g].get('curve'),ref) for g in labels if g!='Q'}
                record['reference'] = {g:z['correct'] for g,z in records.items()}
                ref_rows.append(dict(key=k,seed=seed,kind=kind,part=part,records=records))
            for g in labels:
                vv = [(tr,z[kind][g]) for tr,z in v.items() if tr!='identity']
                record['arms'][g] = {m:status(m,aa[g],vv,span,targets) for m in ('MT','PA','FL')}
            conditions.append(record)
            for tr,b in v.items():
                a = b[kind]
                cohort = pair_cases.get((k,seed,kind,tr))
                if cohort or row['image_id'] in CASES:
                    cases.append(dict(key=k,seed=seed,kind=kind,tr=tr,cohort=cohort,
                        values={g:a[g]['values'] for g in labels},differences={g:error('FL',aa[g]['values']['FL'],a[g]['values']['FL']) for g in labels},
                        diagnostic=r['diagnostics'].get(f'{tr}|{kind}')))
                if row['dataset']=='external34' and a['A']['values']['FL'] is None:
                    recovered.append(dict(key=k,seed=seed,kind=kind,tr=tr,ref=experts[seed][row['image_id']],values={g:a[g]['values']['FL'] for g in labels}))
                for g in labels:
                    q = counts[g]
                    z = a[g]
                    q['views'] += 1
                    q['changed_geometry'] += bool(z.get('changed'))
                    av,bv = a['A']['values']['FL'],z['values']['FL']
                    q['new_missing'] += av is not None and bv is None
                    q['recovered'] += av is None and bv is not None
                    q['replaced'] += av is not None and bv is not None and abs(av-bv)>1e-8
                    if g in MUSCLE and 'far' in z:
                        q['contribution_views'] += 1
                        for f in ('center','far','effective','field_fallback'):
                            q[f] += z[f]
                    if tr!='identity':
                        geom = a[selected if g in MUSCLE else g].get('same_layer',{})
                        for s in SIDES:
                            q[s+'_comparable'] += geom.get(s) is not None
                            q[s+'_changed_layer'] += geom.get(s) is False
                        q['both_changed_layer'] += all(geom.get(s) is False for s in SIDES)
    assert len(conditions)==(469 if phase=='full' else 138)*4
    out = dict(counts=dict(counts),cohorts={},external=external(identities,index,labels) if phase=='full' else None,
               natural=natural(identities,labels) if phase=='full' else None,
               external_recovery=recovery_accuracy(recovered,labels))
    for scope in ('all','development','confirmation'):
        subset = [r for r in conditions if scope=='all' or r['partition']==scope]
        if not subset:
            continue
        out['cohorts'][scope] = dict(pooled=stats(subset,labels),**{str(s):stats([r for r in subset if r['seed']==s],labels) for s in SEEDS},
            **{f'{s}|{k}':stats([r for r in subset if r['seed']==s and r['kind']==k],labels) for s in SEEDS for k in KINDS})
    out['reference_by_side'] = {}
    for scope in ('all','development','confirmation'):
        q = [r for r in ref_rows if scope=='all' or r['part']==scope]
        out['reference_by_side'][scope] = {g:{s:{m:summary([r['records'][g].get(s,{}).get(m) for r in q]) for m in ('coverage','end_error')} for s in SIDES} for g in labels if g!='Q'}
    out['historical_cases'] = {label:{g:dict(n=len(vals),severe=sum(x is not None and x>.2 for x in vals),missing=vals.count(None))
        for g in labels if (vals:=[r['differences'][g] for r in cases if r['cohort']==label])} for label in ('severe','controls')}
    write(DEST/f'{phase}_summary.json',out)
    gzwrite(DEST/f'{phase}_conditions.json.gz',conditions)
    gzwrite(DEST/f'{phase}_case_details.json.gz',cases)
    return out


def fl_bad(z):
    return z['metrics']['FL']['severe']+z['metrics']['FL']['missing']


def lock_boundaries(result,protocol_id):
    counts = result['cohorts']['development']['pooled']
    sides = result['reference_by_side']['development']
    baseline = counts['A']
    families = defaultdict(list)
    evidence = {}
    for g in BOUNDARY:
        if g=='A':
            continue
        families[g.rsplit('_',1)[0]].append(g)
        z = counts[g]
        end = sum(sides[g][s]['end_error']['mean'] for s in SIDES)
        old_end = sum(sides['A'][s]['end_error']['mean'] for s in SIDES)
        loss = sum(max(0,sides['A'][s]['coverage']['mean']-sides[g][s]['coverage']['mean']) for s in SIDES)
        improved = fl_bad(z)<fl_bad(baseline) or old_end-end>.10
        safe = z['reference_lost']==0 and z['metrics']['FL']['new_missing']==0 and fl_bad(z)<=fl_bad(baseline) and loss<.002
        evidence[g] = dict(improved=improved,safe=safe,end_error=end,ref_coverage_loss=loss,bad=fl_bad(z),changed=result['counts'][g]['changed_geometry'])
    def rank(g):
        e = evidence[g]
        return (not e['safe'],e['bad'],e['end_error'],g)
    best = {f:min(gs,key=rank) for f,gs in families.items()}
    qualified = [g for g in best.values() if evidence[g]['safe'] and evidence[g]['improved']]
    selected = min(qualified,key=rank) if qualified else 'A'
    p = DEST/'boundary_lock.json'
    q = dict(protocol=protocol_id,selected=selected,family_best=best,evidence=evidence)
    if p.exists():
        assert j(p)==q
    else:
        write(p,q)
    print('Round55 boundary lock',selected,best,flush=True)


def lock_muscles(result):
    counts = result['cohorts']['development']['pooled']
    a = counts['M0']['metrics']['FL']
    good = [g for g in MUSCLE[1:] if counts[g]['metrics']['FL']['severe']<=a['severe'] and
         counts[g]['metrics']['FL']['missing']<=a['missing'] and
         (counts[g]['metrics']['FL']['severe']<a['severe'] or counts[g]['metrics']['FL']['common_p95']<a['common_p95']-.002)]
    selected = min(good,key=lambda g:(counts[g]['metrics']['FL']['severe'],counts[g]['metrics']['FL']['common_p95'],g)) if good else 'M0'
    p = DEST/'muscle_lock.json'
    q = dict(boundary_lock=sha(DEST/'boundary_lock.json'),selected=selected,comparisons=list(MUSCLE))
    if p.exists():
        assert j(p)==q
    else:
        write(p,q)
    print('Round55 muscle lock',selected,flush=True)
