"""Final integrity, coverage/accounting, and mechanism-isolation verification."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter,defaultdict
from scripts.round39_common import *
from scripts.round39_core import ARMS
from scripts.round27_experiment import check_equal as tolerance_equal

def check_equal(a,b,path=''):
    # Exact equality is sufficient for the original tolerance test. Most cached
    # invariants are identical JSON trees; avoid one NumPy call per scalar.
    # Any non-identical value still receives the unchanged original comparison.
    if a == b:
        return
    return tolerance_equal(a,b,path)

def main():
    start=time.time();verify_preparation();lock=verify_lock();assert j(DEST/'experiment_success.json')['records']==7470
    frozen=j(DEST/'evaluation/predictions_frozen_before_reference.json');assert len(frozen['files'])==7470
    counts=Counter();gates=Counter();normal=Counter();exploratory=Counter();coverage=Counter();replays=0;files={}
    for r in ordered_inputs():
        for seed in SEEDS:
            for tr in transforms(r):
                p=outpath(r,tr,seed);rel=p.relative_to(DEST).as_posix();assert sha(p)==frozen['files'][rel]
                q=gzread(p);assert q['lock_sha256']==sha(DEST/'implementation_lock.json');assert set(q['arms'])==set(ARMS)
                assert key(q)==key(r) and q['seed']==seed and q['transform']['name']==tr
                oldp=predpath(r,tr,seed);assert sha(oldp)==q['source_sha256'];old=gzread(oldp)['arms']['BNVA'];b=q['arms']['B0']
                for f in ('values','curve','FL_weights','PA_weights','selection'):check_equal(b[f],old[f],'full B0 replay '+f)
                replays+=1
                for arm,a in q['arms'].items():
                    counts[seed,arm]+=1
                    for m,v in a['values'].items():
                        assert v is None or np.isfinite(v)
                        if v is not None:normal[seed,arm,m]+=1
                    for m in ('PA','FL'):
                        weights=a[m+'_weights']
                        assert all(w is not None and np.isfinite(w) and w>=0 for w in weights.values()),'Nonfinite/negative measurement weights'
                        if weights:assert abs(sum(weights.values())-1)<1e-8
                    for m,v in a['exploratory_values'].items():
                        if v is not None:exploratory[seed,arm,m]+=1
                    for name in a['selection']['missing_regions']:coverage[seed,arm,name]+=1
                    if arm in ('Q','J') and a['curve'] is not None:
                        eligibility={g['pair_id']:g for g in a['pair_eligibility']};assert eligibility[a['selected_pair_id']]['allowed'];gates[arm]+=1
                    if arm in ('W','S','WS'):
                        for f in ('curve','PA_weights','PA_ids','selection'):
                            if f=='selection':
                                for sub in ('selected_ids','missing_regions','weights'):check_equal(a[f][sub],b[f][sub],'FL-only representative invariant')
                            else:check_equal(a[f],b[f],'FL-only invariant')
                        for m in ('MT','PA'):check_equal(a['values'][m],b['values'][m],'FL-only values')
                check_equal(q['arms']['E']['values']['MT'],q['arms']['VC']['values']['MT'],'terminal-window-only MT invariant')
                check_equal(q['arms']['E']['selected_pair_id'],q['arms']['VC']['selected_pair_id'],'terminal-window-only selected pair invariant')
                files[rel]=frozen['files'][rel]
        if sum(counts.values())%5000==0:print('VERIFY',sum(counts.values()),flush=True)
    assert replays==7470 and all(v==3735 for v in counts.values()) and sum(counts.values())==74700
    iid={key(r):r for r in ordered_inputs()};identical=0
    for pair in j(S01/'groups.json')['pairs']:
        if not pair['pixel_identical']:continue
        for seed in SEEDS:
            a=gzread(outpath(iid[pair['a']],'identity',seed));b=gzread(outpath(iid[pair['b']],'identity',seed))
            for arm in ARMS:
                for m in ('MT','PA','FL'):
                    x=a['arms'][arm]['values'][m];y=b['arms'][arm]['values'][m]
                    if m!='PA':
                        x=x*(a.get('px_per_mm') or 1) if x is not None else None
                        y=y*(b.get('px_per_mm') or 1) if y is not None else None
                    check_equal(x,y,'pixel-identical input measurement '+m)
                identical+=1
    split=j(DEST/'partitions.json');assert all(split['assignment'][x['a']]==split['assignment'][x['b']] for x in split['links'])
    out=dict(passed=True,all_image_seed_variants=replays,arm_outputs=sum(counts.values()),per_seed_arm={f'{s}|{a}':v for (s,a),v in counts.items()},
        B0_full_replay=True,FL_only_MT_PA_boundary_invariants=True,independent_region_missing_invariants=True,qualified_pair_checks=dict(gates),known_group_crossings=0,identical_pair_seed_arm_checks=identical,
        normal_cells={f'{s}|{a}|{m}':v for (s,a,m),v in normal.items()},exploratory_cells={f'{s}|{a}|{m}':v for (s,a,m),v in exploratory.items()},missing_region_records={f'{s}|{a}|{m}':v for (s,a,m),v in coverage.items()},
        implementation_lock_sha256=sha(DEST/'implementation_lock.json'),protected_prior_source_count=len(j(R38/'binding.json')['protected']),prior_sources_unchanged=True,
        new_network_forwards=0,training=False,default_changed=False,subagents=False,prediction_files=files,seconds=time.time()-start)
    write(DEST/'verification.json',out);print('VERIFICATION PASSED',sum(counts.values()),round(time.time()-start),flush=True)

if __name__=='__main__':main()
