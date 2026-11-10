"""CPU-only adapter and golden replay of the unchanged formal measurement chain."""
from scripts.round43_cpu_imports import install
install()
from functools import lru_cache
from scripts.round65_common import *
from scripts.round53_experiment import initialize_worker,settings
from scripts.round48_length_replay import extract,measure as frozen_measure
from scripts.round46_measure import readcached
from scripts.round64_measure_adapter import adapt

_READY=False


def initialize():
    global _READY
    if not _READY:initialize_worker();_READY=True


@lru_cache(maxsize=64)
def fixed_c1(key):
    initialize();md=j(R63/'metadata.json')[key];row=md['row'];cfg,_=settings()
    source=ROOT/md['c1_source'];assert sha(source)==md['c1_source_sha256']
    previous=readgz(source);v=previous['variants']['identity'];roi=v['measurement_roi']
    assert roi['box']==md['region']['scan_box']
    masks,meta=readcached(row,'identity','B_20261023','I')
    assert meta['sha256']==previous['sources']['identity']['B_20261023_I']
    mask=np.zeros_like(masks[0]);a,b,c,d=md['region']['evidence_box'];mask[b:d,a:c]=masks[0,b:d,a:c]
    raw,kept,_,_,_=extract(mask,roi,cfg)
    old=v['arms']['B_20261023|ID']
    assert len(raw)==old['raw_components'] and len(kept)==old['filter_count']
    assert {s['label_id'] for s in kept}=={s['label_id'] for s in md['segments']}
    row={**row,'px_per_mm':md['px_per_mm']}
    return dict(segments=kept,raw_count=len(raw),roi=roi,row=row,
        mask_sha=meta['sha256'],previous=old)


def curve_measure(curve,pairid,c1):
    cfg,cal=settings();row=c1['row']
    transform=dict(sx=1.,sy=1.,tx=0.,ty=0.,width=row['width'],height=row['height'],name='identity')
    return frozen_measure(curve,pairid,c1['segments'],c1['raw_count'],c1['roi'],transform,row,cfg,cal)


def measure(pair,key):
    c1=fixed_c1(key);row=c1['row'];converted,audit=adapt(pair,row['width'],row['height'])
    result=curve_measure(converted['curve'] if converted else None,0 if converted else None,c1)
    # Formal output and actual contributors; no replacement metric definitions.
    slim={k:result.get(k) for k in ('values','PA_weights','FL_weights','stage','selection','exploratory_values','selected_pair_id')}
    slim['contributors']={str(s['label_id']):dict(p0=s['p0'],p1=s['p1'],centroid=s['centroid'])
        for s in c1['segments'] if str(s['label_id']) in {**(slim['PA_weights'] or {}),**(slim['FL_weights'] or {})}}
    return slim,audit


def main():
    register();md=j(R63/'metadata.json');selected=[]
    for ds in ('external34','apo153','dev282'):
        selected.extend([k for k in md if k.startswith(ds+'|')][:6])
    records=[];maximum=0.
    for key in selected:
        c1=fixed_c1(key);old=c1['previous'];out=curve_measure(old['curve'],old['selected_pair_id'],c1)
        errors=[]
        for metric in ('MT','PA','FL'):
            a,b=old['values'][metric],out['values'][metric]
            assert (a is None)==(b is None),(key,metric,a,b)
            if a is not None:errors.append(abs(a-b));assert np.isclose(a,b,rtol=1e-8,atol=1e-8),(key,metric,a,b)
        for name in ('PA_weights','FL_weights'):
            assert set(old[name])==set(out[name]),(key,name)
            assert all(abs(old[name][k]-out[name][k])<1e-9 for k in old[name])
        maximum=max(maximum,max(errors,default=0.))
        # An absent boundary must run through the same missingness path without unpack errors.
        missing=curve_measure(None,None,c1)
        assert all(missing['values'][m] is None for m in ('MT','PA','FL'))
        records.append(dict(key=key,old_values=old['values'],replayed_values=out['values'],weights_equal=True))
    write(OUT/'measurement_adapter_golden.json',dict(passed=True,records=records,max_error=maximum,
        existing_function='scripts.round48_length_replay.measure',observations_only_adaptation=True,
        script_sha=sha(Path(__file__)),cfg_and_calibration_frozen=True))
    print('FORMAL_ADAPTER_GOLDEN_PASSED',len(records),'max_error',maximum,flush=True)


if __name__=='__main__':main()
