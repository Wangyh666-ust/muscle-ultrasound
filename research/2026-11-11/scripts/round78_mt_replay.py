"""Four fixed MT-only groups on all sealed R77 inputs; default remains untouched."""
from copy import deepcopy
import time
from scripts.round77_common import ROOT,OUT as R77,j,write,readgz,sha,SEEDS,VIEWS,NEW
from scripts.round78_mt_core import measure_mt
OUT=ROOT/'runs/round78_mt_geometry_adapter_v1'
ARMS=dict(B=('old','old'),G=('complete','old'),S=('old','qualified'),M=('complete','qualified'))

def register():
    p=OUT/'protocol.json'
    if p.exists():return
    old=j(R77/'protocol.json');write(p,dict(round=78,status='registered_before_MT_replay',training=False,new_inference=False,default_changed=False,subagents=False,
       source='Round77 P with additional prior and fixed A; current actual measurement geometry is baseline',rows=old['rows'],models=old['models'],protected=old['protected'],
       arms=ARMS,primary='M-B',S='negative control: qualification alone may admit distorted old geometry',
       fixed=['same three sampling x positions','same physical units and vertical-spacing definition','same input/models/selection','same PA and FL values/weights','existing qualified flags and uncertainty policy'],
       qualified_interpolation='both bracketing points qualified and at most one original probability-grid step apart; no new clinical threshold',
       no_frame='If source has no legacy sampling positions, keep missing; do not invent a new interval',
       scope=dict(originals=547,views=995,seeds=2,reference_images=112,transforms=[*VIEWS,NEW]),
       gate='no silent loss of baseline outputs; common-cohort MT error and recovered MT quality, missing, stability/extremes all reported before any promotion',
       no_GT_in_adapter=True,independence='historical regression only; no new blind validation',
       source_manifest_sha=sha(R77/'delivery_manifest.json')))

def fixtures():
    import numpy as np
    x=np.arange(11,dtype=float);p=dict(upper=dict(x=x,y=x*0+2,qualified=np.ones(11,bool)),lower=dict(x=x,y=x*0+7,qualified=np.ones(11,bool)))
    cv=dict(paired_x=x.tolist(),max_sample_gap=1,sup=dict(knots_x=[0,10],knots_y=[2,2]),deep=dict(knots_x=[0,10],knots_y=[7,7]))
    args=dict(pair=p,curve=cv,positions=[2.5,5.,7.5],width=11,height=10,px_per_mm=1,native_step=1,geometry='complete',support='qualified')
    good=measure_mt(**args);assert good['value']==5
    bad=deepcopy(p);bad['upper']['qualified'][5]=False;assert measure_mt(**{**args,'pair':bad})['value'] is None
    bad=deepcopy(p);bad['lower']['qualified'][7]=False;assert measure_mt(**{**args,'pair':bad})['value'] is None
    bad=deepcopy(p);bad['upper']['x']=np.array([0,1,2,8,9,10.]);bad['upper']['y']=np.full(6,2.);bad['upper']['qualified']=np.ones(6,bool);assert measure_mt(**{**args,'pair':bad})['value'] is None
    assert measure_mt(**{**args,'positions':[]})['value'] is None
    write(OUT/'functional_checks.json',dict(passed=True,normal_geometry_correct=True,both_upper_and_lower_checked=True,drawable_unqualified_not_accepted=True,
        missing_samples_not_bridged_by_interpolation=True,no_sampling_interval_invented=True))

def run():
    register();fixtures();files=sorted((R77/'results').rglob('*.json.gz'));assert len(files)==995;start=time.monotonic();n=0;checks=0;maxerr=0.
    for file in files:
        r=readgz(file);row=r['row'];pair=r['pairs']['P'];curve=r['curves']['P'];positions=r['outputs'][SEEDS[0]]['P']['mt_positions'];step=1/float(r['matrix'][0][0]);diagnostics={}
        for arm,(geom,support) in ARMS.items():diagnostics[arm]=measure_mt(pair,curve,positions,row['width'],row['height'],row['px_per_mm'],step,geom,support)
        outputs={}
        for seed in SEEDS:
            base=r['outputs'][seed]['P'];got=diagnostics['B']['value'];expected=base['values']['MT'];assert (got is None)==(expected is None),(row['image_id'],r['view'],got,expected)
            if got is not None:maxerr=max(maxerr,abs(got-expected));assert abs(got-expected)<1e-8
            outputs[seed]={}
            for arm in ARMS:
                q=deepcopy(base);q['values']['MT']=diagnostics[arm]['value'];q['MT_reasons']=diagnostics[arm]['reasons'];q['mt_positions']=positions
                if arm!='B':q['stage']='output' if all(v is not None for v in q['values'].values()) else 'partial_or_missing'
                for field in ('PA','FL'):assert q['values'][field]==base['values'][field];checks+=1
                for field in ('PA_weights','FL_weights','selection','boundary_sensitivity'):assert q[field]==base[field];checks+=1
                outputs[seed][arm]=q
        dst=OUT/f"results/{row['dataset']}/{row['image_id']}/{r['view']}.json.gz";write(dst,dict(row=row,view=r['view'],frame=r['frame'],matrix=r['matrix'],source=file.relative_to(ROOT).as_posix(),source_sha=sha(file),outputs=outputs,MT_audit=diagnostics,
            same_sampling_positions=True,MT_only=True))
        n+=1
        if n%100==0:print('R78_MT_ONLY',n,995,'seconds',round(time.monotonic()-start),flush=True)
    write(OUT/'replay_done.json',dict(views=n,outputs=n*8,maximum_baseline_MT_error=maxerr,PA_FL_and_weight_equality_checks=checks,training=False,inference=False,default_changed=False))

if __name__=='__main__':run()
