"""Separate consumed observation domain and boundary reference coordinates."""
import inspect
from scripts.round43_decoder import *
from scripts.round43_decoder import ADAPTED

def build():
    ns=dict(POOL.__globals__)
    src=ADAPTED['pool'].replace('qual_width=None):','qual_width=None,reference_width=None):',1)
    old='factor=1046/max(x1-x0,1)'
    assert src.count(old)==1
    src=src.replace(old,'factor=1046/max(x1-x0 if reference_width is None else reference_width,1)')
    exec(compile(src,'<round45_boundary_reference>','exec'),ns)
    elig=ADAPTED['eligibility'].replace('qual_width=None):','qual_width=None,evidence_box=None):',1)
    old="box=domain(st['roi'],row['width'],row['height'])"
    assert old in elig
    elig=elig.replace(old,"box=domain(st['roi'],row['width'],row['height']) if evidence_box is None else evidence_box")
    exec(compile(elig,'<round45_observation_eligibility>','exec'),ns)
    return ns['normalized_pool_from_columns'],ns['eligibility'],dict(pool=src,eligibility=elig)

POOL45,ELIG45,SOURCES=build()

def pool45(masks,roi,reference_width=None):
    cols=[list(enumerate(core.run_centres_per_column(m))) for m in masks]
    return POOL45(cols,roi,independent=False,qual_width=reference_width,reference_width=reference_width)

def rays45(pairs,roi,reference_width=None):
    rr=copy.deepcopy(roi)
    if reference_width is not None:rr['box'][2]=rr['box'][0]+reference_width
    return smooth_extensions(pairs,rr)

def predict45(p,pt,st,native,cfg,cal,reference_width=None,evidence_box=None):
    gates=ELIG45(p,st,native,reference_width,evidence_box);gates=evidence_gates(p,st,native,cfg,cal,gates)
    a=measure_soft(pt,st,native,cfg,gates)
    if a['curve'] is None:
        exp,_=measure_pool(pt,st,native,cfg);a=suppress(a,exp)
    a=modify_fl(a,cfg,cal,True,True);a['_native_dimensions']=native
    return apply_u(a,cfg,cal)
