"""Track generation is independent of candidate qualification and ray policy."""
import inspect
from scripts.round45_decoder import *
from scripts.round45_decoder import POOL45,SOURCES as PREVIOUS

def build():
    ns=dict(POOL45.__globals__)
    src=PREVIOUS['pool']
    old='qw=w if qual_width is None else qual_width*factor'
    assert src.count(old)==1
    # Generate the same fixed tracks/pairs regardless of later support policy.
    # Minimum 8-cell numerical fit remains; final support threshold is unchanged.
    src=src.replace(old,'qw=0.0')
    exec(compile(src,'<round46_fixed_track_coordinates>','exec'),ns)
    return ns['normalized_pool_from_columns'],src

POOL46,SOURCE46=build()

def pool46(masks,roi,tracking_width):
    cols=[list(enumerate(core.run_centres_per_column(m))) for m in masks]
    return POOL46(cols,roi,independent=False,reference_width=tracking_width)

def rays46(pairs,roi,ray_width):
    return rays45(pairs,roi,ray_width)

def predict46(p,pt,st,native,cfg,cal,qualification_width,evidence_box):
    return predict45(p,pt,st,native,cfg,cal,qualification_width,evidence_box)
