"""Conservative isolated adapter to the existing Round54 measurement functions.

Observed points alone fit the frozen four-piece measurement geometry. Rebuilt or
continued points are not promoted to observations and do not change terminal
measurement rays. Display/finite-curve estimates remain separate. No source edit.
"""
from scripts.round43_cpu_imports import install
install()
import numpy as np
from scripts.round56_boundary import fit_side
from scripts.round54_fl_support import native_measure
from scripts.round53_experiment import initialize_worker,settings
from scripts.round48_length_replay import measure as frozen_measure
from scripts.round63_common import ROOT,j

_READY=False


def initialize():
    global _READY
    if not _READY:initialize_worker();_READY=True


def adapt(pair,width,height):
    if pair is None:return None,dict(reason='no_pair')
    sides={};observed={};discarded={}
    for side,new in [('sup','upper'),('deep','lower')]:
        p=pair[new];x=np.asarray(p['x']);y=np.asarray(p['y']);obs=np.asarray(p['observed'],bool)
        if obs.sum()<8:return None,dict(reason='insufficient_real_observations',side=side)
        fitted=fit_side(np.column_stack([x[obs],y[obs]]),width,height)
        if fitted is None:return None,dict(reason='frozen_four_piece_fit_failed',side=side)
        observed[side]=x[obs];discarded[side]=int((~obs).sum());sides[side]=fitted
    lo=max(sides[s]['support'][0] for s in sides);hi=min(sides[s]['support'][1] for s in sides)
    xs=np.intersect1d(observed['sup'],observed['deep'])
    if hi<=lo or len(xs)<8:return None,dict(reason='insufficient_shared_real_observations')
    check=np.linspace(lo,hi,101)
    if np.any(np.interp(check,sides['sup']['knots_x'],sides['sup']['knots_y'])>=
              np.interp(check,sides['deep']['knots_x'],sides['deep']['knots_y'])):
        return None,dict(reason='fitted_order_failure')
    curve=dict(**sides,support=[lo,hi],paired_x=xs.tolist(),max_sample_gap=max(2,.015*width))
    return dict(pair_id=0,curve=curve),dict(real_observed_columns={s:len(v) for s,v in observed.items()},
        nonobserved_not_promoted=discarded,measurement_extent='shared_real_observations',
        coordinate_frame='original_full_image',four_piece_estimator='frozen_Round56_fit_side')


def measure(pair,segments,raw_count,roi,row,width):
    initialize();p,audit=adapt(pair,width,row['height'])
    transform=dict(sx=1.,sy=1.,tx=0.,ty=0.,width=row['width'],height=row['height'],name='identity')
    if p is None:
        cfg,cal=settings()
        native,output=frozen_measure(None,None,segments,raw_count,roi,transform,row,cfg,cal)
    else:native,output=native_measure(p,segments,raw_count,roi,transform,row)
    return native,output,audit
