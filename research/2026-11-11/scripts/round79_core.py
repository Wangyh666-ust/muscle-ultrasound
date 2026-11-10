"""Opt-in FL aggregation only. Fixed candidates, individual FL, MT/PA and geometry."""
from copy import deepcopy
import numpy as np
from scripts.round43_cpu_imports import install
install()
from scripts.round39_core import combine_fl,robust

def bounds_from_source(source,row):
    matrix=np.asarray(source['matrix'],float);scan=source['frame']['scan_box'];left=source['frame']['input_box'][0]
    scan=np.array([(scan[0]-matrix[0,2])/matrix[0,0],(scan[2]-matrix[0,2])/matrix[0,0]],float)
    domain=source.get('domain');reason='image_only_domain'
    if domain is None or len(domain)!=2:return scan.tolist(),'fallback_no_domain'
    tissue=(np.asarray(domain,float)+left-matrix[0,2])/matrix[0,0]
    tissue=np.clip(tissue,0,row['width'])
    if not np.isfinite(tissue).all() or tissue[1]<=tissue[0]:return scan.tolist(),'fallback_invalid_domain'
    return tissue.tolist(),reason

def capped(weights,limit):
    """Water-fill on the existing simplex; n<3 deliberately unchanged."""
    w=np.asarray(weights,float).copy();n=len(w)
    if n<3 or w.max(initial=0)<=limit+1e-12:return w
    limit=max(limit,1/n);out=np.zeros(n);free=np.ones(n,bool);remaining=1.
    while free.any():
        mass=w[free].sum();v=w[free]/mass*remaining if mass>0 else np.full(free.sum(),remaining/free.sum())
        over=v>limit+1e-12;indices=np.where(free)[0]
        if not over.any():out[indices]=v;break
        out[indices[over]]=limit;free[indices[over]]=False;remaining=1-out.sum()
    assert abs(out.sum()-1)<1e-9 and out.max()<=limit+1e-9
    return out

def calculate(pack,arm,parameters,ac,cal):
    ids=pack['ids'];n=len(ids)
    if not n:return dict(weights={},FL_native=None,regions=[],reason='no_existing_FL',cap_active=False)
    vals=np.asarray(pack['FL_native'],float);q=np.asarray(pack['quality'],float);raw=np.asarray(pack['raw_quality'],float);risk=np.asarray(pack['propagated_factor'],float)
    if 'R' in arm:
        eta=parameters['R']['local_fraction'];q=q*np.maximum(raw/np.maximum(q,1e-300),1.)**(1-eta)
    quality=q*risk;x=np.asarray(pack['x']);bounds=pack['tissue_bounds'] if 'S' in arm else pack['scan_bounds'];span=bounds[1]-bounds[0]
    # Only spatial coordinates change in S. Length/risk normalization and Huber scale remain frozen.
    normalized=(x-bounds[0])/span;kernel=np.exp(-.5*((normalized[:,None]-np.asarray(ac['anchors'])[None,:])/ac['spatial_sigma'])**2)
    kernel/=np.maximum(kernel.sum(1)[:,None],1e-300)
    if not any(k in arm for k in ('G','E','C')):
        w,regions=combine_fl(vals,quality,kernel,[],pack['reference_span'],ac,cal,True,True);cap=False
    else:
        exponent=parameters['G']['evidence_exponent'] if 'G' in arm else parameters['E']['evidence_exponent'] if 'E' in arm else 1.
        maximum=parameters['G']['maximum_track_weight'] if 'G' in arm else parameters['C']['maximum_track_weight'] if 'C' in arm else 1.
        mass=(quality[:,None]*kernel).sum(0);evidence=mass/(mass+cal['lambda_evidence']);rw=np.asarray(ac['region_weights'])*evidence**exponent
        if rw.sum()<=0:return dict(weights=pack['baseline_weights'],FL_native=pack['baseline_FL_native'],regions=[],reason='fallback_zero_mass',cap_active=False)
        rw/=rw.sum();w=np.zeros(n);regions=[]
        for b in range(3):
            ww,value,scale=robust(vals,quality*kernel[:,b],ac['scale_floor_FL_roi']*pack['reference_span'],ac,cal['FL_scale_roi']*pack['reference_span'])
            w+=rw[b]*ww;regions.append(dict(region=['left','middle','right'][b],region_weight=float(rw[b]),mass=float(mass[b]),value=value,scale=scale))
        before=w.copy();w=capped(w,maximum);cap=bool(np.max(abs(w-before))>1e-12)
    if not np.isfinite(w).all() or w.sum()<=0:
        if pack.get('baseline_weights') and pack.get('baseline_FL_native') is not None:return dict(weights=pack['baseline_weights'],FL_native=pack['baseline_FL_native'],regions=[],reason='fallback_no_finite_spatial_evidence',cap_active=False)
        raise ValueError('Invalid FL weights')
    assert abs(w.sum()-1)<1e-8 and np.min(w)>=-1e-12
    return dict(weights={str(i):float(z) for i,z in zip(ids,w)},FL_native=float(w@vals),regions=regions,cap_active=cap,
        maximum_weight=float(w.max()),effective_tracks=float(1/(w@w)),bounds=bounds,reason='fixed_contributors_reweighted')
