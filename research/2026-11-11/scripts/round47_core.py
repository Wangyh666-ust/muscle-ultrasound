"""Separate boundary decoding from a bounded actual-visible-support rebalance.

No model, threshold, direction, intersection or default source changes.
"""
from scripts.round43_cpu_imports import install as cpu_install
cpu_install()
from scripts.round47_common import *
from scripts.round46_decoder import POOL46,rays46,ELIG45,core
from scripts.round40_core import evidence_gates,measure_soft,measure_pool,suppress,modify_fl,apply_u
from scripts.round36_core import interpolated_y
from scipy import ndimage

def centres(mask,reference_width):
    h,w=mask.shape;out=[[] for _ in range(w)]
    # Two canonical reference pixels; keep native pixel cells, no interpolation.
    gap=max(1.,2.*reference_width/1046.)
    flat=np.flatnonzero(np.ascontiguousarray(mask.T))
    if not len(flat):return out
    xs=flat//h;ys=flat%h;cuts=np.r_[0,np.flatnonzero((np.diff(xs)!=0)|(np.diff(ys)>gap+1e-9))+1];ends=np.r_[cuts[1:],len(flat)];n=ends-cuts
    vv=(ys[cuts+(n-1)//2]+ys[cuts+n//2])/2.
    for x,v in zip(xs[cuts].tolist(),vv.tolist()):out[x].append(v)
    return out

def boundary_pool(masks,roi,reference_width):
    cols=[list(enumerate(centres(m,reference_width))) for m in masks]
    return POOL46(cols,roi,independent=False,reference_width=reference_width)

def identity_evidence(pool,masks,width):
    out={};rad=max(1,round(2*width/1046));h,w=masks.shape[1:]
    for p in pool:
        cv=p['curve'];xx=np.rint(cv['paired_x']).astype(int);records=[]
        for side,ch in (('sup',1),('deep',2)):
            yy=np.rint(interpolated_y(cv[side],xx)).astype(int);good=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            samples=np.stack([masks[:,np.clip(yy[good]+dy,0,h-1),xx[good]] for dy in range(-rad,rad+1)]) if good.any() else np.zeros((1,3,0),bool)
            hit=samples.any(0);records.append(dict(side=side,union=float(hit[0].mean()) if good.any() else 0.,role=float(hit[ch].mean()) if good.any() else 0.,samples=int(good.sum())))
        # Inherited .15 boundary / .10 role evidence ratio; no fitted hyperparameter.
        factor=float(np.mean([.6*x['union']+.4*x['role'] for x in records]));out[p['pair_id']]=dict(factor=factor,sides=records)
    return out

def predict_boundary(p,pt,st,native,cfg,cal,width,box,masks):
    ELIG45.__globals__['prepare_evidence']=core.prepare_evidence
    gates=evidence_gates(p,st,native,cfg,cal,ELIG45(p,st,native,width,box));confidence=identity_evidence(p,masks,width)
    for g in gates:
        g['before_identity_score']=g['soft_score'];g['boundary_identity']=confidence[g['pair_id']];g['soft_score']*=confidence[g['pair_id']]['factor']
    a=measure_soft(pt,st,native,cfg,gates)
    if a['curve'] is None:a=suppress(a,measure_pool(pt,st,native,cfg)[0])
    a=modify_fl(a,cfg,cal,True,True);a['_native_dimensions']=native
    return apply_u(a,cfg,cal)

def exposure(xs,lo,hi,anchors):
    xs=np.asarray(xs,float);xx=(xs-lo)/max(hi-lo,1);valid=(xx>=0)&(xx<=1);xx=xx[valid]
    if not len(xx):return np.zeros(3),np.zeros(3),np.zeros(24)
    # Partition of unity at fixed left/middle/right anchors. Integrate actual
    # observed component pixels, never an extrapolated infinite trajectory.
    A=np.asarray(anchors);W=np.stack([np.interp(xx,A,np.eye(3)[b]) for b in range(3)],1)
    thirds=np.bincount(np.minimum((xx*3).astype(int),2),minlength=3)/len(xx)
    hist=np.histogram(xx,bins=np.linspace(0,1,25))[0]/len(xx)
    return W.mean(0),thirds,hist

def selection_rebalance(q,mask,roi,cfg,cal):
    """Consume frozen eligible contributors; preserve all their geometry/risk.

    This is an incremental rebalance, not a claim to remove all inherited
    centroid dependence or to create anatomical track identities.
    """
    out=copy.deepcopy(q);lab,_=ndimage.label(mask,structure=np.ones((3,3),bool));lo,_,hi,_=roi['box'];ac=cfg['aggregate'];lookup={}
    for c in out['display']['segments']:
        sid=c['label_id'];yy,xx=np.nonzero(lab==sid);lookup[str(sid)]=dict(xs=xx,ex=exposure(xx,lo,hi,ac['anchors']))
    audit={}
    for metric in ('FL','PA'):
        weights=out[metric+'_weights'];ids=list(weights)
        if not ids:audit[metric]=dict(contributors=[],region_mass=[0.,0.,0.],coverage=[False]*3,histogram=[0.]*24,effective=0.,max_weight=0.,available=False);continue
        ww=np.array([weights[i] for i in ids]);ee=np.array([lookup[i]['ex'][0] for i in ids]);mass=ww@ee
        target=np.asarray(ac['region_weights'],float);factor=ee@(target/(mass+cal['lambda_evidence']));nw=ww*factor
        # This branch never manufactures a result from zero observed support.
        if nw.sum()>0:nw/=nw.sum()
        else:nw=ww.copy()
        out[metric+'_weights']={i:float(w) for i,w in zip(ids,nw)}
        out['display'][metric+'_weights']=out[metric+'_weights']
        cs={str(c['label_id']):c for c in out['display']['segments']}
        if metric=='FL':vals=[float(np.linalg.norm(np.subtract(cs[i]['sup'],cs[i]['deep'])))/(out.get('_scale') or 1.) for i in ids]
        else:
            from scripts.round29_core import line_at
            vals=[]
            for i in ids:
                c=cs[i];slope=line_at(out['curve']['deep'],c['centroid'][0])[0];angle=abs(c['direction_deg']-np.degrees(np.arctan(slope)))%180;vals.append(min(angle,180-angle))
        out['values'][metric]=float(nw@np.asarray(vals));out['display']['values'][metric]=out['values'][metric]
        thirds=nw@np.array([lookup[i]['ex'][1] for i in ids]);hist=nw@np.array([lookup[i]['ex'][2] for i in ids])
        rec=[]
        for i,old,new in zip(ids,ww,nw):
            rec.append(dict(id=int(i),weight=float(new),before_weight=float(old),observed_region_exposure=lookup[i]['ex'][0].tolist(),observed_thirds=lookup[i]['ex'][1].tolist(),observed_pixel_count=len(lookup[i]['xs'])))
        audit[metric]=dict(contributors=rec,region_mass=thirds.tolist(),coverage=(thirds>0).tolist(),histogram=hist.tolist(),effective=float(1/(nw@nw)),max_weight=float(nw.max()),available=True)
    out['display']['selection_audit']=audit;out['selection_audit']=audit
    # Selected green representatives remain explicitly decorative in this arm.
    out['display']['representatives_only']=True
    if out['values']['FL'] is not None:out['selection']['value_px']=out['values']['FL']*(out.get('_scale') or 1.);out['display']['selection']=out['selection']
    out.pop('_scale',None);return out

def audit_original(q,mask,roi,cfg):
    # Same actual-support definitions as S; report untouched formal weights.
    lab,_=ndimage.label(mask,structure=np.ones((3,3),bool));lo,_,hi,_=roi['box'];audit={}
    for metric in ('FL','PA'):
        weights=q[metric+'_weights'];hist=np.zeros(24);mass=np.zeros(3);recs=[]
        for sid,w in weights.items():
            _,xx=np.nonzero(lab==int(sid));ee,tt,hh=exposure(xx,lo,hi,cfg['aggregate']['anchors']);hist+=w*hh;mass+=w*tt
            recs.append(dict(id=int(sid),weight=w,observed_region_exposure=ee.tolist(),observed_thirds=tt.tolist(),observed_pixel_count=len(xx)))
        audit[metric]=dict(contributors=recs,region_mass=mass.tolist(),coverage=(mass>0).tolist(),histogram=hist.tolist(),effective=1/sum(w*w for w in weights.values()) if weights else 0.,max_weight=max(weights.values(),default=0.),available=bool(weights))
    q['selection_audit']=audit;q['display']['selection_audit']=audit;return q
