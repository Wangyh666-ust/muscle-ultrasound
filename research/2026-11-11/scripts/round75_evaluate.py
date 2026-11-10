"""Full original-image replay and 112-image perturbation comparison after control lock."""
import argparse,time,math
from copy import deepcopy
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,cv2
from PIL import Image
from scripts.round75_common import *
from scripts.round53_experiment import initialize_worker,settings
from scripts.validation112_measure import to_arrays,replay,qualify
from scripts.round46_frame import input_crop
from scripts.round46_measure import readcached
from scripts.round48_length_replay import extract,measure
from scripts.round63_measure import original_pair
from scripts.round71_roi import locate as tissue_locate
from scripts.round64_measure_adapter import adapt
from scripts.round75_core import exact_adapter,install_exact_geometry,reweight_F

_READY=False
def ready():
    global _READY
    if not _READY:initialize_worker();install_exact_geometry();cv2.setNumThreads(1);_READY=True

@lru_cache(maxsize=1)
def frozen():return {(r['key'],r['view']):r for r in j(ROOT/'runs/frozen_boundary_20261110/manifest.json')['rows']}

def load_infer(row,view,kind):
    p=OUT/f"inference/{row['dataset']}/{row['image_id']}/{view}/{kind}.json"
    if not p.exists() and view=='identity' and kind!='boundary':
        old=V112/f"inference/{row['dataset']}/{row['image_id']}/{kind}.npz"
        if old.exists():
            m=j(old.with_suffix('.json'))
            with np.load(old) as z:prob=z['probability'].copy()
            return prob,dict(frame=m['frame'],matrix=[[1,0,0],[0,1,0]])
        mask,m=readcached(row,'identity',kind,'I')
        return mask,dict(frame=m['field'],matrix=[[1,0,0],[0,1,0]],mask_is_native=True)
    meta=j(p)
    if meta.get('reuse_R46'):
        m=j(ROOT/meta['metadata_path']);data=ROOT/m['data_path']
        with np.load(data) as z:
            mask=np.unpackbits(z['bits'] if 'bits' in z else z['packed'],count=int(np.prod(z['shape']))).reshape(z['shape']).astype(bool)
        return mask,dict(frame=m['field'],matrix=meta['matrix'],mask_is_native=True)
    with np.load(ROOT/meta['data_path']) as z:prob=z['probability'].copy()
    return prob,meta

def view_gray(gray,view,matrix):
    if view=='brightness105':gray=np.rint(gray.astype(float)*1.05).clip(0,255).astype(np.uint8)
    return cv2.warpAffine(gray,np.asarray(matrix,float),(gray.shape[1],gray.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def segments(row,view,seed,frame,matrix):
    raw,meta=load_infer(row,view,'B_'+str(seed));h,w=row['height'],row['width'];a,b,c,d=frame['input_box']
    if meta.get('mask_is_native'):mask=raw[0]
    else:
        mask=np.zeros((h,w),bool);l,t,r,bt=max(0,a),max(0,b),min(w,c),min(h,d)
        mask[t:bt,l:r]=raw[0,t-b:bt-b,l-a:r-a]>.5
    evidence=np.zeros_like(mask);l,t,r,bt=frame['evidence_box'];evidence[t:bt,l:r]=mask[t:bt,l:r]
    cfg,_=settings();roi=dict(box=frame['scan_box'],fallback=frame['fallback'],reason=frame['reason'])
    rr,kept,_,_,audit=extract(evidence,roi,cfg)
    m=np.asarray(matrix,float);inverse=np.linalg.inv(m[:,:2]);offset=m[:,2];out=[]
    for s in kept:
        q=deepcopy(s)
        for k in ('p0','p1','centroid'):q[k]=inverse@(np.asarray(q[k])-offset)
        # All registered transforms are positive isotropic scales/translations.
        # Keep the original PCA direction, not the rounded endpoint chord angle.
        q['direction_deg']=s['direction_deg']
        scale=m[0,0]
        for k in ('length_px','width_px'):
            if k in q:q[k]/=scale
        if 'area_px' in q:q['area_px']/=scale**2
        out.append(q)
    return out,len(rr),dict(kept=len(kept),raw=len(rr))

def baseline_measure(curve,segs,raw,roi,row):
    cfg,cal=settings();t=dict(sx=1.,sy=1.,tx=0.,ty=0.,width=row['width'],height=row['height'],name='identity')
    return measure(curve,0 if curve else None,segs,raw,roi,t,row,cfg,cal)

def slim(result):
    return {k:result.get(k) for k in ('values','stage','MT_reasons','mt_positions','PA_weights','FL_weights','selection','boundary_sensitivity')}

def work(row,view):
    ready();tic=time.monotonic();ds,sid=row['dataset'],row['image_id'];dst=OUT/f'results/{ds}/{sid}/{view}.json.gz'
    if dst.exists():return dict(dataset=ds,id=sid,view=view,resumed=True)
    lock=j(OUT/'execution_lock.json');assert lock['admitted']['Q']
    gray=np.array(Image.open(ROOT/row['image_path']).convert('L'));boundary,meta=load_infer(row,view,'boundary');frame=meta['frame'];matrix=meta['matrix'];viewim=view_gray(gray,view,matrix)
    mask,dm=tissue_locate(viewim);native_mask=input_crop(np.repeat(mask.astype(np.uint8)[:,:,None],3,2),frame)[:,:,0].astype(bool)
    yy,xx=np.where(native_mask);domain=[float(xx.min()),float(xx.max())] if len(xx) else [0.,float(boundary.shape[2]-1)]
    segs={};counts={}
    for seed in SEEDS:segs[str(seed)],raw,counts[str(seed)]=segments(row,view,seed,frame,matrix)
    original112=V112/f'results/{ds}/{sid}.json.gz';audit={}
    if view=='identity' and original112.exists():
        old=readgz(original112);pair={**old['pair'],**{s:to_arrays(old['pair'][s]) for s in ('upper','lower')}} if old['pair'] else None
        domain=old['domain'];audit['reused112_exact']=True
    elif view=='identity':
        from scripts.load_frozen_boundary import load_boundary
        fr=frozen()[(ds+'|'+sid,'identity')];pair=load_boundary(ds,sid,'identity','P')
        assert frame['input_box']==fr['input']['input_box']
        if pair:
            pair['interval']=[max(pair[s]['x'][0] for s in ('upper','lower')),min(pair[s]['x'][-1] for s in ('upper','lower'))]
            qualify(pair,native_mask,boundary.shape[1:]);pair=original_pair(pair,dict(input_box=frame['input_box'],matrix=matrix))
        domain=fr['domain'];audit['reused_historical_boundary']=True
    else:
        # Boundary selection continues to use B23 only, exactly as in the frozen baseline.
        a=np.asarray(matrix,float);box=np.asarray(frame['input_box'][:2]);mapped=[]
        for s in segs['20261023']:
            q=deepcopy(s)
            for k in ('p0','p1','centroid'):q[k]=a[:,:2]@q[k]+a[:,2]-box
            mapped.append(q)
        image=input_crop(np.repeat(viewim[:,:,None],3,2),frame)[:,:,0]
        ev=np.zeros_like(viewim,np.uint8);l,t,r,b=frame['evidence_box'];ev[t:b,l:r]=1
        valid=input_crop(np.repeat(ev[:,:,None],3,2),frame)[:,:,0].astype(bool)
        pair,base,audit=replay(boundary,image,valid,frame['reference_width'],mapped,row['px_per_mm']*a[0,0] if row['px_per_mm'] else None,domain)
        if pair:qualify(pair,native_mask,image.shape)
        pair=original_pair(pair,dict(input_box=frame['input_box'],matrix=matrix))
    oldfit,oldau=adapt(pair,row['width'],row['height']);oldcv=oldfit['curve'] if oldfit else None
    qcv,qau=exact_adapter(pair,oldcv,row['width'],row['height'])
    m=np.asarray(matrix,float);l,t,r,b=frame['scan_box'];roi=dict(box=[(l-m[0,2])/m[0,0],(t-m[1,2])/m[1,1],(r-m[0,2])/m[0,0],(b-m[1,2])/m[1,1]],fallback=frame['fallback'],reason=frame['reason'])
    result={};parity=[];cfg,cal=settings()
    for seed in SEEDS:
        seed=str(seed);B=baseline_measure(oldcv,segs[seed],counts[seed]['raw'],roi,row);Q=baseline_measure(qcv,segs[seed],counts[seed]['raw'],roi,row)
        if view=='identity' and original112.exists():
            old=readgz(original112)['formal_outputs'][seed]
            for key in ('MT','PA','FL'):
                x,y=B['values'][key],old['values'][key];assert (x is None)==(y is None),(sid,key,x,y)
                if x is not None:assert abs(x-y)<1e-8,(sid,key,x,y)
            parity.append(seed)
        result[seed]=dict(B=slim(B),Q=slim(Q))
        if lock['admitted'].get('F'):
            F,fa=reweight_F(B,cfg,cal,row['px_per_mm'] or 1.,lock['F']['tau']);result[seed]['F']=slim(F);result[seed]['F']['audit']=fa
    write(dst,dict(row=row,view=view,matrix=matrix,frame=frame,domain=domain,pair=pair,baseline_curve=oldcv,qualified_curve=qcv,
      original_adapter=oldau,Q_adapter=qau,segments=segs,counts=counts,outputs=result,baseline112_parity=parity,boundary_audit=audit,seconds=time.monotonic()-tic))
    return dict(dataset=ds,id=sid,view=view,seconds=time.monotonic()-tic)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--limit',type=int);ap.add_argument('--identity',action='store_true');ap.add_argument('--only112',action='store_true');ap.add_argument('--perturb-only',action='store_true');a=ap.parse_args()
    rows=records();tasks=[]
    for row in rows:
        if a.only112 and row['dataset'] not in ('external34','young78'):continue
        views=('identity',) if a.identity or row['dataset'] not in ('external34','young78') else VIEWS
        if a.perturb_only:views=tuple(v for v in views if v!='identity')
        for view in views:tasks.append((row,view))
    if a.limit:tasks=tasks[:a.limit]
    tic=time.monotonic();done=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(work,*t) for t in tasks]
        for f in as_completed(jobs):
            done.append(f.result());write(OUT/'eval_progress.json',dict(done=len(done),total=len(tasks),last=done[-1],seconds=time.monotonic()-tic))
            if len(done)%10==0:print('R75_EVAL',len(done),len(tasks),'seconds',round(time.monotonic()-tic),done[-1]['id'],flush=True)
    write(OUT/('eval_identity_done.json' if a.identity else 'eval_done.json'),dict(total=len(tasks),seconds=time.monotonic()-tic,rows=done))

if __name__=='__main__':main()
