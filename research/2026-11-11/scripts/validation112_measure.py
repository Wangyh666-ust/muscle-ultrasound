"""Replay the frozen boundary chain; evaluate without any reference-line feedback."""
import argparse,time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,cv2
from PIL import Image
from scripts.round63_common import ROOT,j,sha,write,SIDES
from scripts.round48_length_replay import extract
from scripts.round53_experiment import settings,initialize_worker
from scripts.round46_frame import locate,input_crop
from scripts.round46_measure import readcached
from scripts.round63_paths import decode,choose_pair,image_evidence
from scripts.round64_seams import reconstruct as seams
from scripts.round65_core import keep
from scripts.round68_geometry import localize,reconstruct
from scripts.round68_common import normalize
from scripts.round69_core import classify
from scripts.round71_work import base_pair,local_values
from scripts.round71_roi import locate as tissue_locate
from scripts.round67_work import check_pair
from scripts.round74_geometry import evaluate
from scripts.round63_measure import original_pair
from scripts.load_frozen_boundary import load_boundary

OUT=ROOT/'runs/validation112_current_20261110'

def to_arrays(p):
    if p is None:return None
    q=deepcopy(p)
    for k in ('x','y','kind','observed','confidence','anchor_eligible','source_qualified','qualified','uncertainty','existing','qualified_source'):
        if k in q:q[k]=np.asarray(q[k])
    return q

def c1_segments(row,seed,frame):
    cfg,_=settings();ds,sid=row['dataset'],row['image_id'];roi=dict(box=frame['scan_box'],fallback=frame['fallback'],reason=frame['reason'])
    if ds=='external34':
        cache,_=readcached(row,'identity','B_'+str(seed),'I');mask=cache[0]
    else:
        box=frame['input_box'];a,b,c,d=box
        with np.load(OUT/f'inference/{ds}/{sid}/B_{seed}.npz') as z:prob=z['probability'][0]
        h,w=row['height'],row['width'];mask=np.zeros((h,w),bool)
        l,t,r,bt=max(0,a),max(0,b),min(w,c),min(h,d)
        mask[t:bt,l:r]=prob[t-b:bt-b,l-a:r-a]>.5
    a,b,c,d=frame['evidence_box'];valid=np.zeros_like(mask);valid[b:d,a:c]=mask[b:d,a:c]
    raw,kept,_,_,audit=extract(valid,roi,cfg)
    segments=kept
    return segments,dict(raw=len(raw),kept=len(kept),filter_reasons=[a['flags'] for a in audit])

def replay(prob,image,valid,width,segments,scale,domain):
    cfg=j(ROOT/'runs/round63_boundary_paths_v1/config.json')
    scfg=j(ROOT/'runs/round64_boundary_continuation_v1/controls/S_confirmation_lock.json')['selected']
    wcfg=j(ROOT/'runs/round68_boundary_paths_v1/parameter_lock.json')['selected']
    qcfg=j(ROOT/'runs/round69_measurement_support_v1/parameter_lock.json')['selected']
    pcfg=j(ROOT/'runs/round74_permission_scoring_v1/parameter_lock.json')['selected']
    decoded=decode(prob,image,valid,width,cfg,segments,scale);pools=[];gray=image_evidence(image,width)
    for i in (0,1):
        field=prob[i+1]*(.4+.6*prob[0]);extra,_=seams(decoded['tracks']['rebuilt'][i],field,gray,valid,width,scfg,cfg)
        pools.append(keep(decoded['tracks']['rebuilt'][i],extra,width))
    source,candidates=choose_pair(pools,width,cfg,segments,scale,True,True)
    info=dict(candidate_pairs=len(candidates),track_counts={k:list(map(len,v)) for k,v in decoded['tracks'].items()})
    if source is None:return None,None,info
    ws=[];wjs=[];elig={}
    for i,s in enumerate(SIDES):
        f=prob[i+1]*(.4+.6*prob[0]);role=prob[i+1]
        src=normalize(source[s],f);pool=[normalize(p,f) for p in pools[i]]
        w,aw=localize(src,f,role,gray,valid,width,wcfg['W']['config'])
        wj,aj=reconstruct(w,pool,f,role,gray,valid,width,wcfg['J']['config'])
        # R68 skips the W call only if its signature is exactly unchanged: reconstruct(w)==reconstruct(src).
        ws.append(w);wjs.append(wj)
        eq,_,_=classify(src,w,f,role,gray,valid,width,qcfg);elig[s]=eq
    wp,wfallback=check_pair(source,ws);wjp,jfallback=check_pair(source,wjs)
    # The protection stage is the exact R71 base_pair function, not a fresh ranker.
    old=dict(width=width,arms=dict(W=dict(pair=wp),WJ=dict(pair=wjp)))
    old69=dict(arms=dict(WM=dict(eligibility=elig),W0=dict(pair=wp)))
    base,ba=base_pair(old,old69)
    result=dict(interval=base['interval'],width=float(np.ptp(domain)))
    audits={}
    for s in SIDES:
        outputs=evaluate(base[s],domain,float(np.ptp(domain)),max(image.shape)/512.,pcfg)
        if outputs['P']['path'] is None:return None,base,dict(**info,reason='completion_failed')
        result[s]=outputs['P']['path'];audits[s]=outputs['P']['audit']
    result['interval']=[max(result[s]['x'][0] for s in SIDES),min(result[s]['x'][-1] for s in SIDES)]
    info.update(base_audit=ba,permission_audit=audits,W_order_fallback=wfallback,J_order_fallback=jfallback)
    return result,base,info

def qualify(pair,mask,shape):
    if pair is None:return
    lock=j(ROOT/'runs/round72_penalty_core_v1/parameter_lock.json')
    for i,s in enumerate(SIDES):
        p=pair[s];x,y=p['x'],p['y'];inside=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])
        onmask=np.zeros(len(x),bool);onmask[inside]=mask[np.rint(y[inside]).astype(int).clip(0,mask.shape[0]-1),np.rint(x[inside]).astype(int).clip(0,mask.shape[1]-1)]
        # Geometry qualification follows the existing R72 uncertainty calibration.
        derived=(~p['existing'])|(p['kind']==6)
        uncertainty=p['uncertainty'].copy();uncertainty[derived]*=lock['uncertainty_multiplier'][str(i)]
        p['qualified']=(uncertainty<=2*max(shape)/512.)&onmask
        existing=p['existing']&(p['kind']!=6)
        p['qualified'][existing]&=p['qualified_source'][existing]

def work(row,parity=False):
    initialize_worker();cv2.setNumThreads(1)
    tic=time.monotonic();ds,sid=row['dataset'],row['image_id'];dst=OUT/f'results/{ds}/{sid}.json.gz'
    if dst.exists() and not parity:return dict(image_id=sid,resumed=True)
    gray=np.array(Image.open(ROOT/row['image_path']).convert('L'));fm=locate(gray,'C');box=fm['input_box'];image=input_crop(np.repeat(gray[:,:,None],3,2),fm)[:,:,0]
    a,b,c,d=fm['evidence_box'];l,t=box[:2];valid=np.zeros(image.shape,bool)
    valid[max(0,b-t):min(image.shape[0],d-t),max(0,a-l):min(image.shape[1],c-l)]=True
    segments={};filters={}
    for seed in (20261023,20261024):
        segs,au=c1_segments(row,seed,fm);segments[str(seed)]=segs;filters[str(seed)]=au
    md=j(ROOT/'runs/frozen_boundary_20261110/source_round63_boundary_paths_v1_metadata.json')
    if ds=='external34':
        old=md[ds+'|'+sid];expected=old['segments'];got=segments['20261023']
        assert len(expected)==len(got) and {s['label_id'] for s in got}=={s['label_id'] for s in expected},(sid,'C1 parity failed')
        for s,e in zip(sorted(got,key=lambda s:s['label_id']),sorted(expected,key=lambda s:s['label_id'])):
            for k in ('p0','p1','centroid','direction_deg'):assert np.max(abs(np.asarray(s[k])-e[k]))<1e-8,(sid,k)
    rec=dict(input_box=box,matrix=[[1,0,0],[0,1,0]])
    mask,dm=tissue_locate(gray);native_mask=input_crop(np.repeat(mask.astype(np.uint8)[:,:,None],3,2),fm)[:,:,0].astype(bool)
    yy,xx=np.where(native_mask);domain=[float(xx.min()),float(xx.max())] if len(xx) else [0.,float(image.shape[1]-1)]
    mapped=[dict(s,**{k:np.asarray(s[k])-np.asarray(box[:2]) for k in ('p0','p1','centroid')}) for s in segments['20261023']]
    with np.load(OUT/f'inference/{ds}/{sid}/boundary.npz') as z:prob=z['probability']
    if ds=='external34' and not parity:
        pair=load_boundary(ds,sid,'identity','P');info=dict(reused_frozen_P=True);base=load_boundary(ds,sid,'identity','B')
        frozen=j(ROOT/'runs/frozen_boundary_20261110/manifest.json')
        fr=next(r for r in frozen['rows'] if r['dataset']==ds and r['image_id']==sid and r['view']=='identity')
        assert box==fr['input']['input_box'] and image.shape==tuple(fr['shape'])
        domain=fr['domain']
    else:
        pair,base,info=replay(prob,image,valid,fm['reference_width'],mapped,row['px_per_mm'],domain)
    if parity:
        target=load_boundary(ds,sid,'identity','P');source_target=load_boundary(ds,sid,'identity','B')
        out=dict(id=sid,pair_presence_equal=(pair is None)==(target is None),sides={})
        if pair is not None and target is not None:
            for s in SIDES:
                out['sides'][s]=dict(x_equal=np.array_equal(pair[s]['x'],target[s]['x']),
                    mae=float(np.mean(abs(np.interp(target[s]['x'],pair[s]['x'],pair[s]['y'])-target[s]['y']))),
                    max=float(np.max(abs(np.interp(target[s]['x'],pair[s]['x'],pair[s]['y'])-target[s]['y']))),
                    base_max=float(np.max(abs(np.interp(source_target[s]['x'],base[s]['x'],base[s]['y'])-source_target[s]['y']))))
        write(OUT/f'parity/{sid}.json',out);return out
    if pair is not None:
        pair={**pair,**{s:to_arrays(pair[s]) for s in SIDES}}
        pair['interval']=[max(pair[s]['x'][0] for s in SIDES),min(pair[s]['x'][-1] for s in SIDES)]
        qualify(pair,native_mask,image.shape)
    full=original_pair(pair,rec);outputs={}
    for seed,segs in segments.items():
        m=local_values(full,mask,segs,row['px_per_mm'],float(np.ptp(domain)),None,None)
        outputs[seed]=m
    write(dst,dict(row=row,frame=fm,domain=domain,domain_meta=dm,pair=full,filter_audit=filters,segments=segments,
                   boundary_audit=info,outputs=outputs,seconds=time.monotonic()-tic))
    return dict(image_id=sid,pair=pair is not None,counts=[filters[s]['kept'] for s in filters],seconds=time.monotonic()-tic)

def main():
    p=argparse.ArgumentParser();p.add_argument('--parity',action='store_true');p.add_argument('--workers',type=int,default=3);p.add_argument('--limit',type=int);a=p.parse_args()
    rows=j(OUT/'protocol.json')['rows']
    if a.parity:rows=[r for r in rows if r['dataset']=='external34'][:a.limit or 34]
    elif a.limit:rows=rows[:a.limit]
    tic=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=[pool.submit(work,r,a.parity) for r in rows]
        for f in as_completed(tasks):
            r=f.result();results.append(r);print('MEASURE112',len(results),len(rows),r,flush=True)
            write(OUT/('parity_progress.json' if a.parity else 'measure_progress.json'),dict(done=len(results),total=len(rows),last=r,seconds=time.monotonic()-tic))
    write(OUT/('parity_done.json' if a.parity else 'measure_done.json'),dict(results=results,seconds=time.monotonic()-tic))

if __name__=='__main__':main()
