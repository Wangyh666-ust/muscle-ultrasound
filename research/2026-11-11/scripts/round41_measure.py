"""Apply exactly frozen Round40 K to the five C1 models and two boundary seeds."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import sys,time,argparse,concurrent.futures as cf
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.round40_common import *
from scripts.round40_core import *
from scripts.round38_core import state
from scripts.round38_speed import install
from scripts.round39_verify import check_equal
from scripts.round41_native import spec as nspec
R40=ROOT/'runs/round40_intersection_qualification_v1';DEST=ROOT/'runs/round41_attribution_consistency_v1'
KINDS=('F','A_20261023','B_20261023','A_20261024','B_20261024')
NEW41=('r41_translate1_-2','r41_scale0985','r41_scale1015','r41_pad12','r41_brightness0975')
FIELDS=('values','curve','selection','FL_weights','PA_weights','aggregation','selected_pair_id','stage')
def inferpath(r,tr,kind):return DEST/f"inference/{kind}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz"
def resultpath(r,seed):return DEST/f"measurements/{seed}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
def finish(a,t,r):
    q=to_original(a,t,r);q.pop('_native_dimensions',None)
    if a.get('rejected_curve') is not None:q['rejected_curve']=map_curve(a['rejected_curve'],inverse_spec(t))
    if a['exploratory_values'].get('MT') is not None:q['exploratory_values']['MT']=a['exploratory_values']['MT']/t['sy']/(r.get('px_per_mm') or 1.)
    return q
def execute(r,seed,pilot=False):
    cv2.setNumThreads(1);install();cfg=j(CFG);cal=j(R40/'calibration.json');dst=resultpath(r,seed);binding=sha(DEST/'training_lock.json')
    if dst.exists() and not pilot:
        meta=j(dst.with_suffix('.sha.json'));assert sha(dst)==meta['sha256'] and meta['code_sha256']==sha(Path(__file__)) and meta['training_lock_sha256']==binding
        return dict(key=key(r),seed=seed,resumed=True)
    old0=gzread(outpath(r,'identity',seed));row={k:old0[k] for k in ('dataset','image_id','width','height','px_per_mm','group_id','image_path','pixel_sha256')}
    variants={};source={};start=time.time()
    for tr in transforms(r)+NEW41:
        bp=DEST/f"inference_local_boundary/{seed}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz" if tr in NEW41 else cachepath(r,tr,str(seed));br=j(bp.with_suffix('.json'));assert sha(bp)==br['sha256']
        bm=unpack(bp) if tr in NEW41 else unpack(bp)[3:];frec=j(inferpath(r,tr,'F').with_suffix('.json'));roi=frec['roi'];t=frec['transform'];expected=nspec(r['width'],r['height'],tr);assert t==expected
        for k in ('sx','sy','tx','ty','width','height','name'):assert t[k]==br['transform'][k]
        native=dict(width=t['width'],height=t['height'],px_per_mm=None)
        pool=normalized_pool(masked(bm,roi),roi);pt=smooth_extensions(pool,roi);arms={};source[tr]={'boundary':br['sha256']}
        for kind in KINDS:
            cp=inferpath(r,tr,kind);cr=j(cp.with_suffix('.json'));assert sha(cp)==cr['sha256'] and cr['roi']==roi and cr['transform']==t
            mask=unpack(cp)[0];st=state(mask,roi,cfg);st['raw_mask']=mask
            oldg=eligibility(pool,st,native);gates=evidence_gates(pool,st,native,cfg,cal,oldg)
            a=measure_soft(pt,st,native,cfg,gates)
            if a['curve'] is None:
                exp,_=measure_pool(pt,st,native,cfg);a=suppress(a,exp)
            a=modify_fl(a,cfg,cal,True,True);a['_native_dimensions']=native;a=apply_u(a,cfg,cal);arms[kind]=finish(a,t,row);source[tr][kind]=cr['sha256']
        parity=None;roi_exact=None
        if tr not in NEW41:
            hist=gzread(outpath(r,tr,seed))['arms']['K'];parity=frec['historical_mask_exact']
            roi_exact=roi==j(cachepath(r,tr,'c1').with_suffix('.json'))['roi']
            if parity and roi_exact:
                for field in FIELDS:check_equal(arms['F'][field],hist[field],'same-environment F replay '+key(r)+' '+tr+' '+field)
            # Keep measured historical K even when cloud threshold bits differ.
            arms['historical_K']=hist
        variants[tr]=dict(arms=arms,transform=t,historical_mask_exact=parity,historical_ROI_exact=roi_exact,boundary_environment='historical_local' if tr not in NEW41 else 'local_new5')
    row.update(boundary_seed=seed,variants=variants,sources=source,seconds=time.time()-start)
    if pilot:dst=DEST/f"measurement_pilot/{seed}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
    gzwrite(dst,row);write(dst.with_suffix('.sha.json'),dict(sha256=sha(dst),code_sha256=sha(Path(__file__)),training_lock_sha256=binding))
    return dict(key=key(r),seed=seed,variants=len(variants),seconds=row['seconds'])
def main(workers,limit):
    verify_lock();rows=ordered_inputs();rows=rows[:limit] if limit else rows
    if not all((DEST/f'inference_local_boundary_{s}_success.json').exists() for s in SEEDS):
        print('WAITING_FOR_FROZEN_BOUNDARY_NEW5_LOCAL_ENVIRONMENT_CACHE',flush=True)
        while not all((DEST/f'inference_local_boundary_{s}_success.json').exists() for s in SEEDS):time.sleep(20)
    if not limit:
        for k in KINDS+tuple('boundary_'+str(s) for s in SEEDS):assert (DEST/f'inference_{k}_success.json').exists()
    done=[];start=time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(execute,r,s,bool(limit)) for r in rows for s in SEEDS]
        for f in cf.as_completed(fs):
            done.append(f.result())
            if len(done)%5==0:print('MEASUREMENT',len(done),len(fs),round(time.time()-start),flush=True)
            write(DEST/'measurement_progress.json',dict(done=len(done),total=len(fs),seconds=time.time()-start))
    write(DEST/f'measurement_{"pilot" if limit else "success"}.json',dict(details=done,seconds=time.time()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=10);p.add_argument('--limit',type=int);a=p.parse_args();main(a.workers,a.limit)
