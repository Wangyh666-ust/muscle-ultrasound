"""Four-arm factorial measurement, both networks perturbed, frozen parameters."""
from scripts.round43_cpu_imports import install as cpu_install
cpu_install()
import argparse,concurrent.futures as cf
from scripts.round46_frame import *
from scripts.round46_decoder import pool46,rays46,predict46,ELIG45,core
from scripts.round43_decoder import pool,rays,predict
from scripts.round43_measure import slim
from scripts.round43_fast2 import install
from scripts.round38_core import state
from scripts.round45_measure import compact
from src.umud_baseline.data import read_image_gray3

def readcached(r,tr,kind,mode):
    sp=DEST/f"inference/{mode}/{kind}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.json";m=j(sp);p=ROOT/m['data_path'];assert sha(p)==m['sha256']
    with np.load(p) as z:a=np.unpackbits(z['bits'] if 'bits' in z else z['packed'],count=int(np.prod(z['shape']))).reshape(z['shape']).astype(bool)
    return a,m

def execute(r,s,phase):
    cv2.setNumThreads(1);install();ELIG45.__globals__['prepare_evidence']=core.prepare_evidence
    cfg=j(CFG);cal=j(R40/'calibration.json');old=gzread(R45/f"measurements/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz");im=read_image_gray3(ROOT/r['image_path']);start=time.time()
    destination=DEST/f"{'pilot' if phase=='pilot' else 'measurements'}/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
    binding=dict(measure=sha(Path(__file__)),decoder=sha(ROOT/'scripts/round46_decoder.py'),frame=sha(ROOT/'scripts/round46_frame.py'),crop=sha(DEST/'control_crop.json'),phase=phase)
    if phase!='pilot':binding['lock']=sha(DEST/'lock.json')
    if destination.exists():
        m=j(destination.with_suffix('.sha.json'));assert m['binding']==binding and sha(destination)==m['sha256'];return dict(key=key(r),seed=s,resumed=True)
    variants={};sources={};reuse=0
    trs=['identity','r43_gray08'] if phase=='pilot' else list(old['variants'])+list(NEW46)
    for tr in trs:
        actual='identity' if tr=='local_identity' else tr;t=spec(r['width'],r['height'],actual);view=warp(im,t);fm=locate(view[:,:,0],j(DEST/'control_crop.json')['selected']['method']);legacy_roi=image_roi(view[:,:,0]);of,om=__import__('scripts.round43_input',fromlist=['field']).field(view[:,:,0],.03)
        new_roi=dict(box=fm['scan_box'],fallback=fm['fallback'],reason=fm['reason']);native=dict(width=t['width'],height=t['height'],px_per_mm=None)
        outer=np.zeros(view.shape[:2],bool);a,b,c,d=fm['evidence_box'];outer[b:d,a:c]=True
        bm={};meta={};pools={};audit={};memo={};arms={};sources[tr]={};oldv=old['variants'].get(tr)
        for mode in ('B','I'):
            bm[mode],meta[mode]=readcached(r,tr,'boundary_'+str(s),mode);sources[tr]['boundary_'+mode]=meta[mode]['sha256']
        for g in GROUPS:
            mode='I' if g in ('I','ID') else 'B';newdecode=g in ('D','ID');roi=new_roi if newdecode else legacy_roi
            masks=bm[mode]&outer[None] if newdecode else core.masked(bm[mode],roi)&of[None]
            signature=(hashlib.sha256(np.packbits(masks)).hexdigest(),newdecode,tuple(roi['box']))
            if signature not in memo:
                if newdecode:p,au=pool46(masks,roi,fm['reference_width']);pt=rays46(p,roi,fm['reference_width'])
                else:p,au=pool(masks,roi);pt=rays(p,roi)
                memo[signature]=(p,pt,au)
            p,pt,au=memo[signature];pools[g]=(p,pt,roi,newdecode)
            audit[g]=dict(candidate_count=len(p),early_pairs=au,tracking_width=fm['reference_width'] if newdecode else roi['box'][2]-roi['box'][0],qualification_width=fm['reference_width'] if newdecode else roi['box'][2]-roi['box'][0],ray_width=fm['reference_width'] if newdecode else roi['box'][2]-roi['box'][0],evidence_box=fm['evidence_box'] if newdecode else core.domain(roi,t['width'],t['height']))
        for kind in KINDS:
            states={};cm={};cmeta={}
            for mode in ('B','I'):
                raw,cmeta[mode]=readcached(r,tr,kind,mode);cm[mode]=raw[0];sources[tr][kind+'_'+mode]=cmeta[mode]['sha256']
            done={}
            for g,(p,pt,roi,newdecode) in pools.items():
                mode='I' if g in ('I','ID') else 'B';csig=(hashlib.sha256(np.packbits(cm[mode])).hexdigest(),tuple(roi['box']))
                if csig not in states:
                    st=state(cm[mode],roi,cfg);st['raw_mask']=cm[mode];states[csig]=st
                st=states[csig]
                boundary_equal=meta['B']['reused_old_same_environment'] or meta['B']['historical_mask_changed_pixels']==0
                c1_equal=cmeta['B']['reused_old_same_environment'] or cmeta['B']['historical_mask_changed_pixels']==0
                if g=='B' and oldv is not None and boundary_equal and c1_equal:
                    assert legacy_roi==oldv['evaluation_roi'];q=compact(oldv['arms'][kind+'|B']);reuse+=1
                else:
                    if newdecode:z=predict46(p,pt,st,native,cfg,cal,fm['reference_width'],fm['evidence_box'])
                    else:z=predict(p,pt,st,native,cfg,cal)
                    q=compact(slim(finish(z,t,old)))
                arms[kind+'|'+g]=q
        variants[tr]=dict(arms=arms,transform=t,field=om,region=fm,evaluation_roi=legacy_roi,measurement_roi=new_roi,boundary_audit=audit,historical_baseline={k:oldv['arms'][k+'|B']['values'] for k in KINDS} if oldv else None)
    out={k:old[k] for k in ('dataset','image_id','image_path','width','height','px_per_mm','group_id','pixel_sha256')};out.update(variants=variants,sources=sources,seconds=time.time()-start,historical_exact_reuse=reuse)
    gzwrite(destination,out);write(destination.with_suffix('.sha.json'),dict(binding=binding,sha256=sha(destination)))
    return dict(key=key(r),seed=s,seconds=out['seconds'],variants=len(variants),historical_exact_reuse=reuse)

def main(phase,part,workers):
    register();rr=rows('development')[:12] if phase=='pilot' else rows(None if part=='all' else part);done=[];start=time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(execute,r,s,phase) for r in rr for s in SEEDS]
        for f in cf.as_completed(fs):
            done.append(f.result());progress=dict(done=len(done),total=len(fs),seconds=time.time()-start,last=done[-1]);write(DEST/f'measure_progress_{phase}_{part}.json',progress)
            if len(done)%4==0:print('R46_MEASURE',phase,part,len(done),len(fs),round(time.time()-start),flush=True)
    write(DEST/f'measure_{phase}_{part}_success.json',dict(seconds=time.time()-start,details=done));print('R46_MEASURE_COMPLETE',phase,part,len(done),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('--part',default='all');p.add_argument('--workers',type=int,default=6);a=p.parse_args();main(a.phase,a.part,a.workers)
