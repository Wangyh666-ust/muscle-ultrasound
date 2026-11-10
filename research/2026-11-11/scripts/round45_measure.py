"""CPU replay only; shared C1, immutable B raw heads, no new neural inference."""
from scripts.round43_cpu_imports import install as cpu_install
cpu_install()
import argparse,concurrent.futures as cf
from scripts.round45_common import *
from scripts.round45_frame import frame
from scripts.round45_decoder import pool45,rays45,predict45
from scripts.round44_region import spec,warp,oldfield
from scripts.round44_evaluate import readrow as read44
from scripts.round43_measure import readmask,slim
from scripts.round43_decoder import pool,rays,predict,core
from scripts.round42_core import fit_state
from scripts.round38_core import state
from scripts.round43_fast2 import install
from scripts.round39_verify import check_equal
from src.umud_baseline.data import read_image_gray3

def compact(a):
    a=dict(a)
    d=dict(a.get('display',{}));a['display']=d;risk=d.get('risk')
    if risk:d['risk']={k:risk[k] for k in ('all_tracks_unreliable','aggregate_risk') if k in risk}
    return a

def execute(r,s,phase='all'):
    cv2.setNumThreads(1);install();cfg=j(CFG);cal=j(R40/'calibration.json');start=time.time()
    # Include the isolated eligibility namespace in exact pure-function caching.
    from scripts.round45_decoder import ELIG45
    ELIG45.__globals__['prepare_evidence']=core.prepare_evidence
    loc=j(DEST/'control_frames.json')['selected']['method'];old=read44(r,s);ex=gzread(R42/f"extraction/{r['dataset']}/{Path(r['image_id']).stem}.json.gz");im0=read_image_gray3(ROOT/r['image_path'])
    dst=DEST/f"{'pilot' if phase=='pilot' else 'measurements'}/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
    binding=dict(measure=sha(Path(__file__)),decoder=sha(ROOT/'scripts/round45_decoder.py'),frame=sha(ROOT/'scripts/round45_frame.py'),control=sha(DEST/'control_frames.json'),phase=phase,baseline=[sha(R44/f"{f}/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz") for f in ('measurements','new')])
    if phase!='pilot':binding['lock']=sha(DEST/'lock.json')
    if dst.exists():
        m=j(dst.with_suffix('.sha.json'));assert m['binding']==binding and sha(dst)==m['sha256'];return dict(key=key(r),seed=s,resumed=True)
    variants={};checks=[];sources={};base={k:old[k] for k in ('dataset','image_id','image_path','width','height','px_per_mm','group_id','pixel_sha256')}
    trs=['identity','pad10','r43_gray08','scale095','brightness105'] if phase=='pilot' else list(old['variants'])
    for tr in trs:
        v=old['variants'][tr];t=v['transform'];actual='identity' if tr=='local_identity' else tr;view=warp(im0,spec(r['width'],r['height'],actual));roi=v['evaluation_roi'];fm=frame(view[:,:,0],roi,loc)
        new=tr in NEW44;from43new=tr in NEW43 or tr=='local_identity'
        if new:vf,oldfm=oldfield(view[:,:,0],.03)
        else:
            from scripts.round43_input import loadfield
            vf,oldfm=loadfield(r,actual,.03)
        bp=(R44 if new else R43)/f"inference/I/boundary_{s}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz";raw=readmask(bp);sources[tr]=dict(boundary_B=sha(bp),C1={})
        before=core.masked(raw,roi)&vf[None];a,b,c,d=fm['evidence_box'];observed=np.zeros(raw.shape[1:],bool);observed[b:d,a:c]=True;domainmask=raw&observed[None]
        pools={};audit={};memo={}
        for g in (() if phase=='pilot' else ('D','Q','DQ')):
            masks=domainmask if 'D' in g else before;rw=fm['reference_width'] if 'Q' in g else None;signature=(hashlib.sha256(np.packbits(masks)).hexdigest(),rw)
            if signature not in memo:
                p,au=pool45(masks,roi,rw);pt=rays45(p,roi,rw);memo[signature]=(p,pt,au)
            p,pt,au=memo[signature];pools[g]=(p,pt,rw);audit[g]=dict(candidate_count=len(p),early_pairs=au,reference_width=rw or roi['box'][2]-roi['box'][0],added_to_B=[int((x&~y).sum()) for x,y in zip(masks,before)],removed_from_B=[int((y&~x).sum()) for x,y in zip(masks,before)],evidence_box=fm['evidence_box'] if 'D' in g else core.domain(roi,t['width'],t['height']))
        if phase=='pilot':p0,au=pool(before,roi,False,None);pt0=rays(p0,roi,False)
        arms={}
        for kind in KINDS:
            cp=(R44 if new else R43)/f"inference/C1/{kind}/{tr}/{r['dataset']}/{Path(r['image_id']).stem}.npz" if new or from43new else inferpath(r,tr,kind)
            sources[tr]['C1'][kind]=sha(cp)
            mask=readmask(cp)[0] if new or from43new else unpack(cp)[0]
            st=state(mask,roi,cfg) if new or from43new else fit_state(mask,ex['models'][kind][tr],'K');st['raw_mask']=mask
            if phase=='pilot':
                q=slim(finish(predict(p0,pt0,st,dict(width=t['width'],height=t['height'],px_per_mm=None),cfg,cal,None),t,old));check_equal(q,v['arms'][kind+'|B'],'R45 exact B replay');checks.append(dict(key=key(r),seed=s,kind=kind,transform=tr,exact=True))
                # Separate namespace must reproduce B with neither intervention.
                pp,_=pool45(before,roi);pq=slim(finish(predict45(pp,rays45(pp,roi),st,dict(width=t['width'],height=t['height'],px_per_mm=None),cfg,cal),t,old));check_equal(pq,q,'R45 no-intervention adapter')
            for g,src in (('H','H'),('B','B'),('G','LE')):arms[kind+'|'+g]=compact(v['arms'][kind+'|'+src])
            for g,(p,pt,rw) in pools.items():
                z=predict45(p,pt,st,dict(width=t['width'],height=t['height'],px_per_mm=None),cfg,cal,rw,fm['evidence_box'] if 'D' in g else None)
                arms[kind+'|'+g]=compact(slim(finish(z,t,old)))
        for g,src in (('H','H'),('B','B'),('G','LE')):audit[g]=v['boundary_audit'][src]
        variants[tr]=dict(arms=arms,transform=t,boundary_audit=audit,field=oldfm,region=fm,evaluation_roi=roi)
    out=dict(**base,variants=variants,sources=sources,seconds=time.time()-start,replays=checks);gzwrite(dst,out);write(dst.with_suffix('.sha.json'),dict(binding=binding,sha256=sha(dst)))
    return dict(key=key(r),seed=s,seconds=out['seconds'],variants=len(variants),replays=len(checks))

def main(part,phase,workers):
    register();rr=rows(None if part=='all' else part);rr=rr[:12] if phase=='pilot' else rr;done=[]
    with cf.ProcessPoolExecutor(max_workers=workers) as e:
        fs=[e.submit(execute,r,s,phase) for r in rr for s in SEEDS]
        for f in cf.as_completed(fs):
            done.append(f.result())
            if len(done)%4==0:print('R45_MEASURE',part,phase,len(done),len(fs),done[-1],flush=True)
    write(DEST/f'measure_{phase}_{part}_success.json',dict(details=done))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part');p.add_argument('--phase',default='all');p.add_argument('--workers',type=int,default=6);a=p.parse_args();main(a.part,a.phase,a.workers)
