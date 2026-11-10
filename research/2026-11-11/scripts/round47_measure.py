"""Reuse immutable Round46 neural masks, independently cross two interventions."""
from scripts.round47_core import *
from scripts.round46_measure import readcached
from scripts.round46_decoder import pool46,predict46
from scripts.round38_core import state
from scripts.round43_measure import slim
from scripts.round45_measure import compact
from scripts.round46_frame import finish
import argparse,concurrent.futures as cf

def initialize():
    cv2.setNumThreads(1)
    from scripts.round43_fast2 import install
    install()
    from scripts.round46_fast import install
    install()
    from scripts.round46_column_fast import install
    install()
    ELIG45.__globals__['prepare_evidence']=core.prepare_evidence

def execute(r,s,phase='all'):
    start=time.time();cfg=j(CFG);cal=j(R40/'calibration.json');src=R46/f"measurements/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz";old=gzread(src)
    dst=DEST/f"{'pilot' if phase=='pilot' else 'measurements'}/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
    binding=dict(source=sha(src),core=sha(ROOT/'scripts/round47_core.py'),measure=sha(Path(__file__)),lock=sha(DEST/'lock.json') if phase=='all' else None)
    if dst.exists():
        m=j(dst.with_suffix('.sha.json'));assert m['binding']==binding and sha(dst)==m['sha256'];return dict(key=key(r),seed=s,resumed=True)
    variants={};sources={};checks=[]
    trs=['identity','pad10','translate-2_1','r43_gray08','scale095','brightness105'] if phase=='pilot' else list(old['variants'])
    trs=[t for t in trs if t in old['variants']]
    for tr in trs:
        v=old['variants'][tr];fm=v['region'];roi=v['measurement_roi'];native=dict(width=v['transform']['width'],height=v['transform']['height'],px_per_mm=None)
        raw,meta=readcached(r,tr,'boundary_'+str(s),'I');sources[tr]={'boundary':meta['sha256']};a,b,c,d=fm['evidence_box'];masks=np.zeros_like(raw);masks[:,b:d,a:c]=raw[:,b:d,a:c]
        p,au=boundary_pool(masks,roi,fm['reference_width']);pt=rays46(p,roi,fm['reference_width']);arms={};boundary_audit={}
        for kind in KINDS:
            cm,mt=readcached(r,tr,kind,'I');mask=np.zeros_like(cm[0]);mask[b:d,a:c]=cm[0,b:d,a:c];sources[tr][kind]=mt['sha256']
            R=copy.deepcopy(v['arms'][kind+'|ID']);R=audit_original(R,mask,roi,cfg)
            if phase=='pilot':
                p0,_=pool46(masks,roi,fm['reference_width']);st=state(mask,roi,cfg);st['raw_mask']=mask
                replay=compact(slim(finish(predict46(p0,rays46(p0,roi,fm['reference_width']),st,native,cfg,cal,fm['reference_width'],fm['evidence_box']),v['transform'],old)))
                from scripts.round39_verify import check_equal
                for field in ('values','curve','FL_weights','PA_weights','selection'):check_equal(replay[field],R[field],'R47 exact Round46 ID '+field)
                checks.append(dict(kind=kind,transform=tr,exact=True))
            else:st=state(mask,roi,cfg);st['raw_mask']=mask
            z=predict_boundary(p,pt,st,native,cfg,cal,fm['reference_width'],fm['evidence_box'],masks)
            E=compact(slim(finish(z,v['transform'],old)));E['boundary_score_audit']=[{k:g[k] for k in ('pair_id','soft_score','before_identity_score','boundary_identity','structural_allowed','structural_reasons')} for g in z.get('pair_eligibility',[])];E=audit_original(E,mask,roi,cfg)
            S=selection_rebalance({**R,'_scale':old.get('px_per_mm') or 1.},mask,roi,cfg,cal)
            ES=selection_rebalance({**E,'_scale':old.get('px_per_mm') or 1.},mask,roi,cfg,cal)
            for g,q in (('R',R),('E',E),('S',S),('ES',ES)):
                arms[kind+'|'+g]=q
                assert q['curve']==(R if g in ('R','S') else E)['curve']
                assert q['values']['MT']==(R if g in ('R','S') else E)['values']['MT']
                boundary_audit[g]=dict(candidate_count=len(p),tracking_width=fm['reference_width'],qualification_width=fm['reference_width'],evidence_box=fm['evidence_box'],early_pairs=au) if g in ('E','ES') else copy.deepcopy(v['boundary_audit']['ID'])
        variants[tr]={**{k:v[k] for k in ('transform','field','region','evaluation_roi','measurement_roi')},'boundary_audit':boundary_audit,'arms':arms}
    out={k:old[k] for k in ('dataset','image_id','image_path','width','height','px_per_mm','group_id','pixel_sha256')};out.update(variants=variants,sources=sources,seconds=time.time()-start,replays=checks)
    gzwrite(dst,out);write(dst.with_suffix('.sha.json'),dict(binding=binding,sha256=sha(dst)));return dict(key=key(r),seed=s,seconds=out['seconds'],variants=len(variants),replays=len(checks))

def main(phase,part,workers,limit):
    register()
    if phase=='all':
        for f,h in j(DEST/'lock.json')['sources'].items():assert sha(ROOT/f)==h,f
    rr=rows(None if part=='all' else part);rr=rr[:limit] if limit else rr;start=time.time();done=[]
    with cf.ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        fs=[pool.submit(execute,r,s,phase) for r in rr for s in SEEDS]
        for f in cf.as_completed(fs):
            done.append(f.result());progress=dict(done=len(done),total=len(fs),seconds=time.time()-start,last=done[-1]);write(DEST/f'progress_{phase}_{part}.json',progress)
            print('R47',phase,part,len(done),len(fs),round(time.time()-start),flush=True)
    write(DEST/f'measure_{phase}_{part}_success.json',dict(details=done,seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','all']);p.add_argument('--part',default='development');p.add_argument('--workers',type=int,default=6);p.add_argument('--limit',type=int);a=p.parse_args();main(a.phase,a.part,a.workers,a.limit)
