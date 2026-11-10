"""Seven paired interventions with frozen C1 K and immutable old baseline."""
import argparse,concurrent.futures as cf
from scripts.round43_input import *
from scripts.round43_decoder import pool,rays,predict,core
from scripts.round42_core import fit_state
from scripts.round38_core import state
from scripts.round42_fast_v2 import install
from scripts.round39_verify import check_equal
from scripts.round40_gallery import display

def readmask(p):
    with np.load(p) as z:return np.unpackbits(z['bits'],count=int(np.prod(z['shape']))).reshape(z['shape']).astype(bool)

def slim(a):
    keep=('values','curve','stage','selected_pair_id','roi','raw_components','filter_count','FL_weights','PA_weights','selection','exploratory_values')
    q={k:a[k] for k in keep};q['display']=display(a)
    q['gates']=[{k:g[k] for k in ('pair_id','structural_allowed','structural_reasons','soft_score','C1_containment') if k in g} for g in a.get('pair_eligibility') or []]
    q['display']['raw_components']=a['raw_components'];q['display']['filter_count']=a['filter_count']
    return q

def execute(r,s,phase):
    cv2.setNumThreads(1);install();cfg=j(CFG);cal=j(R40/'calibration.json');margin=j(DEST/'control_geometry.json')['selected_margin']
    old=gzread(old_resultpath(r,s));ex=gzread(R42/f"extraction/{r['dataset']}/{Path(r['image_id']).stem}.json.gz")
    dst=DEST/f"{phase}/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
    binding=dict(measure=sha(Path(__file__)),decoder=sha(ROOT/'scripts/round43_decoder.py'),input=sha(ROOT/'scripts/round43_input.py'),baseline=sha(old_resultpath(r,s)),phase=phase)
    if phase!='pilot':binding['lock']=sha(DEST/'lock.json')
    if dst.exists():
        m=j(dst.with_suffix('.sha.json'));assert m['binding']==binding and sha(dst)==m['sha256'];return dict(key=key(r),seed=s,resumed=True)
    start=time.time();row={k:old[k] for k in ('dataset','image_id','image_path','width','height','px_per_mm','group_id','pixel_sha256')};variants={};parities=[];sources={}
    trs=['identity'] if phase=='pilot' else list(NEW43)+['local_identity'] if phase=='new' else TRS[key(r)]
    for tr in trs:
        actual='identity' if tr=='local_identity' else tr;isnew=tr in NEW43 or tr=='local_identity'
        vf,fm=loadfield(r,actual,margin);t=fm['transform'];roi=image_roi(warp(__import__('src.umud_baseline.data',fromlist=['read_image_gray3']).read_image_gray3(ROOT/r['image_path']),t)[:,:,0]) if isnew else ex['models']['F'][tr]['roi']
        native=dict(width=t['width'],height=t['height'],px_per_mm=None)
        bm=readmask(probpath(r,tr,'boundary_'+str(s),'R')) if isnew else getboundary(r,tr,s)[0]
        ip=probpath(r,tr,'boundary_'+str(s),'I');im=readmask(ip);sources[tr]=dict(I=sha(ip),R=sha(probpath(r,tr,'boundary_'+str(s),'R')) if isnew else sha(getboundary(r,tr,s)[2]))
        masks={g:core.masked(im if g.startswith('I') else bm,roi) for g in GROUPS}
        for g in GROUPS:
            if g in ('D','ID','IDS','IDQ','IDSQ'):masks[g]&=vf[None]
        pools={};audits={};memo={}
        for g in GROUPS:
            qwidth=fm['valid_width'] if g in ('IDQ','IDSQ') and not fm['fallback'] else None;ind=g in ('IDS','IDSQ')
            signature=(hashlib.sha256(np.packbits(masks[g])).hexdigest(),ind,qwidth)
            if signature in memo:p,pt,audit=memo[signature]
            else:
                p,audit=pool(masks[g],roi,ind,qwidth);pt=rays(p,roi,ind);memo[signature]=(p,pt,audit)
            pools[g]=(p,pt,qwidth,signature);audits[g]=dict(candidate_count=len(p),early_pairs=audit,own_support=[{z:v['curve'][z]['support'] for z in ('sup','deep')} for v in p])
        arms={}
        for kind in KINDS:
            cp=probpath(r,tr,kind,'C1') if isnew else inferpath(r,tr,kind);mask=readmask(cp)[0] if isnew else unpack(cp)[0]
            if isnew:st=state(mask,roi,cfg);st['raw_mask']=mask
            else:st=fit_state(mask,ex['models'][kind][tr],'K')
            done={}
            for g in GROUPS:
                p,pt,qwidth,signature=pools[g]
                if g=='R' and not isnew and phase!='pilot':a=old['variants'][tr]['arms'][kind];a=slim(a);done[signature]=a
                elif signature in done:a=done[signature]
                else:
                    full=finish(predict(p,pt,st,native,cfg,cal,qwidth),t,old)
                    if g=='R' and not isnew:
                        for field in ('values','curve','selection','FL_weights','PA_weights','selected_pair_id','stage'):check_equal(full[field],old['variants'][tr]['arms'][kind][field],'R43 exact baseline '+field)
                        parities.append(dict(kind=kind,transform=tr,passed=True))
                    a=slim(full);done[signature]=a
                arms[kind+'|'+g]=a
        variants[tr]=dict(arms=arms,transform=t,boundary_audit=audits,field=fm,evaluation_roi=roi)
    row.update(variants=variants,sources=sources,replays=parities,seconds=time.time()-start);gzwrite(dst,row);write(dst.with_suffix('.sha.json'),dict(binding=binding,sha256=sha(dst)))
    return dict(key=key(r),seed=s,seconds=row['seconds'],variants=len(variants),replays=len(parities))

def main(phase,partition,workers,limit):
    register();rr=rows(None if partition=='all' else partition);rr=rr[:limit] if limit else rr
    if phase!='pilot':assert (DEST/'lock.json').exists()
    done=[];start=time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as e:
        fs=[e.submit(execute,r,s,phase) for r in rr for s in SEEDS]
        for f in cf.as_completed(fs):
            done.append(f.result());progress=dict(done=len(done),total=len(fs),seconds=time.time()-start,last=done[-1]);write(DEST/f'measure_progress_{phase}_{partition}.json',progress)
            if len(done)%5==0:print('R43_MEASURE',progress,flush=True)
    write(DEST/f'measure_{phase}_{partition}_success.json',dict(details=done,seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','measurements','new']);p.add_argument('--partition',default='development');p.add_argument('--workers',type=int,default=6);p.add_argument('--limit',type=int);a=p.parse_args();main(a.phase,a.partition,a.workers,a.limit)
