"""Round37 full-cohort and fixed-boundary replay; prediction code never reads GT."""
import argparse,copy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scipy import ndimage
from scripts.round37_io import *
from scripts.round37_core import raw_components,filter_components,measure,gate,begin_row
from scripts.round36_core import choose,interpolated_y,support
from scripts.round32_common import map_segment,inverse_spec,map_point
from scripts.round37_infer import back
from scripts.round27_experiment import check_equal

def scores(pool,evidence):
    if not pool:return []
    lo=min(p['curve']['support'][0] for p in pool);hi=max(p['curve']['support'][1] for p in pool)
    yy,xx=np.nonzero(evidence);keep=(xx>=lo)&(xx<=hi);xx=xx[keep];yy=yy[keep]
    bins=np.minimum(((xx-lo)/max(hi+1-lo,1)*3).astype(int),2);active=[b for b in range(3) if np.any(bins==b)]
    result=[]
    for p in pool:
        cv=p['curve'];ins=support(cv,xx)&(yy>=interpolated_y(cv['sup'],xx))&(yy<=interpolated_y(cv['deep'],xx))
        c=float(np.mean([ins[bins==b].mean() for b in active])) if active else 0.
        result.append(dict(pair_id=p['pair_id'],old_containment=c,old_score=c*p['old_quality'],quality=p['old_quality']))
    return result

def states(mask,roi,cfg,t=None,wh=None):
    ex,raw=raw_components(mask);labs,_=ndimage.label(mask,structure=np.ones((3,3),bool));out={}
    mappedroi=copy.deepcopy(roi)
    if t:
        iv=inverse_spec(t);a,b,c,d=roi['box'];mappedroi['box']=map_point([a,b],iv)+map_point([c,d],iv)
    for mode in ('old','scale','soft'):
        segs,audit=filter_components(raw,roi,cfg,mode);ev=np.isin(labs,[s['label_id'] for s in segs])
        if t:
            segs=[map_segment(s,iv) for s in segs];ev=back(ev.astype(np.uint8),t,*wh,cv2.INTER_NEAREST)>0
        out[mode]={'segments':segs,'evidence':ev,'audit':audit,'roi':mappedroi}
    return out,ex

def arm_state(arm,old,new):
    return (new if arm in ('I','HI') else old)['soft' if arm in ('F1','H','HI') else 'scale' if arm=='F0' else 'old']

def run_row(r,old,new,cfg,verify=False,with_fixed=True):
    begin_row()
    arms={};fixed={};scorecache={};measurecache={}
    base=r['arms']['M'];fixedcurve=base['curve'];fixedpid=base['selected_pair_id']
    for arm in ARMS:
        st=arm_state(arm,old,new);sig=id(st)
        if sig not in scorecache:
            sc=scores(r['pool'],st['evidence']);p=choose(r['pool'],sc,r);scorecache[sig]=(sc,p)
        sc,p=scorecache[sig];cv=p['curve'] if p else None;pid=p['pair_id'] if p else None
        out=measure(cv,st['segments'],r,pid,st['roi'],cfg,arm)
        out['candidate_scores']=sc;out['roi']=st['roi'];out['filter_audit']=st['audit'];out['input_kind']='normalized' if arm in ('I','HI') else 'old'
        if verify and arm=='R':
            for k in ('values','selection','PA_ids','PA_invalid','MT_reasons','mt_positions'):
                check_equal(out[k],base[k],'Round37 frozen R '+r['image_id']+' '+k)
            assert pid==fixedpid
        arms[arm]=out
        if with_fixed:
            f=out if pid==fixedpid else measure(fixedcurve,st['segments'],r,fixedpid,st['roi'],cfg,arm)
            fixed[arm]={k:copy.deepcopy(f[k]) for k in ('values','selection','PA_ids','PA_invalid','stability_status')}
    return arms,fixed

def originals_run(seed,limit=None):
    verify_binding();cfg=j(CFG);manifest=j(BASE/'inference.json');assert sha(BASE/'inference.json')==j(BASE/'inference_success.json')['manifest_sha256']
    idx={r['key']:r for r in manifest['rows'] if r['network']=='normalized' and r['transform']['name']=='identity'};ins={key(r):r for r in inputs()};rois=j(BASE/'roi.json')
    cv2.setNumThreads(1);total=0;start=time.time()
    for ds in DS:
        dest=BASE/f'predictions/{seed}/{ds}.json.gz'
        if dest.exists() and not limit:
            assert sha(dest)==dest.with_suffix('.sha256').read_text().strip();print('EXISTS',seed,ds,flush=True);continue
        rr=gzread(OLD/f'predictions/{seed}/{ds}.json.gz');out=[]
        for r in (rr[:limit] if limit else rr):
            k=key(r);sr=ins[k];oldmask=binary(ROOT/sr['mask_path']);info=idx[k];p=BASE/info['path'];assert sha(p)==info['sha256']
            with np.load(p) as z:newmask=z['mask']
            old,ex=states(oldmask,rois[k],cfg);new,_=states(newmask,rois[k],cfg)
            arms,fixed=run_row(r,old,new,cfg,True)
            q={k:r[k] for k in ('dataset','image_id','width','height','seed','px_per_mm','group_id','image_path','pixel_sha256')}
            q.update(arms=arms,fixed_boundary=fixed,pool=r['pool'],baseline_replayed=True,n_components={'old':ex['n_components'],'normalized':len(new['old']['audit'])})
            out.append(q);total+=1
            if len(out)%10==0:print('PRED',seed,ds,len(out),len(rr),'seconds',round(time.time()-start),flush=True)
        if limit:dest=BASE/f'smoke/{seed}/{ds}.json.gz'
        gzwrite(dest,out)
        if not limit:dest.with_suffix('.sha256').write_text(sha(dest),encoding='ascii')
    if not limit:status(f'predictions_{seed}',{'files':{d:sha(BASE/f'predictions/{seed}/{d}.json.gz') for d in DS},'records':469,'arms':ARMS,'references_read':False})
    print('PREDICTIONS DONE',seed,total,flush=True)

def transformed_run(seed):
    verify_binding();cfg=j(CFG);info=j(BASE/'inference.json');rr=originals(seed);ins={key(r):r for r in inputs()}
    transformed={}
    for p in info['rows']:
        if p['transform']['name']=='identity':continue
        transformed.setdefault((p['key'],p['transform']['name']),{})[p['network']]=p
    cv2.setNumThreads(1);out=[];start=time.time()
    for i,((k,name),pair) in enumerate(transformed.items()):
        path=BASE/f"stability_cache/{seed}/{name}/{ins[k]['dataset']}/{Path(ins[k]['image_id']).stem}.json.gz"
        if path.exists():out.append(gzread(path));continue
        r=rr[k];t=pair['old']['transform'];ss={}
        for net in ('old','normalized'):
            p=pair[net];f=ROOT/p['root_path'] if 'root_path' in p else BASE/p['path'];assert sha(f)==p['sha256']
            with np.load(f) as z:mask=z['mask']
            ss[net],_=states(mask,p['roi'],cfg,t,(r['width'],r['height']))
        arms,fixed=run_row(r,ss['old'],ss['normalized'],cfg,False)
        rec=dict(key=k,seed=seed,transform=name,arms={},fixed_boundary=fixed,roi=ss['old']['old']['roi'])
        for a,v in arms.items():
            rec['arms'][a]={f:v[f] for f in ('values','selection','PA_ids','PA_invalid','stability_status','selected_pair_id')}
            rec['arms'][a]['segments']=[c['segment'] for c in v['candidates']]
            rec['arms'][a]['filter_audit']=v['filter_audit']
        gzwrite(path,rec);out.append(rec)
        if (i+1)%15==0:print('TRANSFORMS',seed,i+1,len(transformed),'seconds',round(time.time()-start),flush=True)
    gzwrite(BASE/f'stability/{seed}.json.gz',out);status(f'stability_{seed}',{'records':len(out),'sha256':sha(BASE/f'stability/{seed}.json.gz')})
    print('TRANSFORMS DONE',seed,len(out),flush=True)

def analytic():
    cfg=j(CFG);rois=j(BASE/'roi.json');out=[]
    for i,r in enumerate(inputs()):
        _,raw=raw_components(binary(ROOT/r['mask_path']));roi=rois[key(r)]
        for scale in (.75,1.25):
            roi2={**roi,'box':[x*scale for x in roi['box']]};t={'sx':scale,'sy':scale,'tx':0.,'ty':0.}
            rec={'key':key(r),'scale':scale,'changes':{}}
            for soft in (False,True):
                changed=[s['label_id'] for s in raw if gate(s,roi,cfg,soft)[0]!=gate(map_segment(s,t),roi2,cfg,soft)[0]]
                assert not changed,'analytic gate invariance';rec['changes']['F1' if soft else 'F0']=changed
            out.append(rec)
        if (i+1)%100==0:print('ANALYTIC',i+1,469,flush=True)
    write(BASE/'analytic.json',{'rows':out,'all_exact_gate_parity':True,'no_bitmap_or_model':True})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['original','transforms','analytic']);p.add_argument('--seed',type=int,default=SEEDS[0]);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.stage=='original':originals_run(a.seed,a.limit)
    elif a.stage=='transforms':transformed_run(a.seed)
    else:analytic()
