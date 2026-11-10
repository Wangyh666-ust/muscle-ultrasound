"""Candidate fitting interventions with the entire frozen K downstream chain."""
from scripts.round42_common import *
from scripts.round40_core import (normalized_pool,masked,smooth_extensions,eligibility,evidence_gates,measure_soft,measure_pool,
    suppress,modify_fl,apply_u)
from scripts.round38_speed import install
from scripts.round39_verify import check_equal

def fit_state(mask,v,method):
    segments=[];audit=[]
    for c in v['components']:
        s=copy.deepcopy(c['segment']);s['filter_quality']=c['filter']['quality'];s['filter_evidence']=copy.deepcopy(c['filter'])
        keep=c['accepted'] or (method=='F' and c['filter']['flags']==['length_lt_30_reference'])
        audit.append(dict(id=s['label_id'],accepted=keep,**c['filter']))
        if not keep:continue
        if keep and not c['accepted']:s['round42_length_rescued']=True
        if method in c['fits']:
            fit=c['fits'][method]
            for field in ('direction_deg','centroid','p0','p1','length_px','width_px'):s[field]=copy.deepcopy(fit[field])
        segments.append(s)
    labs,_=ndimage.label(mask,structure=np.ones((3,3),bool));evidence=np.isin(labs,[s['label_id'] for s in segments])
    return dict(segments=segments,evidence=evidence,raw_mask=mask,audit=audit,roi=v['roi'],raw_count=len(v['components']))

def predict(pool,pt,st,native,cfg,cal):
    gates=eligibility(pool,st,native);gates=evidence_gates(pool,st,native,cfg,cal,gates)
    a=measure_soft(pt,st,native,cfg,gates)
    if a['curve'] is None:
        exp,_=measure_pool(pt,st,native,cfg);a=suppress(a,exp)
    a=modify_fl(a,cfg,cal,True,True);a['_native_dimensions']=native
    return apply_u(a,cfg,cal)

def execute(r,seed,methods=('O1','O2'),pilot=False):
    cv2.setNumThreads(1);install();cfg=j(CFG);cal=j(R40/'calibration.json');old=gzread(old_resultpath(r,seed));ex=gzread(extractpath(r))
    folder='pilot/'+ '-'.join(methods) if pilot else 'measurements'
    dst=DEST/f"{folder}/{seed}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz"
    binding=dict(core=sha(Path(__file__)),extraction=sha(extractpath(r)),baseline=sha(old_resultpath(r,seed)),methods=list(methods))
    if not pilot:
        lock=j(DEST/'implementation_lock.json');binding['lock']=sha(DEST/'implementation_lock.json');assert list(methods)==lock['methods']
    if dst.exists():
        meta=j(dst.with_suffix('.sha.json'));assert meta['binding']==binding and meta['sha256']==sha(dst);return dict(key=key(r),seed=seed,resumed=True)
    row={k:v for k,v in old.items() if k not in ('variants','sources')};variants={};start=time.time();parities=[]
    for tr,q in old['variants'].items():
        bm,br,_=getboundary(r,tr,seed);roi=ex['models']['F'][tr]['roi'];t=ex['models']['F'][tr]['transform'];native=dict(width=t['width'],height=t['height'],px_per_mm=None)
        pool=normalized_pool(masked(bm,roi),roi);pt=smooth_extensions(pool,roi);arms={}
        for kind in KINDS:
            v=ex['models'][kind][tr];mask=unpack(inferpath(r,tr,kind))[0]
            arms[kind+'|K']=q['arms'][kind]
            if pilot:
                st=fit_state(mask,v,'K');a=finish(predict(pool,pt,st,native,cfg,cal),t,old)
                for field in ('values','curve','FL_weights','PA_weights','selected_pair_id'):check_equal(a[field],q['arms'][kind][field],'R42 baseline replay '+field)
                parities.append(dict(transform=tr,kind=kind,passed=True))
            for method in methods:
                st=fit_state(mask,v,method);a=predict(pool,pt,st,native,cfg,cal)
                arms[kind+'|'+method]=finish(a,t,old)
        variants[tr]=dict(arms=arms,transform=t)
    row.update(variants=variants,methods=list(methods),seconds=time.time()-start,replays=parities)
    gzwrite(dst,row);write(dst.with_suffix('.sha.json'),dict(sha256=sha(dst),binding=binding))
    return dict(key=key(r),seed=seed,seconds=row['seconds'],variants=len(variants),replays=len(parities))
