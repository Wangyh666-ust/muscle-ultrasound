"""Exact foreground-only label reads for PCA and boundary C1 evidence."""
from scripts.round43_cpu_imports import install as cpu_install
cpu_install()
import inspect,sys,types
from scripts.round47_fast import component_pixels,install as install_v1

def initialize():
    from scripts.round47_measure import initialize as base
    from scripts.fascicle_geometry import extract_segments
    from scripts.round36_core import prepare_evidence
    changes={extract_segments:[('np.nonzero(labels == lab)','component_pixels(labels,lab)')],prepare_evidence:[('ev=np.isin(labs,ids);ys,xs=np.nonzero(ev)','fy,fx=np.nonzero(labs);labels=labs[fy,fx];ev=np.isin(labels,ids);ys,xs=fy[ev],fx[ev]'),('yy,xx=np.nonzero(np.isin(labs,g));pixels[s[\'label_id\']]=(xx,yy)','member=np.isin(labels,g);pixels[s[\'label_id\']]=(fx[member],fy[member])')]}
    mapping={}
    for fn,edits in changes.items():
        if getattr(fn,'_round47_fast_v2',False):continue
        src=inspect.getsource(fn)
        for a,b in edits:assert src.count(a)==1,a;src=src.replace(a,b)
        ns=dict(fn.__globals__);ns['component_pixels']=component_pixels;exec(compile(src,'<round47_foreground_only_label_reads>','exec'),ns);q=ns[fn.__name__];q._round47_fast_v2=True;mapping[fn]=q
    spaces=[vars(m) for name,m in list(sys.modules.items()) if name.startswith('scripts.') and m is not None];seen=set()
    while spaces:
        ns=spaces.pop()
        if id(ns) in seen:continue
        seen.add(id(ns))
        for k,v in list(ns.items()):
            if isinstance(v,types.FunctionType) and v.__module__.startswith('scripts.'):
                spaces.append(v.__globals__)
                if v in mapping:ns[k]=mapping[v]
    base();install_v1()

def verify():
    from scripts.round47_common import rows,SEEDS,DEST,ROOT,Path,gzread,write,sha
    import scripts.round47_measure as module
    initialize();ns=dict(vars(module));src=inspect.getsource(module.execute).replace('dst=DEST/f','dst=DEST/"fast_v2_verify"/f',1);exec(compile(src,'<round47_fast_v2_verify>','exec'),ns)
    r=next(r for r in rows() if r['dataset']=='dev282' and r['image_id']=='image_0566.tif');tasks=[(rows('development')[0],s,'pilot') for s in SEEDS]+[(r,SEEDS[0],'all')];checks=[]
    for r,s,phase in tasks:
        ns['execute'](r,s,phase);p=DEST/f"{'pilot' if phase=='pilot' else 'measurements'}/{s}/{r['dataset']}/{Path(r['image_id']).stem}.json.gz";a=gzread(p);b=gzread(DEST/'fast_v2_verify'/p.relative_to(DEST));assert a['variants']==b['variants'] and a['sources']==b['sources'];checks.append(dict(image=r['image_id'],seed=s,phase=phase,variants=len(a['variants']),exact=True))
    write(DEST/'fast_v2_exact_verification.json',dict(passed=True,source=sha(ROOT/'scripts/round47_fast_v2.py'),prior=sha(ROOT/'scripts/round47_fast.py'),checks=checks,outputs=sum(x['variants']*8 for x in checks),operation='Read the same row-major pixels of the same labeled components without repeated full-canvas scans. No changed fits, scores, filters, eligibility, weights or measurements.'))
    print('R47_FAST_V2_EXACT_PASS',flush=True)

if __name__=='__main__':verify()
