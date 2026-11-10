"""Exact component pixel enumeration in one scan; no numerical rule change."""
from scripts.round43_cpu_imports import install as cpu_install
cpu_install()
import inspect,weakref,sys
from collections import OrderedDict
import numpy as np
PIXELS=OrderedDict()

def component_pixels(lab,sid):
    k=id(lab);entry=PIXELS.get(k)
    if entry is None or entry[0]() is not lab:
        yy,xx=np.nonzero(lab);labels=lab[yy,xx];order=np.argsort(labels,kind='stable');ss=labels[order];cuts=np.r_[0,np.flatnonzero(np.diff(ss))+1,len(ss)];out={}
        for a,b in zip(cuts[:-1],cuts[1:]):
            if a==b:continue
            ii=order[a:b];out[int(ss[a])]=(yy[ii],xx[ii])
        entry=(weakref.ref(lab),out);PIXELS[k]=entry
        while len(PIXELS)>8:PIXELS.popitem(last=False)
    return entry[1].get(int(sid),(np.array([],int),np.array([],int)))

def install():
    import scripts.round47_core as module
    funcs={}
    for name in ('audit_original','selection_rebalance'):
        old=getattr(module,name)
        if getattr(old,'_round47_fast',False):continue
        source=inspect.getsource(old);source=source.replace('np.nonzero(lab==sid)','component_pixels(lab,sid)').replace('np.nonzero(lab==int(sid))','component_pixels(lab,int(sid))')
        ns=dict(old.__globals__);ns['component_pixels']=component_pixels;exec(compile(source,'<round47_exact_pixel_enumeration>','exec'),ns);new=ns[name];new._round47_fast=True;funcs[old]=new
    for name,m in list(sys.modules.items()):
        if not name.startswith('scripts.') or m is None:continue
        for k,v in list(vars(m).items()):
            try:new=funcs.get(v)
            except TypeError:continue
            if new is not None:setattr(m,k,new)

def initialize():
    from scripts.round47_measure import initialize as base
    base();install()

def verify():
    from scripts.round47_common import rows,SEEDS,DEST,sha,write,ROOT,gzread
    import scripts.round47_measure as module
    module.initialize();old=module.execute;ns=dict(vars(module));source=inspect.getsource(old).replace('dst=DEST/f','dst=DEST/"fast_verify"/f',1);exec(compile(source,'<round47_full_output_acceleration_verify>','exec'),ns)
    selected=rows('development')[:3];initialize();checks=[]
    for r in selected:
        for s in SEEDS:
            ns['selection_rebalance']=module.selection_rebalance;ns['audit_original']=module.audit_original;ns['execute'](r,s,'pilot')
            from pathlib import Path
            p=DEST/f'pilot/{s}/{r["dataset"]}/{Path(r["image_id"]).stem}.json.gz';a=gzread(p);b=gzread(DEST/'fast_verify'/p.relative_to(DEST));assert a['variants']==b['variants'] and a['sources']==b['sources'];checks.append(dict(image=r['image_id'],seed=s,exact=True,variants=len(a['variants'])))
    lab=np.array([[0,2,0,1],[2,1,0,1],[0,2,1,0]],int)
    for sid in (0,1,2,3):
        if sid==0:continue
        a=np.nonzero(lab==sid);b=component_pixels(lab,sid);assert all(np.array_equal(x,y) for x,y in zip(a,b))
    write(DEST/'fast_exact_verification.json',dict(passed=True,source=sha(ROOT/'scripts/round47_fast.py'),checks=checks,outputs=sum(x['variants']*8 for x in checks),operation='Replace repeated full-array label equality/nonzero scans with stable grouped enumeration. Exact original row-major pixel order, same exposure arithmetic, no change to model, geometry, thresholds, weights, labels or output.'))
    print('R47_FAST_EXACT_PASS',flush=True)

if __name__=='__main__':verify()
