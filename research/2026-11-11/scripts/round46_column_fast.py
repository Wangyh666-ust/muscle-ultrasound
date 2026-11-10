"""Exactly equivalent contiguous sparse column enumeration, no gap-rule change."""
import sys,types
import numpy as np

def centres(mask):
    h,w=mask.shape;out=[[] for _ in range(w)];gap=max(2,round(h*.002))
    flat=np.flatnonzero(np.ascontiguousarray(mask.T))
    if not len(flat):return out
    xs=flat//h;ys=flat%h;cuts=np.r_[0,np.flatnonzero((np.diff(xs)!=0)|(np.diff(ys)>gap))+1];ends=np.r_[cuts[1:],len(flat)];n=ends-cuts
    lo=cuts+(n-1)//2;hi=cuts+n//2;values=(ys[lo]+ys[hi])/2.
    for x,v in zip(xs[cuts].tolist(),values.tolist()):out[x].append(v)
    return out

def install():
    import scripts.round39_core as core
    target=core.run_centres_per_column
    if target is centres:return
    spaces=[vars(m) for name,m in list(sys.modules.items()) if name.startswith('scripts.') and m is not None];seen=set()
    while spaces:
        ns=spaces.pop()
        if id(ns) in seen:continue
        seen.add(id(ns))
        for k,v in list(ns.items()):
            if isinstance(v,types.FunctionType) and v.__module__.startswith('scripts.'):
                spaces.append(v.__globals__)
                if v is target:ns[k]=centres
