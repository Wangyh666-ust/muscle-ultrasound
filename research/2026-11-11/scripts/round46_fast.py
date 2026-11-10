"""Memoize pure extraction/coverage calls only; original arithmetic retained."""
import copy,functools,hashlib,pickle,sys,types
from collections import OrderedDict
COUNTS={}

def memo(fn,n):
    cache=OrderedDict();counts=COUNTS.setdefault(fn.__name__,dict(hit=0,miss=0))
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        k=hashlib.sha256(pickle.dumps((args,kwargs),protocol=5)).digest()
        if k in cache:counts['hit']+=1;cache.move_to_end(k);return copy.deepcopy(cache[k])
        counts['miss']+=1;q=fn(*args,**kwargs);cache[k]=copy.deepcopy(q)
        if len(cache)>n:cache.popitem(last=False)
        return q
    wrapped._round46_memo=True
    return wrapped

def install():
    import scripts.round39_core as core
    import scripts.round38_core as c38
    from scipy import ndimage
    targets=[(core.coverage,8192),(core.run_centres_per_column,16),(c38.raw_components,16)]
    mapping={fn:memo(fn,n) for fn,n in targets if not getattr(fn,'_round46_memo',False)}
    spaces=[vars(m) for name,m in list(sys.modules.items()) if name.startswith('scripts.') and m is not None];seen=set()
    while spaces:
        ns=spaces.pop()
        if id(ns) in seen:continue
        seen.add(id(ns))
        for name,obj in list(ns.items()):
            if isinstance(obj,types.FunctionType) and obj.__module__.startswith('scripts.'):
                spaces.append(obj.__globals__)
                if obj in mapping:ns[name]=mapping[obj]
    if not getattr(ndimage.label,'_round46_memo',False):ndimage.label=memo(ndimage.label,8)
