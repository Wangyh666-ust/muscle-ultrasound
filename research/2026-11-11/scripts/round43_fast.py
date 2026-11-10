"""Exact memoization of pure geometry functions; no numerical changes."""
import functools,pickle,copy,hashlib,sys
COUNTS={}

def memo(fn,maximum=256):
    from collections import OrderedDict
    cache=OrderedDict();count=COUNTS.setdefault(fn.__name__,dict(hits=0,misses=0))
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        key=hashlib.sha256(pickle.dumps((args,kwargs),protocol=5)).digest()
        if key in cache:
            count['hits']+=1;cache.move_to_end(key);return copy.deepcopy(cache[key])
        count['misses']+=1;out=fn(*args,**kwargs);cache[key]=copy.deepcopy(out)
        if len(cache)>maximum:cache.popitem(last=False)
        return out
    wrapped._round43_memo=True
    return wrapped

def install():
    from scripts.round42_fast_v2 import install as old
    old()
    import scripts.round40_core as c40
    import scripts.round29_core as c29
    funcs=[c40.propagated_risk,c29.extended_fl];mapping={f:memo(f) for f in funcs if not getattr(f,'_round43_memo',False)}
    for name,module in list(sys.modules.items()):
        if not name.startswith('scripts.') or module is None:continue
        for field,obj in list(vars(module).items()):
            if callable(obj):
                try:new=mapping.get(obj)
                except TypeError:continue
                if new is not None:setattr(module,field,new)
