"""Exact caching of repeated C1 labeling/evidence and identical risk probes."""
import sys
from scripts.round43_fast import memo,COUNTS

def install():
    from scripts.round42_fast_v2 import install as old
    old()
    import scripts.round40_core as c40
    import scripts.round36_core as c36
    import scripts.round43_decoder as dec
    mapping={f:memo(f,n) for f,n in ((c40.propagated_risk,256),(c36.prepare_evidence,16)) if not getattr(f,'_round43_memo',False)}
    namespaces=[vars(m) for name,m in list(sys.modules.items()) if name.startswith('scripts.') and m is not None]
    namespaces.append(dec.ELIG.__globals__)
    for ns in namespaces:
        for name,obj in list(ns.items()):
            if callable(obj):
                try:new=mapping.get(obj)
                except TypeError:continue
                if new is not None:ns[name]=new
