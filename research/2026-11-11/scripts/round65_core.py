"""Round65 arm isolation: unchanged fields, ranker baseline and fixed C1."""
from copy import deepcopy
import numpy as np
from scripts.round63_paths import decode,choose_pair,path_quality,image_evidence
from scripts.round64_seams import reconstruct as reconstruct_s
from scripts.round65_geometry import narrow_reconstruct,retreat_reconstruct,extensions,protected_candidate,protected_score


def keep(old,new,width):return list(old)+sorted(new,key=lambda p:-path_quality(p,width))[:10]


def protected_select(pool,reference,width,cfg,base,segments,scale):
    if reference is None:return choose_pair(pool,width,base,segments,scale,True,True)[0],dict(no_reference=True)
    guarded=[];rejections={}
    for paths,side in zip(pool,('upper','lower')):
        sides=[reference[side]]
        for path in paths:
            q,why=protected_candidate(path,reference[side],width,cfg)
            if q is not None:sides.append(q)
            else:rejections[why]=rejections.get(why,0)+1
        guarded.append(sides)
    _,pairs=choose_pair(guarded,width,base,segments,scale,True,True)
    if not pairs:return reference,dict(rejections=rejections)
    for p in pairs:p['protected_score']=protected_score(p,width,cfg)
    original=protected_score(reference,width,cfg)
    best=max(pairs,key=lambda p:p['protected_score'])
    result=deepcopy(best if best['protected_score']>original+cfg['replacement_margin']+1e-12 else reference)
    return result,dict(rejections=rejections,original_protected_score=original,
        best_protected_score=best['protected_score'],replaced=result is not reference and result.get('protected_score',original)>original)


def process(probability,image,valid,width,base,configs,segments=(),scale=None,modules=('N','T','P','E')):
    old=decode(probability,image,valid,width,base,segments,scale);raw=old['tracks']['rebuilt']
    gray=image_evidence(image,width);arms={'R':old['arms']['P']};pools={a:[] for a in ('S','N','N_T','N_E','N_T_P_E')}
    audits={};exploratory=[]
    for i,side in enumerate(('upper','lower')):
        field=probability[i+1]*(.4+.6*probability[0])
        s,sa=reconstruct_s(raw[i],field,gray,valid,width,configs['N'],base)
        n,na=narrow_reconstruct(raw[i],field,gray,valid,width,configs['N'],base)
        pools['S'].append(keep(raw[i],s,width));npool=keep(raw[i],n,width);pools['N'].append(npool)
        t,ta=retreat_reconstruct(raw[i],field,gray,valid,width,configs['T'],base) if 'T' in modules else ([],{})
        tp=keep(npool,t,width);pools['N_T'].append(tp)
        e=extensions(raw[i],field,gray,valid,width,configs['E']) if 'E' in modules else []
        pools['N_E'].append(keep(npool,e,width))
        te=extensions(tp,field,gray,valid,width,configs['E']) if 'E' in modules else []
        pools['N_T_P_E'].append(keep(tp,te,width))
        exploratory.append(extensions([old['arms']['P'][side]],field,gray,valid,width,configs['E'],True) if old['arms']['P'] and 'E' in modules else [])
        audits[side]=dict(N_added=len(n),T_added=len(t),E_added=len(e),N_rejections=na,T_rejections=ta)
    for a in ('S','N','N_T','N_E'):arms[a]=choose_pair(pools[a],width,base,segments,scale,True,True)[0]
    for a,poolname in [('N_P','N'),('N_T_P','N_T'),('N_T_P_E','N_T_P_E')]:
        arms[a],audit=protected_select(pools[poolname],arms['R'],width,configs['P'],base,segments,scale);audits[a]=audit
    return dict(arms=arms,audit=audits,exploratory=exploratory,pools=pools)
