"""Explicit S/Q interventions; old coordinate frame and all unrelated gates fixed."""
import inspect
from scripts.round43_common import *
import scripts.round39_core as core
from scripts.round40_core import smooth_extensions,evidence_gates,measure_soft,measure_pool,suppress,modify_fl,apply_u
from scripts.round25_boundary_fit import _design

def fit_individual(pts,h,w):
    a=np.asarray(pts,float);lo,hi=a[0,0],a[-1,0];A,_=_design(a[:,0],lo,hi)
    if A is None or np.linalg.matrix_rank(A)<5:return None
    knots=np.linalg.lstsq(A,a[:,1],rcond=None)[0];p90=float(np.quantile(abs(A@knots-a[:,1]),.9))
    if p90>max(3,.02*h) or not np.isfinite(knots).all():return None
    return dict(knots_u=[0,.25,.5,.75,1],knots_x=np.linspace(lo,hi,5).tolist(),knots_y=knots.tolist(),support=[lo,hi],paired_x=a[:,0].tolist(),max_sample_gap=max(2,.015*w),image_shape=[h,w],residual_p90_px=p90,rank=5,n_observations=len(a))

def build_functions():
    # Frozen source copied into a separate namespace; only designated strings change.
    ns=dict(vars(core))
    baseline=inspect.getsource(core.baseline_normalized).replace('def baseline_normalized(columns,raw,w,h,pitch):','def baseline_normalized(columns,raw,w,h,pitch,qwidth=None):\n    qw=w if qwidth is None else qwidth')
    baseline=baseline.replace('.3*w','.3*qw');exec(compile(baseline,'<round43_baseline_policy>','exec'),ns)
    src=inspect.getsource(core.normalized_pool_from_columns)
    src=src.replace('pitch_native=1.):','pitch_native=1.,independent=False,qual_width=None):')
    src=src.replace('cols=[','qw=w if qual_width is None else qual_width*factor\n    audit=[]\n    cols=[',1)
    src=src.replace('def append(pts,d):','def append(pts,d,a=None,b=None):')
    src=src.replace('cv=fit_float(pts,h,w)','cv=fit_float(pts,h,w)\n        own=None\n        if independent and a is None:\n            aa=[m for m in maps[0] if all(x in m and abs(m[x]-y)<1e-8 for x,y,z in pts)]\n            bb=[m for m in maps[0] if all(x in m and abs(m[x]-z)<1e-8 for x,y,z in pts)]\n            if aa and bb:a,b=max(aa,key=len),max(bb,key=len)\n        if independent and a is not None and b is not None:\n            own={"sup":sorted(a.items()),"deep":sorted(b.items())}\n            fits={k:fit_individual(v,h,w) for k,v in own.items()}\n            if cv is not None and all(fits.values()):\n                cv.update(fits)\n                test=sorted(set(xs for k in own for xs,y in own[k]))\n                common=[x for x in test if max(fits["sup"]["support"][0],fits["deep"]["support"][0])<=x<=min(fits["sup"]["support"][1],fits["deep"]["support"][1])]\n                if any(np.interp(x,fits["sup"]["knots_x"],fits["sup"]["knots_y"])>=np.interp(x,fits["deep"]["knots_x"],fits["deep"]["knots_y"]) for x in common):cv=None\n                else:cv=add_extensions(cv,pts,BCFG)\n            else:cv=None')
    src=src.replace('quality=qual,points=pts)','quality=qual,points=pts,own_observations=own)')
    # Independent fits have different knot abscissae. Checking their union is
    # necessary and sufficient for ordering of two piecewise-linear curves.
    src=src.replace('test=sorted(set(xs for k in own for xs,y in own[k]))','test=sorted({x for k in own for x,y in own[k]} | {float(x) for k in fits for x in fits[k]["knots_x"]})')
    src=src.replace('baseline_normalized(cols[0],raw[0],w,h,pitch)','baseline_normalized(cols[0],raw[0],w,h,pitch,qw)')
    src=src.replace('.3*w','.3*qw').replace('append(pts,d)','append(pts,d,a,b)')
    src=src.replace('xs=sorted(a.keys()&b.keys())','xs=sorted(a.keys()&b.keys())\n            audit.append(dict(decoder=d,upper_coverage=coverage(list(a),pitch),lower_coverage=coverage(list(b),pitch),common_coverage=coverage(xs,pitch),common_span=(xs[-1]-xs[0]) if xs else 0,threshold=.3*qw))')
    src=src.replace("p['ranking_height']=y1-y0","p['ranking_height']=y1-y0\n        own=pools[p['source_decoder']][p['source_pair_id']].get('own_observations')\n        if own:p['own_observations']={k:[(x/factor+x0,y/factor+y0) for x,y in v] for k,v in own.items()}")
    src=src.replace('return merged','return merged,audit')
    ns['fit_individual']=fit_individual;exec(compile(src,'<round43_support_policy>','exec'),ns)
    elig=inspect.getsource(core.eligibility).replace('def eligibility(pool,st,row):','def eligibility(pool,st,row,qual_width=None):')
    elig=elig.replace("span=st['roi']['box'][2]-st['roi']['box'][0]","span=qual_width if qual_width is not None else st['roi']['box'][2]-st['roi']['box'][0]")
    exec(compile(elig,'<round43_final_qualification>','exec'),ns)
    return ns['normalized_pool_from_columns'],ns['eligibility'],dict(pool=src,eligibility=elig,baseline=baseline)

POOL,ELIG,ADAPTED=build_functions()

def pool(masks,roi,independent=False,qual_width=None):
    cols=[list(enumerate(core.run_centres_per_column(m))) for m in masks]
    return POOL(cols,roi,independent=independent,qual_width=qual_width)

def rays(pairs,roi,independent=False):
    out=smooth_extensions(pairs,roi)
    if not independent:return out
    span=roi['box'][2]-roi['box'][0]
    for p in out:
        if not p.get('own_observations'):continue
        for side,pts in p['own_observations'].items():
            a=np.asarray(pts);f=p['curve'][side]
            for at,name in ((0,'left_ray'),(-1,'right_ray')):
                x=a[at,0];w=np.exp(-.5*((a[:,0]-x)/(.25*span))**2);A=np.column_stack([a[:,0]-x,np.ones(len(a))]);fit=np.linalg.lstsq(A*np.sqrt(w[:,None]),a[:,1]*np.sqrt(w),rcond=None)[0]
                f[name]=[float(fit[0]),float(f['knots_y'][at]-fit[0]*x)]
    return out

def predict(p,pt,st,native,cfg,cal,qual_width=None):
    gates=ELIG(p,st,native,qual_width);gates=evidence_gates(p,st,native,cfg,cal,gates)
    a=measure_soft(pt,st,native,cfg,gates)
    if a['curve'] is None:
        exp,_=measure_pool(pt,st,native,cfg);a=suppress(a,exp)
    a=modify_fl(a,cfg,cal,True,True);a['_native_dimensions']=native
    return apply_u(a,cfg,cal)

def controlled():
    rr=dict(box=[0,0,1000,600]);cases={};xs=np.arange(100,900)
    for name in ('full','one_short','staggered','gap','parallel','no_upper'):
        masks=np.zeros((3,600,1000),bool);up=xs.copy();down=xs.copy()
        if name=='one_short':down=xs[xs>550]
        if name=='staggered':up=xs[xs<650];down=xs[xs>350]
        if name=='gap':up=xs[(xs<400)|(xs>410)]
        if name=='no_upper':up=[]
        masks[1,150,up]=1;masks[2,350,down]=1;masks[0]=masks[1]|masks[2]
        if name=='parallel':masks[0,300,xs]=1;masks[2,300,xs]=1
        result={}
        for mode,s,q in (('R',False,None),('S',True,None),('Q',False,800),('SQ',True,800)):
            p,a=pool(masks,rr,s,q);pt=rays(p,rr,s);result[mode]=dict(count=len(p),audit=a,support=[{k:z['curve'][k]['support'] for k in ('sup','deep')} for z in pt])
        if name=='no_upper':assert all(z['count']==0 for z in result.values())
        if name=='full':assert result['R']['count'] and result['S']['count']
        if name=='one_short':assert any(z['sup'][0]<z['deep'][0] for z in result['S']['support'])
        cases[name]=result
    write(DEST/'control_support.json',cases);write(DEST/'adapted_decoder_sources.json',ADAPTED)
    return cases

if __name__=='__main__':controlled();print('SUPPORT_CONTROLS_PASS')
