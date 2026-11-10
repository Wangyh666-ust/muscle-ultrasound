"""Lock FL qualification on development, replay the frozen 26-view suite."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import time

from scripts.round53_experiment import (
    DEST as R53, ROOT, R41, CFG, SEEDS, KINDS, SPLIT, NATURAL, TRANSFORMS, Path, check_equal,
    source_path, load_fascicles, initialize_worker, settings, key, rows, j, sha, write, gzread, gzwrite, np,
)
from scripts.round53_boundary_attribution import compare, near
from scripts.round54_fl_support import (
    boundary_mask, reconstruct, native_measure, audit, controls, eligible, qualifying, subset_fl,
)
from scripts.round48_evaluate import error
from scripts.round25_evaluate import REF, f1_score, validate_reference


DEST = ROOT/'runs/round54_short_boundary_fl_v1'
PANEL = ('identity','translate-2_1','r41_scale1015','brightness105')
DIAG_IMAGES = {'image_1707.tif','im_12_arch','image_1777.tif','image_0642.tif'}
GRID = tuple(product((.02,.05),(.60,.80)))
POLICIES = ('A','B','P','S','SR','F80','X')


def tag(distance,mass,risk):
    return f'{int(100*distance)}:{int(100*mass)}:{int(risk)}'


def ref_index():
    return validate_reference(REF,{r['image_id']:r for r in rows() if r['dataset']=='apo153'})[0]


def source(r,seed):
    folder = 'development' if SPLIT['assignment'][key(r)]=='development' else 'locked'
    return R53/'measurements'/folder/str(seed)/r['dataset']/(Path(r['image_id']).stem+'.json.gz')


def audit_path(r,seed):
    return DEST/'audit'/str(seed)/r['dataset']/(Path(r['image_id']).stem+'.json.gz')


def points_to_ref(tracks,t,ref):
    out = {}
    for side in ('sup','deep'):
        checked = []
        xs,ys = ref['reference_x'],ref['reference_'+side]
        for track in tracks:
            x,y = track['endpoints'][side]['point']
            x,y = (x-t['tx'])/t['sx'],(y-t['ty'])/t['sy']
            if xs[0]<=x<=xs[-1]:
                checked.append([track['weight'],abs(y-np.interp(x,xs,ys))])
        weight = sum(z[0] for z in checked)
        out[side] = dict(weight=weight, weighted_error=sum(w*d for w,d in checked)/weight if weight else None)
    return out


def case_metrics(pair, maps, factor, offset, width, segs, raw, roi, t, old, frozen, grid, ref):
    native,mapped = native_measure(pair,segs,len(raw),roi,t,old)
    check_equal(mapped['values'],frozen['values'],'Round53 exact measurement')
    ev = audit(pair,maps,factor,offset,width,native)
    tracks = ev['tracks']
    values = {}
    for distance,mass in grid:
        for risk in (False,True):
            passed = qualifying(tracks,distance,mass,risk)
            values[tag(distance,mass,risk)] = dict(value=subset_fl(native,mapped,passed),
                      pass_mass=sum(q['weight'] for q in passed),pass_ids=[q['id'] for q in passed])
    strict = [q for q in tracks if q['extrap']<=.80]
    values['F80'] = dict(value=subset_fl(native,mapped,strict),pass_mass=sum(q['weight'] for q in strict))
    ev['outputs'] = values
    if ref:
        ev['reference_correct'] = bool(f1_score(dict(status='candidate',curve=mapped['curve']),ref)[0])
        ev['reference_endpoints'] = points_to_ref(tracks,t,ref)
    return ev


def task(r,seed,requested,grid,ref,protocol_id):
    p = audit_path(r,seed)
    if p.exists():
        q = gzread(p)
        assert q['protocol']==protocol_id and q['requested']==[list(x) for x in requested]
        return p
    cfg,_ = settings()
    frozen = gzread(source(r,seed))
    assert tuple(frozen['variants'])==TRANSFORMS
    condition = SPLIT['assignment'][key(r)]
    requested_set = set(requested)
    old = None
    result = {}
    for tr,by_kind in frozen['variants'].items():
        relevant = {k:v for k,v in by_kind.items() if v['A']!=v['Qlocal'] or (tr,k) in requested_set or
                    condition=='development' and tr in PANEL or tr=='identity' and r['dataset']=='apo153'}
        if not relevant:
            continue
        if old is None:
            old = gzread(source_path(r,seed))
        v = old['variants'][tr]
        t,roi,w = v['transform'],v['measurement_roi'],v['region']['reference_width']
        p_all,maps,factor,offset = reconstruct(boundary_mask(r,seed,tr,old),roi,w)
        pool = {q['pair_id']:q for q in p_all}
        assert len(pool)==v['boundary_audit']['ID']['candidate_count']
        result[tr] = {}
        for kind,arms in relevant.items():
            _,raw,segs = load_fascicles(r,tr,kind,old,roi,cfg)
            cache = {}
            vals = {}
            for group in ('A','Qlocal','Q25') if r['image_id']=='image_0642.tif' and tr=='scale095' else ('A','Qlocal'):
                a = arms[group]
                i = a['selected_pair_id']
                if i is None:
                    vals[group] = None
                    continue
                if i not in cache:
                    cache[i] = case_metrics(pool[i],maps,factor,offset,w,segs,raw,roi,t,old,a,grid,ref)
                vals[group] = cache[i]
            a,b = (arms[g]['selected_pair_id'] for g in ('A','Qlocal'))
            vals['same_layer'] = bool(a is not None and b is not None and near(compare(pool[a]['curve'],pool[b]['curve'],w),.10))
            result[tr][kind] = vals
    gzwrite(p,dict(key=key(r),seed=seed,protocol=protocol_id,requested=requested,variants=result))
    return p


def policy_values(a,b,ev,lock):
    av,bv = (o['values']['FL'] for o in (a,b))
    pp = av if av is not None else bv
    sa,sra,f80,x = av,av,av,av
    if ev is not None and ev['Qlocal'] is not None and bv is not None:
        distance,mass = lock['distance'],lock['mass']
        s,sr = (ev['Qlocal']['outputs'][tag(distance,mass,risk)] for risk in (False,True))
        if av is None:
            sa,sra,f80,x = s['value'],sr['value'],ev['Qlocal']['outputs']['F80']['value'],sr['value']
        elif ev['A'] is not None and ev['same_layer'] and a['selected_pair_id']!=b['selected_pair_id']:
            base = ev['A']['outputs'][tag(distance,mass,True)]
            if sr['value'] is not None and base['value'] is None:
                x = sr['value']
    return dict(A=av,B=bv,P=pp,S=sa,SR=sra,F80=f80,X=x)


def evaluate_dev(paths):
    index = {key(r):r for r in rows('development')}
    q = defaultdict(Counter)
    for path in paths:
        u = gzread(path)
        r = index[u['key']]
        frozen = gzread(source(r,u['seed']))['variants']
        for tr,by_kind in u['variants'].items():
            for k,ev in by_kind.items():
                a,b = [frozen[tr][k][g]['values']['FL'] for g in ('A','Qlocal')]
                if a is not None or b is None:
                    continue
                anchor = frozen['identity'][k]['A']['values']['FL']
                for distance,mass in GRID:
                    for risk in (False,True):
                        tok = tag(distance,mass,risk)
                        new = ev['Qlocal']['outputs'][tok]['value']
                        d = error('FL',anchor,new)
                        q[tok]['attempts'] += 1
                        q[tok]['retained'] += new is not None
                        q[tok]['comparable'] += d is not None
                        q[tok]['good'] += d is not None and d<=.20
                        q[tok]['severe'] += d is not None and d>.20
    return {k:dict(v) for k,v in q.items()}


def pick(q):
    safe = [(d,m) for d,m in GRID if q.get(tag(d,m,True),{}).get('good',0)>0 and
            q[tag(d,m,True)].get('severe',0)==0]
    if not safe:
        return .02,.80,'no demonstrated development recovery with an anchor; conservative predeclared fallback'
    d,m = min(safe,key=lambda z:(-q[tag(*z,True)]['good'],z[0],-z[1]))
    return d,m,'maximum nonsevere anchored development recoveries, zero observed severe; tie favors narrower distance and larger mass'


def protocol():
    p = DEST/'protocol.json'
    files = (Path(__file__),ROOT/'scripts/round54_fl_support.py',ROOT/'scripts/round54_fl_evaluate.py',
             ROOT/'scripts/round54_fl_occlusion.py',R53/'experiment_lock.json',R53/'verification.json',
             R53/'full_events.json.gz',R41/'partitions.json',REF,CFG)
    q = dict(sources={f.relative_to(ROOT).as_posix():sha(f) for f in files},controls=controls(),
            grid=[list(x) for x in GRID],scope=list(TRANSFORMS),policies=list(POLICIES),
            panel=list(PANEL),no_training=True,no_default_change=True,no_subagents=True)
    if p.exists():
        assert j(p)==q
    else:
        write(p,q)
    return p


def run(phase,workers):
    p = protocol()
    lock_path = DEST/'lock.json'
    if phase=='controls':
        print('Round54 controls passed',flush=True)
        return
    if phase=='full':
        lock = j(lock_path)
        assert lock['protocol']==sha(p)
    requested = defaultdict(set)
    if phase=='full':
        for e in gzread(R53/'full_events.json.gz'):
            requested[e['key'],e['seed']].update(((e['transform'],e['kind']),('identity',e['kind'])))
    refs = ref_index()
    dataset = rows('development' if phase=='development' else 'confirmation')
    cases = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as proc:
        pending = []
        for r in dataset:
            for s in SEEDS:
                extras = requested[key(r),s]
                if phase=='full' and r['image_id'] in DIAG_IMAGES:
                    extras.update((tr,k) for tr in TRANSFORMS for k in KINDS)
                ref = refs.get(r['image_id']) if r['dataset']=='apo153' else None
                pending.append(proc.submit(task,r,s,tuple(sorted(extras)),GRID if phase=='development' else ((lock['distance'],lock['mass']),),ref,sha(p)))
        for f in as_completed(pending):
            cases.append(f.result())
            if len(cases)%40==0:
                print('Round54',phase,len(cases),len(pending),round(time.time()-start),flush=True)
    if phase=='development':
        data = evaluate_dev(cases)
        distance,mass,reason = pick(data)
        write(DEST/'development.json',data)
        candidate = dict(distance=distance,mass=mass,reason=reason,protocol=sha(p),development_sha256=sha(DEST/'development.json'))
        if lock_path.exists():
            assert j(lock_path)==candidate
        else:
            write(lock_path,candidate)
        print('Round54 lock',candidate,flush=True)
    else:
        from scripts.round54_fl_evaluate import evaluate
        evaluate(lock)
    print('Round54 complete',phase,round(time.time()-start),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=('controls','development','full'))
    p.add_argument('--workers',type=int,default=12)
    args = p.parse_args()
    run(args.phase,args.workers)
