"""Round55 staged replay on the frozen Round52/53 prediction archives."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from itertools import product
from time import monotonic

from scripts.round54_fl_replay import (
    ROOT, R53, KINDS, SEEDS, SPLIT, TRANSFORMS, Path, key, rows, j, sha, write,
    gzread, gzwrite, source, source_path, settings, load_fascicles, initialize_worker,
    native_measure, check_equal, ref_index, np,
)
from scripts.round54_fl_support import boundary_mask, owned_columns, root_support
from scripts.round55_boundary import raw_pool, linked_pairs, reframe, controls, SIDES, rays46, line_at
from scripts.round46_measure import readcached
from scripts.round52_layer_scoring import rank_layers, separator_evidence
from scripts.round39_core import coverage
from scripts.round40_core import evidence_gates
from scripts.round32_common import map_curve, inverse_spec
from scripts.round53_boundary_attribution import corridor_coverage


DEST = ROOT/'runs/round55_boundary_reconstruction_v1'
DEV = ('identity','translate-3_-2','r41_scale1015','r44_scale_gray','r43_black08','r44_bright_gray','r46_bright_scale_gray')
CASES = {'image_1707.tif','im_12_arch','image_1009.tif','image_1763.tif'}
SIDE_MODES = {'U':('sup',),'D':('deep',),'UD':SIDES}
BOUNDARY = {'A':None,**{f'B{b}_{side}_{i}':(b,side,gap,window)
    for b,side,i in product((1,2,3),SIDE_MODES,(1,2))
    for gap,window in [((.01,.02)[i-1],(.12,.20)[i-1])]}}
MUSCLE = ('M0','M1','M2','M3')


def probability(r,seed,tr,meta,mask):
    with np.load(ROOT/meta['data_path']) as z:
        if 'probability' in z.files:
            return z['probability']
    suffix = Path(str(seed))/r['dataset']/Path(r['image_id']).stem/(tr+'.npz')
    for p in (DEST/'probability'/suffix,R53/'probability'/suffix):
        if p.exists():
            with np.load(p) as z:
                return z['probability']
    return None


def rank_rebuilt(base,pairs,mask,segments,roi,native,evidence_box):
    if not pairs:
        return base
    cfg,cal = settings()
    st = dict(raw_mask=mask,segments=segments,roi=roi)
    left,top,right,bottom = evidence_box
    structural = []
    for p in pairs:
        cv = p['curve']
        knots = cv['sup']['knots_x']
        ok = knots[0]>=left and knots[-1]<right and min(cv['sup']['knots_y'])>=top and max(cv['deep']['knots_y'])<bottom
        structural.append(dict(reasons=[] if ok else ['outside_image_valid_domain']))
    gates = evidence_gates(pairs,st,native,cfg,cal,structural)
    if base is not None:
        pair,gate = base
        pairs,gates = [pair,*pairs],[gate,*gates]
    chosen = rank_layers(pairs,gates,native,quality_field='single_track_quality')
    if chosen is None:
        return base
    return next(((p,g) for p,g in zip(pairs,gates) if p['pair_id']==chosen['pair_id']),None)


def compact_geometry(pair,t,roi_width,seen=None,obs=None):
    if pair is None:
        return None
    cv = pair['curve']
    out = dict(fit=map_curve(cv,inverse_spec(t)),common=coverage(cv['paired_x'],1.)/roi_width,
               strong_gaps=float(np.max(np.diff(cv['paired_x']))/roi_width))
    if seen is not None and obs is not None:
        out['sides'] = {s:dict(n=len(obs[s]['columns']),max_gap=float(np.max(np.diff(np.sort(obs[s]['columns'])))/roi_width) if len(obs[s]['columns'])>1 else None) for s in SIDES}
    return out


def muscle_groups(pair,mapped,obs,t,field,width):
    original = mapped['values']
    weights = mapped['FL_weights']
    if not weights:
        return {m:dict(values=original) for m in MUSCLE}
    segs = mapped['display']['segments'] if 'display' in mapped else [dict(c['segment'],sup=c['intersection_sup'],deep=c['intersection_deep']) for c in mapped['candidates']]
    segs = {str(z['label_id']):z for z in segs}
    if set(weights)-set(segs):
        raise ValueError('incomplete contribution archive')
    lo,_,hi,_ = field['box']
    orig_cv = mapped['curve']
    center, distances, sin_angle, far, lengths = [],[],[],[],[]
    ids = list(weights)
    for i in ids:
        seg = segs[i]
        c = seg['centroid'][0]*t['sx']+t['tx']
        center.append(not field['fallback'] and lo+(hi-lo)/3<=c<=hi-(hi-lo)/3)
        if seg['sup'] is None or seg['deep'] is None:
            raise ValueError('a contributing muscle must have two intersections')
        ds = []
        aa = []
        for s in SIDES:
            x,y = seg[s]
            fit = orig_cv[s]
            aa.append(abs(np.sin(np.deg2rad(seg['direction_deg'])-np.arctan(line_at(fit,x)[0]))))
            nx,ny = x*t['sx']+t['tx'], y*t['sy']+t['ty']
            cv = pair['curve'][s]
            evidence = root_support(dict(point=[nx,ny],piece=0),obs[s],width,cv)
            ds.append(evidence['distance'] if evidence['distance'] is not None else 1.)
        distances.append(max(ds))
        far.append(max(ds)>.05)
        sin_angle.append(min(aa))
        lengths.append(float(np.linalg.norm(np.subtract(seg['sup'],seg['deep']))))
    old = np.asarray(list(weights.values()))
    lengths = np.asarray(lengths)
    scale = original['FL']/float(old@lengths)
    center = np.asarray(center,bool)
    dist = np.asarray(distances)
    # Existing weights already include perturbed geometry risk. Penalize far/near-parallel roots explicitly.
    angle_factor = np.minimum(1.,np.asarray(sin_angle)/np.sin(np.deg2rad(12)))
    factor = angle_factor/(1+dist/.05)
    close = np.floor(dist/.02+1e-10)==np.floor(dist.min()/.02+1e-10)
    policies = (old,old*np.where(center,1.5,1.),old*factor,old*factor*np.where(center&close,1.5,1.))
    out = {}
    for label,w in zip(MUSCLE,policies):
        w = w/w.sum() if w.sum()>0 else old
        values = {**original,'FL':original['FL'] if label=='M0' else float(w@lengths)*scale}
        out[label] = dict(values=values,center=float(w@center),far=float(w@np.asarray(far)),effective=float(1/(w@w)),field_fallback=field['fallback'])
    return out


def side_similarity(a,b,span):
    if a is None or b is None:
        return {s:None for s in SIDES}
    lo,hi = max(a['support'][0],b['support'][0]),min(a['support'][1],b['support'][1])
    if hi-lo<.06*span:
        return {s:None for s in SIDES}
    x = np.linspace(lo,hi,25)
    yy = [[np.interp(x,c[s]['knots_x'],c[s]['knots_y']) for s in SIDES] for c in (a,b)]
    tol = max(4*span/1046,.10*np.median(yy[0][1]-yy[0][0]))
    return {s:bool(np.mean(abs(u-v)<=tol)>=.8) for s,u,v in zip(SIDES,*yy)}


def task(args):
    r,seed,phase,groups,selected,protocol_id = args
    destination = DEST/'replay'/phase/str(seed)/r['dataset']/(Path(r['image_id']).stem+'.json.gz')
    if destination.exists():
        z = gzread(destination)
        assert z['protocol']==protocol_id and z['groups']==list(groups) and z['selected']==selected
        return destination
    old = gzread(source_path(r,seed))
    frozen = gzread(source(r,seed))
    cfg,_ = settings()
    variants = {}
    diagnostics = {}
    stage = TRANSFORMS if phase=='full' else DEV
    all_cases = r['image_id'] in CASES
    identity_curves = {}
    for tr in stage:
        v = old['variants'][tr]
        t,roi,fm = v['transform'],v['measurement_roi'],v['region']
        width = fm['reference_width']
        native = dict(width=t['width'],height=t['height'],px_per_mm=None)
        mask,meta = readcached(r,tr,'boundary_'+str(seed),'I')
        assert old['sources'][tr]['boundary_I']==meta['sha256']
        p = probability(r,seed,tr,meta,mask)
        x,y,u,d = fm['evidence_box']
        bounded = np.zeros_like(mask)
        bounded[:,y:d,x:u] = mask[:,y:d,x:u]
        if p is not None:
            p = p.copy()
            p[:,:y] = 0;p[:,d:] = 0;p[:,:,:x] = 0;p[:,:,u:] = 0
        pool,maps,factor,offset = raw_pool(bounded,roi,width)
        assert len(pool)==v['boundary_audit']['ID']['candidate_count']
        variants[tr] = {}
        cached_links = {}
        rays = {}
        def baseline_ray(i):
            if i is None:
                return None
            if i not in rays:
                rays[i] = rays46([pool[i]],roi,width)[0]
            return rays[i]

        for kind in KINDS:
            old_arms = frozen['variants'][tr][kind]
            legacy = v['arms'][kind+'|ID']
            frozen_a = old_arms['A']
            frozen_q = old_arms['Qlocal']
            a = baseline_ray(frozen_a['selected_pair_id'])
            q = baseline_ray(frozen_q['selected_pair_id']) if all_cases else None
            a_gate = next((z for z in legacy['gates'] if a is not None and z['pair_id']==a['pair_id']),None)
            if a is not None:
                a = dict(a)
                rows0 = [((c-offset[0])*factor,(up-offset[1])*factor,(dn-offset[1])*factor) for c,up,dn in a['observations']]
                z = separator_evidence(rows0,maps[0] if a['source_decoder']=='D0' else maps[1]+maps[2],factor,coverage)
                a['separator'] = {'single_track_quality':(z['shared']-z['strongest'])/1046}
            @cache
            def fascicles():
                return load_fascicles(r,tr,kind,old,roi,cfg)

            by_curve = {}
            def measure_pair(pair):
                if pair is None:
                    return dict(values=dict(MT=None,PA=None,FL=None),curve=None,FL_weights={})
                i = pair['pair_id']
                if a is not None and pair['curve']==a['curve'] and i==legacy['selected_pair_id']:
                    check_equal(legacy['values'],frozen_a['values'],'Round55 frozen parity')
                    return legacy
                token = (i,tuple(tuple(pair['curve'][s][f]) for s in SIDES for f in ('knots_y','left_ray','right_ray')))
                if token not in by_curve:
                    _,raw,kept = fascicles()
                    by_curve[token] = native_measure(pair,kept,len(raw),roi,t,old)[1]
                return by_curve[token]

            choices = {'A':a}
            results = {}
            for g in groups:
                if g=='A' or g=='Q':
                    continue
                method,side,gap,window = BOUNDARY[g]
                pair = a
                if method in (1,3):
                    cache_key = side,gap
                    if cache_key not in cached_links:
                        ps,ns = linked_pairs(pool,maps,factor,offset,bounded,p,width,roi,SIDE_MODES[side],gap)
                        cached_links[cache_key] = (rays46(ps,roi,width),ns)
                    new,_ = cached_links[cache_key]
                    # Preserve the baseline's ranking score on an existing pair.
                    if new:
                        _,_,segs = fascicles()
                        base_gate = a_gate and dict(a_gate,soft_factor=a_gate['soft_score']/a['old_quality'] if a['old_quality'] else 1.)
                        pair_with_gate = rank_rebuilt((a,base_gate) if a is not None else None,new,fascicles()[0],segs,roi,native,fm['evidence_box'])
                        pair = pair_with_gate[0] if pair_with_gate else None
                if method in (2,3) and pair is not None:
                    pair = reframe(pair,width,window,SIDE_MODES[side],native,fm['evidence_box'])
                choices[g] = pair
            for g in groups:
                pair = choices.get(g)
                z = frozen_q if g=='Q' else frozen_a if pair is a else measure_pair(pair)
                item = dict(values=z['values'],changed=g not in ('A','Q') and pair is not a)
                if pair is not None:
                    cv = map_curve(pair['curve'],inverse_spec(t))
                    if tr=='identity':
                        identity_curves[kind,g] = cv
                    item['same_layer'] = side_similarity(identity_curves.get((kind,g)),cv,frozen['roi_width'])
                    if tr=='identity' and r['dataset']=='apo153' or all_cases:
                        item['curve'] = cv
                    item['joint'] = coverage(cv['paired_x'],1.)/frozen['roi_width']
                results[g] = item
            if phase in ('muscle','full'):
                pair = choices[selected]
                if pair is not None:
                    mapped = measure_pair(pair)
                    check_equal(mapped['values'],results[selected]['values'],'chosen boundary replay')
                    results.update(muscle_groups(pair,mapped,owned_columns(pair,maps,factor,offset),t,v['evaluation_roi'],width))
                else:
                    results.update({m:dict(values=results[selected]['values']) for m in MUSCLE})
            if all_cases and tr in ('identity','translate-3_-2','r41_scale1015','r44_scale_gray'):
                note = dict(probability=p is not None,baseline_pair=frozen_a['selected_pair_id'],qlocal_pair=frozen_q['selected_pair_id'],
                            linked={str(k):v[1] for k,v in cached_links.items()},
                            selected_pairs={g:x['pair_id'] if x is not None else None for g,x in choices.items()})
                if tr=='identity' and a is not None:
                    identity_curves[kind,'audit'] = map_curve(a['curve'],inverse_spec(t))
                guide = identity_curves.get((kind,'audit'))
                if guide is not None:
                    note['response_aligned_to_original'] = corridor_coverage(map_curve(guide,t),width,mask,bounded,p,fm['evidence_box'],meta['input_box'])
                if q is not None:
                    note['qlocal_geometry'] = compact_geometry(q,t,width,True,owned_columns(q,maps,factor,offset))
                    note['qlocal_terminal_control'] = {side:measure_pair(reframe(q,width,.20,SIDE_MODES[side],native,fm['evidence_box']))['values'] for side in SIDE_MODES}
                diagnostics[tr,kind] = note
            variants[tr][kind] = results
    gzwrite(destination,dict(key=key(r),seed=seed,protocol=protocol_id,groups=groups,selected=selected,
        roi_width=frozen['roi_width'],scale=frozen['scale'],variants=variants,
        diagnostics={f'{t}|{k}':v for (t,k),v in diagnostics.items()}))
    return destination


def protocol():
    p = DEST/'protocol.json'
    files = (Path(__file__),ROOT/'scripts/round55_boundary.py',ROOT/'scripts/round55_evaluate.py',
             ROOT/'scripts/round55_probability.py',ROOT/'thinking/ROUND55_BOUNDARY_RECONSTRUCTION_PLAN_20261105.txt',
             R53/'experiment_lock.json',ROOT/'runs/round54_short_boundary_fl_v1'/'summary.json')
    q = dict(sources={f.relative_to(ROOT).as_posix():sha(f) for f in files},controls=controls(),
             groups=list(BOUNDARY),dev=list(DEV),full=list(TRANSFORMS),seeds=list(SEEDS))
    if p.exists():
        assert j(p)==q
    else:
        write(p,q)
    return p


def run(phase,workers,limit=None):
    from scripts.round55_evaluate import evaluate, lock_boundaries, lock_muscles
    p = protocol()
    if phase=='controls':
        print(j(p)['controls'],flush=True)
        return
    selected = None
    if phase in ('boundary','pilot'):
        groups = ('Q',*BOUNDARY)
    else:
        lock = j(DEST/'boundary_lock.json')
        assert lock['protocol']==sha(p)
        selected = lock['selected']
        groups = tuple(dict.fromkeys(('A','Q',*lock['family_best'].values(),selected)))
        if phase=='full':
            assert j(DEST/'muscle_lock.json')['boundary_lock']==sha(DEST/'boundary_lock.json')
    rr = rows() if phase=='full' else rows('development')
    if phase=='pilot':
        rr = [r for r in rows() if r['image_id'] in CASES]
    if limit:
        rr = rr[:limit]
    start = monotonic()
    args = [(r,s,phase,groups,selected,sha(p)) for r,s in product(rr,SEEDS)]
    paths = []
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
        for f in as_completed(pool.submit(task,a) for a in args):
            paths.append(f.result())
            if len(paths)%30==0:
                print('Round55',phase,len(paths),len(args),round(monotonic()-start),flush=True)
    if limit or phase=='pilot':
        print('Round55 pilot',len(paths),round(monotonic()-start),flush=True)
        return
    result = evaluate(paths,phase,groups,selected)
    if phase=='boundary':
        lock_boundaries(result,sha(p))
    if phase=='muscle':
        lock_muscles(result)
    print('Round55 complete',phase,len(paths),round(monotonic()-start),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('controls','pilot','boundary','muscle','full'))
    parser.add_argument('--workers',type=int,default=14)
    parser.add_argument('--limit',type=int)
    opt = parser.parse_args()
    run(opt.phase,opt.workers,opt.limit)
