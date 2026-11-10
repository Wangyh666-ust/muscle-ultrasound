"""Independent eligibility and ranking interventions against Round52 conservative."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
import time

from scripts.round52_layer_replay import (
    ROOT, R40, R41, CFG, KINDS, SEEDS, SPLIT, NATURAL, Path, check_equal,
    gzread, gzwrite, initialize, j, key, measure, np, readcached, rows, sha,
    source_path, write, load_fascicles,
)
from scripts.round46_common import SCOPES
from scripts.round46_decoder import rays46
from scripts.round39_core import coverage
from scripts.round40_core import evidence_gates
from scripts.round36_core import choose, support
from scripts.round47_core import identity_evidence
from scripts.round52_layer_scoring import rank_layers, scored_pool


DEST = ROOT/'runs/round53_boundary_stage_audit_v1'
TRANSFORMS = ('identity', *SCOPES['all26'])
GROUPS = ('A','Q25','Q20','Qlocal','R')
QMODES = GROUPS[1:4]
METRICS = ('MT','PA','FL')
REGION_WEIGHTS = np.array([.25,.50,.25])


@cache
def settings():
    return j(CFG),j(R40/'calibration.json')


def initialize_worker():
    initialize()
    from scripts.round46_column_fast import install
    install()


def source_columns(masks, width, curve):
    """Role hits observed near the fitted side, never count interpolated pixels."""
    _,h,w = masks.shape
    rad = max(1,round(3*width/1046))
    span = curve['support']
    xs = np.arange(max(0,int(np.ceil(span[0]-.10*width))),min(w,int(span[1]+.10*width)+1))
    output = []
    for role,side in ((1,'sup'),(2,'deep')):
        fit = curve[side]
        knots = fit['knots_x']
        yy = np.interp(xs,knots,fit['knots_y'])
        for where,name in ((xs<knots[0],'left_ray'),(xs>knots[-1],'right_ray')):
            yy[where] = np.polyval(fit[name],xs[where])
        yi = np.rint(yy).astype(int)
        neighbors = np.clip(yi[:,None]+np.arange(-rad,rad+1),0,h-1)
        valid = (yi>=0)&(yi<h)
        hit = (masks[0,neighbors,xs[:,None]] | masks[role,neighbors,xs[:,None]]).any(1)&valid
        output.append(float(hit.sum()/width))
    return min(output)


def c1_position(curve, segments, roi):
    """Vote with actual observed segment locations, without extending fascicles."""
    l,_,r,_ = roi['box']
    totals = np.zeros(3)
    inside = np.zeros(3)
    inside_segments = 0
    for s in segments:
        x,y = np.linspace(s['p0'],s['p1'],9).T
        observed = support(curve,x)&(x>=l)&(x<r)
        if observed.sum() < 3:
            continue
        reg = np.clip(((x[observed]-l)/(r-l)*3).astype(int),0,2)
        fit = [np.interp(x[observed],curve[k]['knots_x'],curve[k]['knots_y']) for k in ('sup','deep')]
        within = (y[observed]>=fit[0])&(y[observed]<=fit[1])
        w = min(s['length_px']/(r-l),.10)/len(x)
        totals += np.bincount(reg,minlength=3)*w
        inside += np.bincount(reg[within],minlength=3)*w
        inside_segments += within.sum() >= 3
    # Regularize unobserved regions to neutral evidence, not a contradiction.
    regional = (inside+.025)/(totals+.05)
    return float(REGION_WEIGHTS@regional),inside_segments


def choose_candidates(pool,gates,native,mask,width,roi,get_fascicles,groups,q_combination):
    a = rank_layers(pool,gates,native,quality_field='single_track_quality')
    selections = {group:a for group in groups}
    if not pool:
        return selections
    by_id = {g['pair_id']:g for g in gates}
    qualified = {g['pair_id'] for g in gates if g['structural_allowed']}
    weak = {g['pair_id'] for g in gates if g['structural_reasons'] == ['insufficient_observed_support']}
    if not weak and len(qualified)<2:
        return selections
    sides, locations = {},{}
    requested_q = set(groups)&set(QMODES)
    if 'QR' in groups:
        requested_q.add(q_combination)
    covers = {p['pair_id']:coverage(p['curve']['paired_x'],1.)/width for p in pool} if requested_q and weak else {}
    def location(p):
        i = p['pair_id']
        if i not in locations:
            locations[i] = c1_position(p['curve'],get_fascicles()[2],roi)
        return locations[i]

    def allowed_q(group,p):
        i = p['pair_id']
        if i not in weak:
            return False
        q = covers[i]
        if group != 'Qlocal':
            return q >= (.25 if group=='Q25' else .20)
        if q < .15:
            return False
        cv = p['curve']
        lo,hi = cv['support']
        if support(cv,np.linspace(lo,hi,5)[1:-1]).sum() < 2:
            return False
        if i not in sides:
            sides[i] = source_columns(mask,width,cv)
        return sides[i]>=.28 and location(p)[1]>=2

    role = None
    for group in groups:
        if group == 'A':
            continue
        ranking = group in ('R','QR')
        q_group = q_combination if group=='QR' else group
        ids = qualified.copy()
        if q_group in QMODES:
            ids.update(p['pair_id'] for p in pool if allowed_q(q_group,p))
        if not ids or (ids==qualified and (not ranking or len(ids)<2)):
            continue
        if ranking and role is None:
            role = identity_evidence(pool,mask,width)
        scores = []
        for p in pool:
            i = p['pair_id']
            if i not in ids:
                continue
            base = p['separator']['single_track_quality']*by_id[i]['soft_factor']
            if ranking:
                # Both head roles must support the fit; C1 votes on its actual observed fragments.
                role_score = min(s['role'] for s in role[i]['sides'])
                base *= (.65+.35*role_score)*(.75+.25*location(p)[0])
            scores.append(dict(pair_id=i,old_score=base))
        selections[group] = choose(pool,scores,{**native,'height':pool[0]['ranking_height']})
    return selections


def task(r,seed,transforms,groups,q_combination,lock_id):
    phase = 'development' if lock_id is None else 'locked'
    dest = DEST/'measurements'/phase/str(seed)/r['dataset']/(Path(r['image_id']).stem+'.json.gz')
    if dest.exists():
        x = gzread(dest)
        assert x['lock']==lock_id and tuple(x['variants'])==transforms and all(set(v)==set(groups) for q in x['variants'].values() for v in q.values())
        return dest
    cfg,cal = settings()
    old = gzread(source_path(r,seed))
    variants = {}
    for tr in transforms:
        v = old['variants'][tr]
        t,roi,fm = v['transform'],v['measurement_roi'],v['region']
        w = fm['reference_width']
        native = dict(width=t['width'],height=t['height'])
        masks,meta = readcached(r,tr,'boundary_'+str(seed),'I')
        assert old['sources'][tr]['boundary_I']==meta['sha256']
        left,top,right,bottom = fm['evidence_box']
        bounded = np.zeros_like(masks)
        bounded[:,top:bottom,left:right] = masks[:,top:bottom,left:right]
        pool = scored_pool(bounded,roi,w)
        assert len(pool)==v['boundary_audit']['ID']['candidate_count']
        variants[tr] = {}
        rays = None
        for kind in KINDS:
            base = v['arms'][kind+'|ID']
            gates = [dict(g) for g in base['gates']]
            assert [p['pair_id'] for p in pool]==[g['pair_id'] for g in gates]
            def get_fascicles():
                return load_fascicles(r,tr,kind,old,roi,cfg)

            get_fascicles = cache(get_fascicles)
            if any(p['old_quality']==0 for p in pool):
                mask,_,kept = get_fascicles()
                recomputed = evidence_gates(pool,dict(raw_mask=mask,roi=roi,segments=kept),native,cfg,cal,
                                            [{'reasons':g['structural_reasons']} for g in gates])
                for g,q in zip(gates,recomputed):
                    check_equal(g['soft_score'],q['soft_score'],'frozen C1 score')
                    g['soft_factor'] = q['soft_factor']
            else:
                for p,g in zip(pool,gates):
                    g['soft_factor'] = g['soft_score']/p['old_quality']
                    assert 0 < g['soft_factor'] <= 1+1e-8
            legacy = rank_layers(pool,gates,native)
            assert (legacy['pair_id'] if legacy else None)==base['selected_pair_id']
            selected = choose_candidates(pool,gates,native,bounded,w,roi,get_fascicles,groups,q_combination)
            outcomes = {base['selected_pair_id']:dict(values=base['values'],selected_pair_id=base['selected_pair_id'],curve=base['curve'])}
            result = {}
            for group,p in selected.items():
                i = p['pair_id'] if p else None
                if i not in outcomes:
                    if rays is None:
                        rays = {z['pair_id']:z['curve'] for z in rays46(pool,roi,w)}
                    _,raw,kept = get_fascicles()
                    out = measure(rays[i],i,kept,len(raw),roi,t,old,cfg,cal)
                    outcomes[i] = {k:out[k] for k in ('values','selected_pair_id','curve')}
                out = outcomes[i]
                result[group] = {k:out[k] for k in ('values','selected_pair_id')}
                if tr=='identity' and r['dataset']=='apo153':
                    result[group]['curve'] = out['curve']
            variants[tr][kind] = result
    gzwrite(dest,dict(lock=lock_id,key=key(r),seed=seed,variants=variants,roi_width=old['variants']['identity']['region']['reference_width'],scale=old['px_per_mm']))
    return dest


def controls():
    def candidate(i,q,score=.4):
        n = int(q*100)
        curve = dict(support=[0.,n-1.],paired_x=list(range(n)),max_sample_gap=2.,
                     sup=dict(knots_x=[0,n-1],knots_y=[25.,25.],left_ray=[0,25],right_ray=[0,25]),
                     deep=dict(knots_x=[0,n-1],knots_y=[55.,55.],left_ray=[0,55],right_ray=[0,55]))
        return dict(pair_id=i,curve=curve,old_quality=score,ranking_height=100,separator=dict(single_track_quality=score))

    def gate(i,reasons=(),score=.4):
        return dict(pair_id=i,structural_allowed=not reasons,structural_reasons=list(reasons),soft_score=score,soft_factor=1.)

    mask = np.zeros((3,100,100),bool)
    mask[1,25,:] = True
    mask[2,55,:] = True
    observed = [{'p0':[0,31],'p1':[22,43],'length_px':24}, {'p0':[1,35],'p1':[23,45],'length_px':24}]
    current = candidate(0,.40,.3)
    cases = {}
    for name,n,m,segments,reason in (
        ('well_supported',.27,mask,observed,'insufficient_observed_support'),
        ('short_but_observed',.22,mask,observed,'insufficient_observed_support'),
        ('one_side_missing',.22,mask & np.array([True,True,False])[:,None,None],observed,'insufficient_observed_support'),
        ('no_muscle_evidence',.22,mask,[],'insufficient_observed_support'),
        ('outside_domain',.22,mask,observed,'outside_image_valid_domain'),
    ):
        chosen = choose_candidates([current,candidate(1,n)],[gate(0,score=.3),gate(1,(reason,))],dict(height=100),m,100,
                                   dict(box=[0,0,100,100]),lambda:(None,None,segments),GROUPS,None)
        cases[name] = {g:p['pair_id'] for g,p in chosen.items()}
    assert cases['well_supported']==dict(A=0,Q25=1,Q20=1,Qlocal=1,R=0)
    assert cases['short_but_observed']==dict(A=0,Q25=0,Q20=1,Qlocal=1,R=0)
    assert cases['one_side_missing']['Qlocal']==cases['no_muscle_evidence']['Qlocal']==0
    assert all(i==0 for i in cases['outside_domain'].values())
    return cases


def protocol():
    p = DEST/'experiment_protocol.json'
    files = (Path(__file__),CFG,R40/'calibration.json',R41/'partitions.json',NATURAL,
             ROOT/'scripts/round52_layer_scoring.py',ROOT/'scripts/round53_experiment_evaluate.py')
    record = dict(sources={f.relative_to(ROOT).as_posix():sha(f) for f in files},groups=list(GROUPS),
                  transforms=list(TRANSFORMS),controls=controls(),training=False,default_changed=False,
                  primary='Development gate: fewer severe FL conditions pooled, no increase in either boundary seed, '
                          'no new FL missing condition and no net loss on annotated originals. Eligibility tie: smallest intervention. '
                          'Confirmation reports both boundary seeds and both C1s independently; no tuning.',
                  diagnosis='The 102 residual events and the 38 matched events are not used in the development gate.')
    if p.exists():
        assert j(p)==record
    else:
        write(p,record)
    return p


def run(phase,workers):
    p = protocol()
    if phase=='control':
        return
    from scripts.round53_experiment_evaluate import evaluate, lock_development
    lock_path = DEST/'experiment_lock.json'
    if phase=='development':
        groups,q_combination,lock_id = GROUPS,None,None
        rr = rows('development')
    else:
        lock = j(lock_path)
        assert lock['protocol_sha256']==sha(p)
        groups,q_combination,lock_id = tuple(lock['full_groups']),lock['qualification'],sha(lock_path)
        rr = rows('confirmation')
    paths = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
        fs = [pool.submit(task,r,s,TRANSFORMS,groups,q_combination,lock_id) for r in rr for s in SEEDS]
        for f in as_completed(fs):
            paths.append(f.result())
            if len(paths)%20==0:
                print('R53',phase,len(paths),len(fs),round(time.time()-start),flush=True)
    if phase=='development':
        digest = evaluate(paths,groups,'development')
        lock_development(digest,p,lock_path)
    else:
        # Development records supply the same locked arms, with a locked rerun only when combined.
        if 'QR' in groups:
            dev_groups = ('A','QR')
            with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
                fs = [pool.submit(task,r,s,TRANSFORMS,dev_groups,q_combination,lock_id) for r in rows('development') for s in SEEDS]
                for f in as_completed(fs):
                    paths.append(f.result())
        else:
            paths += sorted((DEST/'measurements/development').glob('*/*/*.json.gz'))
        evaluate(paths,groups,'full')
    print('R53',phase,'complete',len(paths),round(time.time()-start),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('control','development','full'))
    parser.add_argument('--workers',type=int,default=14)
    opts = parser.parse_args()
    run(opts.phase,opts.workers)
