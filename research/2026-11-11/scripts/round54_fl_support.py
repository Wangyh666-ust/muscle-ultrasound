"""Read-only side-specific FL support and risk audit for an experimental replay."""
from scripts.round53_experiment import np, readcached, settings, rays46
from scripts.round53_boundary_attribution import TRACED, core
from scripts.round38_core import base_measure, qualified, aggregate
from scripts.round39_core import fl_inputs, combine_fl, modify_fl, MCONF
from scripts.round40_core import apply_u
from scripts.round41_measure import finish
from scripts.round29_core import extended_fl
from scripts.round36_core import independent_segments


def reconstruct(mask, roi, width):
    cols = [list(enumerate(core.run_centres_per_column(m))) for m in mask]
    pool, _, maps, factor, offset = TRACED(cols,roi,reference_width=width)
    return rays46(pool,roi,width), maps, factor, offset


def boundary_mask(r, seed, tr, old):
    masks, meta = readcached(r,tr,'boundary_'+str(seed),'I')
    assert meta['sha256']==old['sources'][tr]['boundary_I']
    x,y,z,t = old['variants'][tr]['region']['evidence_box']
    bounded = np.zeros_like(masks)
    bounded[:,y:t,x:z] = masks[:,y:t,x:z]
    return bounded


def owned_columns(pair, maps, factor, offset):
    """Only extend evidence with one unambiguously associated observed track per side."""
    x0,y0 = offset
    points = np.asarray(pair['observations'])
    reference_x = np.rint(points[:,0]).astype(int)
    result = {}
    for i,side in enumerate(('sup','deep'),1):
        allowed = maps[0] if pair['source_decoder']=='D0' else maps[i]
        proposals = {}
        for track in allowed:
            xy = np.asarray(list(track.items()))/factor+[x0,y0]
            index = {round(float(x)):float(y) for x,y in xy}
            hit = sum(abs(index.get(int(x),np.inf)-y)<=.51 for x,y in zip(reference_x,points[:,i]))
            if hit>=.80*len(points):
                proposals[tuple(index.items())] = hit
        ranked = sorted(proposals,key=lambda p:-proposals[p])
        unique = bool(ranked and (len(ranked)==1 or proposals[ranked[0]]-proposals[ranked[1]]>=max(3,.05*len(points))))
        xy = np.asarray(ranked[0]) if unique else points[:,[0,i]]
        result[side] = dict(columns=xy[:,0], ordinates=xy[:,1],
                            association='single_track' if unique else 'paired_observations_only')
    return result


def root_support(root, obs, width, fit):
    x = root['point'][0]
    xs = obs['columns']
    knots = fit['knots_x']
    expected = np.interp(xs,knots,fit['knots_y'])
    for outside,ray_name in ((xs<knots[0],'left_ray'),(xs>knots[-1],'right_ray')):
        expected[outside] = np.polyval(fit[ray_name],xs[outside])
    # The observed track must also follow this particular curve near the root.
    # A track that bends away cannot support its extrapolated intersection.
    xs = xs[abs(obs['ordinates']-expected)<=max(1,4*width/1046)]
    if not len(xs):
        return dict(distance=None,zone='unknown',ray=str(root['piece']).endswith('ray'))
    d = float(np.min(abs(xs-x)))
    ray = str(root['piece']).endswith('ray')
    # A column cell, not a fitted interval, supplies direct observed support.
    zone = ('observed' if d<=.5 else 'gap' if xs.min()<=x<=xs.max() else 'beyond_observation')
    return dict(distance=d/width,zone=zone,ray=ray,
                near_columns=int(np.sum(abs(xs-x)<=max(1,4*width/1046))))


def track_record(seg, cv, obs, width, native, risk, weight, sid, independent):
    f = extended_fl(seg,cv,native['width'],native['height'])
    roots = f.get('roots',{})
    assert all(len(roots[s]['roots'])==1 for s in ('sup','deep'))
    ts = [roots[s]['roots'][0]['t'] for s in ('sup','deep')]
    a = np.radians(seg['direction_deg'])
    extent = sorted((np.asarray([seg['p0'],seg['p1']])-seg['centroid'])@np.array([np.cos(a),np.sin(a)]))
    total = f['fl_px_raw']
    low, high = sorted(ts)
    end = [float(np.clip((extent[0]-low)/total,0,1)),float(np.clip((high-extent[1])/total,0,1))]
    if ts[0]>ts[1]:
        end.reverse()
    rr = {}
    for s,e in zip(('sup','deep'),end):
        root = roots[s]['roots'][0]
        rr[s] = dict(**root_support(root,obs[s],width,cv[s]),along=e,point=root['point'],association=obs[s]['association'])
    return dict(id=sid,weight=weight,endpoints=rr,extrap=f['extrapolation_ratio'],independent=independent,
                hard_ok=not any(x in ('intersection_outside_image','inverted_boundary_continuation') for x in f['fl_failure_conditions']),
                risk=dict(u=risk['u'],lost=risk['lost'],root_switches=risk['root_switches'],reliable=risk['reliable']))


def native_measure(pair, segments, n, roi, transform, row):
    cfg,cal = settings()
    cv = pair['curve']
    dims = dict(width=transform['width'],height=transform['height'],px_per_mm=None)
    st = dict(segments=segments,raw_count=n,roi=roi,evidence=np.zeros((1,1),bool))
    base,cs = base_measure([dict(pair_id=pair['pair_id'],curve=cv,old_quality=1.,quality=1.)],st,dims)
    q = modify_fl(aggregate(qualified(base,cs),cs,dims,cfg),cfg,cal,True,True)
    q['_native_dimensions'] = dims
    out = apply_u(q,cfg,cal)
    return out,finish(out,transform,row)


def audit(pair, maps, factor, offset, width, native):
    selected = {int(s) for s in native['FL_weights']}
    candidates = {c['segment']['label_id']:c for c in native['candidates']}
    if not selected:
        return dict(tracks=[], rejected={str(i):list(c['invalid_conditions']) for i,c in candidates.items()})
    obs = owned_columns(pair,maps,factor,offset)
    segs = [candidates[i]['segment'] for i in sorted(selected)]
    _,groups = independent_segments(segs,native['_native_dimensions']['height'],MCONF['c1'])
    assignment = {i:k for k,g in enumerate(groups) for i in g}
    tracks = [track_record(candidates[i]['segment'],pair['curve'],obs,width,native['_native_dimensions'],
                           native['boundary_sensitivity']['tracks'][str(i)],native['FL_weights'][str(i)],str(i),assignment[i])
              for i in sorted(selected)]
    assert abs(sum(t['weight'] for t in tracks)-1)<1e-8
    return dict(tracks=tracks,rejected={str(i):list(c['invalid_conditions']) for i,c in candidates.items() if i not in selected})


def eligible(tr, distance, risk):
    return tr['hard_ok'] and all(e['distance'] is not None and e['distance']<=distance
        for e in tr['endpoints'].values()) and (not risk or (tr['risk']['reliable'] and tr['risk']['root_switches']==0))


def qualifying(tracks, distance, mass, risk):
    passed = [t for t in tracks if eligible(t,distance,risk)]
    return (passed if len({t['independent'] for t in passed})>=2 and
            sum(t['weight'] for t in passed)>=mass-1e-12 else [])


def subset_fl(native, mapped, tracks):
    if not tracks:
        return None
    if len(tracks)==len(native['FL_weights']):
        return mapped['values']['FL']
    cfg,cal = settings()
    ids,vals,quality,kernel,_,span = fl_inputs(native,cfg,True)
    accepted = {int(t['id']) for t in tracks}
    indexes = [j for j,i in enumerate(ids) if i in accepted]
    picked = [ids[j] for j in indexes]
    diag = native['boundary_sensitivity']['tracks']
    q = quality[indexes]*np.asarray([diag[str(i)]['risk_factor'] for i in picked])
    w,_ = combine_fl(vals[indexes],q,kernel[indexes],[],span,cfg['aggregate'],cal,True,True)
    assert np.isclose(w.sum(),1.)
    orig_lengths = {c['segment']['label_id']:c['raw_FL'] for c in mapped['candidates']}
    unscaled = sum(v*orig_lengths[int(i)] for i,v in native['FL_weights'].items())
    conversion = mapped['values']['FL']/unscaled
    return float(sum(v*orig_lengths[i] for i,v in zip(picked,w))*conversion)


def controls():
    cv = dict(support=[10,90],paired_x=list(range(10,91)),max_sample_gap=2.,
        sup=dict(knots_x=[10,90],knots_y=[30.,30.],left_ray=[0,30],right_ray=[0,30]),
        deep=dict(knots_x=[10,90],knots_y=[70.,70.],left_ray=[0,70],right_ray=[0,70]))
    seg = dict(centroid=[50,50],p0=[45,45],p1=[55,55],direction_deg=45.)
    good = dict(u=.01,lost=0,root_switches=0,reliable=True)
    out = {}
    for name,xs,xd,risk in (
        ('known_intersections',range(10,91),range(10,91),good),
        ('upper_short',range(36,91),range(10,91),good),
        ('lower_short',range(10,91),range(10,64),good),
        ('internal_gap',list(range(10,23))+list(range(38,91)),range(10,91),good),
        ('staggered_sides',range(10,50),range(51,91),good),
        ('small_motion_changes_root',range(10,91),range(10,91),{**good,'root_switches':1}),
    ):
        obs = {s:dict(columns=np.array(x),ordinates=np.full(len(x),y),association='single_track')
               for s,x,y in zip(('sup','deep'),(xs,xd),(30.,70.))}
        tr = track_record(seg,cv,obs,100,dict(width=100,height=100),risk,1.,'1',0)
        assert np.allclose(tr['endpoints']['sup']['point'],[30,30]) and np.allclose(tr['endpoints']['deep']['point'],[70,70])
        assert np.isclose(tr['extrap'],.75)
        out[name] = [eligible(tr,.05,False),eligible(tr,.05,True)]
    assert out=={'known_intersections':[True,True],'upper_short':[False,False],
                 'lower_short':[False,False],'internal_gap':[False,False],
                 'staggered_sides':[True,True],'small_motion_changes_root':[True,False]}
    shifted = dict(columns=np.arange(10,91),ordinates=np.full(81,36.))
    assert root_support(dict(point=[30,30],piece=0),shifted,100,cv['sup'])['distance'] is None
    out['nearby_parallel_echo'] = [False,False]
    return out
