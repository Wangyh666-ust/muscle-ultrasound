"""Experimental same-side linking and endpoint fitting on frozen predictions."""
from copy import deepcopy
from itertools import product

from scripts.round54_fl_replay import np
from scripts.round36_core import support
from scripts.round39_core import coverage, fit_float
from scripts.round29_core import line_at
from scripts.round32_common import map_curve
from scripts.round52_layer_scoring import separator_evidence
from scripts.round53_boundary_attribution import TRACED, core
from scripts.round54_fl_support import owned_columns
from scripts.round46_decoder import rays46


SIDES = ('sup','deep')


def raw_pool(mask, roi, width):
    columns = [list(enumerate(core.run_centres_per_column(a))) for a in mask]
    pool, _, maps, factor, offset = TRACED(columns,roi,reference_width=width)
    return pool,maps,factor,offset


def link_options(tracks, evidence, factor, offset, max_gap):
    """Associate one unambiguous pair of anchored fragments through weak evidence."""
    if len(tracks)<2:
        return []
    x0,y0 = offset
    prepared = []
    for t in tracks:
        xy = np.asarray(list(t.items()))
        if len(xy)<8:
            prepared.append(None)
            continue
        a,b = [np.polyfit(z[:,0],z[:,1],1) for z in (xy[:8],xy[-8:])]
        prepared.append((xy,a,b))
    choices = []
    for i,one in enumerate(prepared):
        if one is None:
            continue
        a,_,left_line = one
        x,y = a[-1]
        for j,two in enumerate(prepared):
            if two is None or i==j:
                continue
            b,right_line,_ = two
            u,v = b[0]
            gap = u-x
            if gap<=1.01*factor or gap>max_gap*1046:
                continue
            slope = (v-y)/gap
            angular_error = max(abs(np.arctan(slope)-np.arctan(q[0])) for q in (left_line,right_line))
            if angular_error>np.deg2rad(14) or abs(v-np.polyval(left_line,u))>8 or abs(y-np.polyval(right_line,x))>8:
                continue
            xx = np.arange(max(0,round(x/factor+x0)+1),min(evidence.shape[1],round(u/factor+x0)))
            if not len(xx):
                continue
            yy = np.rint((y+slope*((xx-x0)*factor-x))/factor+y0).astype(int)
            rad = max(1,round(2/factor))
            rows = np.clip(yy[:,None]+np.arange(-rad,rad+1),0,evidence.shape[0]-1)
            hit = evidence[rows,xx[:,None]].any(1)&(yy>=0)&(yy<evidence.shape[0])
            if hit.mean()>=.65:
                choices.append((i,j,int(hit.sum()),len(hit)))
    # Competing continuations indicate ambiguous identity. Keep the original fragments.
    unique = [(i,j,h,n) for i,j,h,n in choices if sum(a==i for a,_,_,_ in choices)==1 and sum(b==j for _,b,_,_ in choices)==1]
    return [({**tracks[i],**tracks[j]},h,n) for i,j,h,n in unique]


def linked_pairs(pool, maps, factor, offset, mask, prob, width, roi, sides, gap):
    options = {}
    for i,s in enumerate(SIDES,1):
        if s not in sides:
            continue
        # The role-specific head supplies both the anchor tracks and the corridor.
        evidence = (prob[i]>.35) if prob is not None else mask[i]
        options[s] = link_options(maps[i],evidence,factor,offset,gap)
    if not any(options.values()):
        return [],{}
    combinations = []
    for s,linked in options.items():
        opposite = maps[2 if s=='sup' else 1]
        for t,h,n in linked:
            combinations.extend((t,d,h,n) if s=='sup' else (d,t,h,n) for d in opposite)
    if len(sides)==2:
        combinations.extend((u,d,a+b,c+e) for (u,a,c),(d,b,e) in product(options['sup'],options['deep']))
    out = []
    seen = set()
    x0,y0 = offset
    height = roi['box'][3]-roi['box'][1]
    for u,d,h,n in combinations:
        xs = sorted(u.keys()&d.keys())
        if coverage(xs,factor)<.30*1046-1e-8:
            continue
        pts = [(x,u[x],d[x]) for x in xs]
        if min((b-a for _,a,b in pts),default=0)<=2:
            continue
        sig = tuple(pts)
        if sig in seen:
            continue
        seen.add(sig)
        curve = fit_float(pts,height*factor,1046)
        if curve is None:
            continue
        quality = separator_evidence(pts,maps[1]+maps[2],factor,coverage)
        out.append(dict(pair_id=len(pool)+len(out), source_decoder='D1', ranking_height=height,
            curve=map_curve(curve,dict(sx=1/factor,sy=1/factor,tx=x0,ty=y0)),
            observations=[(x/factor+x0,a/factor+y0,b/factor+y0) for x,a,b in pts],
            old_quality=quality['shared']/1046, separator=dict(single_track_quality=(quality['shared']-quality['strongest'])/1046),
            weak_columns=h, gap_columns=n))
    return out,{s:len(v) for s,v in options.items()}


def robust_terminal(curve, xy, width, fraction):
    """Replace a terminal kink when an adjacent observed trend supports the change."""
    out = deepcopy(curve)
    kx = np.asarray(curve['knots_x'])
    ky = np.asarray(curve['knots_y'])
    delta = width/1046
    for i,adj,ray in ((0,1,'left_ray'),(-1,-2,'right_ray')):
        x = kx[i]
        section = xy[(abs(xy[:,0]-x)<=max(fraction*width,abs(x-kx[adj]))) &
                     (xy[:,0]>=kx[0]) & (xy[:,0]<=kx[-1])]
        if len(section)<8 or np.ptp(section[:,0])<8*delta:
            continue
        dx = section[:,0]-x
        design = np.column_stack((dx,np.ones(len(dx))))
        yy = section[:,1]
        coeff = np.linalg.lstsq(design,yy,rcond=None)[0]
        for _ in range(4):
            res = yy-design@coeff
            scale = max(delta,float(1.4826*np.median(abs(res-np.median(res)))))
            weights = np.sqrt(np.minimum(1.,1.5*scale/np.maximum(abs(res),1e-8)))
            coeff = np.linalg.lstsq(design*weights[:,None],yy*weights,rcond=None)[0]
        slope,tail = coeff
        inside_slope = (ky[i]-ky[adj])/(kx[i]-kx[adj])
        next_index = 2 if i==0 else -3
        neighbor_slope = (ky[adj]-ky[next_index])/(kx[adj]-kx[next_index])
        turn = abs(np.arctan(inside_slope)-np.arctan(neighbor_slope))
        # A supported gradual bend should survive. Require both a kink and local error improvement.
        if turn<np.deg2rad(3) or abs(tail-ky[i])<delta:
            continue
        revised = ky.copy()
        revised[i] = tail
        tail_set = section[:,0]<=kx[adj] if i==0 else section[:,0]>=kx[adj]
        if tail_set.sum()<5:
            continue
        old_err,new_err = [np.median(abs(yy[tail_set]-np.interp(section[tail_set,0],kx,k))) for k in (ky,revised)]
        cap = max(5*delta,.03*(kx[-1]-kx[0]))
        new_slope = (tail-ky[adj])/(x-kx[adj])
        if abs(tail-ky[i])>cap or new_err>old_err-.15*delta:
            continue
        if abs(np.arctan(new_slope)-np.arctan(neighbor_slope))>=turn:
            continue
        out['knots_y'][i] = float(tail)
        out[ray] = [float(slope),float(tail-slope*x)]
    return out


def reframe(pair, width, fraction, sides, native, evidence_box):
    curve = deepcopy(pair['curve'])
    for side,col in (('sup',1),('deep',2)):
        if side in sides:
            curve[side] = robust_terminal(curve[side],np.asarray(pair['observations'])[:,[0,col]],width,fraction)
    if curve==pair['curve']:
        return pair
    xs = np.unique(np.concatenate([curve[s]['knots_x'] for s in SIDES]))
    y = [np.interp(xs,curve[s]['knots_x'],curve[s]['knots_y']) for s in SIDES]
    _,top,_,bottom = evidence_box
    if np.any(y[0]>=y[1]) or np.any(y[0]<max(0,top)) or np.any(y[1]>=min(bottom,native['height'])):
        return pair
    return {**pair,'curve':curve}


def controls():
    x = np.arange(100)
    left = {float(i):25. for i in x[:45]}
    right = {float(i):25. for i in x[46:]}
    off = {float(i):29. for i in x[46:]}
    yes = np.zeros((70,100),bool)
    yes[25,45] = True
    no = np.zeros_like(yes)
    assert len(link_options([left,right],yes,1,(0,0),.02))==1
    assert not link_options([left,right],no,1,(0,0),.02)
    assert not link_options([left,off],yes,1,(0,0),.02)
    curve = dict(knots_x=[0,25,50,75,99],knots_y=[25,25,25,25,22.3],left_ray=[0,25],right_ray=[0,22.3])
    y = np.full(100,25.);y[-1]=22.3
    edited = robust_terminal(curve,np.column_stack((x,y)),100,.20)
    assert edited['knots_y'][-1]>24 and edited['knots_y'][:4]==curve['knots_y'][:4]
    bend = dict(knots_x=[0,25,50,75,99],knots_y=[20,21,24,29,36],left_ray=[.04,20],right_ray=[.29,7.3])
    assert robust_terminal(bend,np.column_stack((x,np.interp(x,bend['knots_x'],bend['knots_y']))),100,.20)==bend
    return {'same_layer_gap':True,'no_probability_in_gap':True,'parallel_false_track':True,
            'isolated_terminal_point':True,'supported_bend_retained':True}
