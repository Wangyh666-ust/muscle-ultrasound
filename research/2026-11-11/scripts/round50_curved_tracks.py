"""Frozen-probability ablation of oriented hysteresis and curved centerlines."""
import math
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from scripts.fascicle_geometry import extract_segments
from scripts.round37_core import gate

ARMS = ('A', 'B', 'C', 'D')
CONNECT = np.ones((3, 3), bool)
REF_WIDTH = 1046.
SETTINGS = dict(high=.50, extension_ref=15., perpendicular_ref=2.5,
                join_gap_ref=14., join_perpendicular_ref=2.5, join_angle_deg=13.,
                ambiguity_ref=1., min_bow_ref=1.7, max_turn_deg=40.,
                local_width_ref=8., holdout_relative_gain=.25)


def component_pixels(binary):
    labels, count = ndimage.label(binary, structure=CONNECT)
    for sid, box in enumerate(ndimage.find_objects(labels, count), start=1):
        if box is not None:
            y, x = np.nonzero(labels[box] == sid)
            yield sid, np.column_stack((x + box[1].start, y + box[0].start))


def axis_delta(a, b):
    return abs((a-b+90.) % 180.-90.)


def segment(points, sid):
    # Delegate geometry to the baseline PCA, preserving row-major pixels and IDs.
    lo, hi = points.min(0), points.max(0)
    crop = np.zeros((hi[1]-lo[1]+1, hi[0]-lo[0]+1), bool)
    xy = points-lo
    crop[xy[:, 1], xy[:, 0]] = True
    ex = extract_segments(crop)
    one = ex['segments'] + ex['rejected']
    assert len(one)==1
    s=one[0]
    s['label_id']=sid
    for field in ('p0', 'p1', 'centroid'):
        s[field]=(np.asarray(s[field])+lo).tolist()
    return s


def connected_points(points):
    lo, hi=points.min(0),points.max(0)
    xy=points-lo
    mask=np.zeros((hi[1]-lo[1]+1, hi[0]-lo[0]+1),bool)
    mask[xy[:,1],xy[:,0]]=True
    return [p+lo for _,p in component_pixels(mask)]


def vectors(direction):
    angle=math.radians(direction)
    u=np.array([math.cos(angle),math.sin(angle)])
    return u, np.array([-u[1],u[0]])


def likely_join(a, b, native_ref):
    if min(a['length_px'], b['length_px']) < 8*native_ref:
        return False
    if axis_delta(a['direction_deg'], b['direction_deg'])>SETTINGS['join_angle_deg']:
        return False
    def centered_ends(s):
        u,_=vectors(s['direction_deg'])
        center=np.asarray(s['centroid'])
        endpoints=np.asarray([s['p0'],s['p1']])
        return center+((endpoints-center)@u)[:,None]*u

    for first, other in ((a,b),(b,a)):
        u,v=vectors(first['direction_deg'])
        ea,eb=centered_ends(first),centered_ends(other)
        aa=sorted(ea@u)
        bb=sorted(eb@u)
        gap=max(aa[0],bb[0])-min(aa[1],bb[1])
        if not -2*native_ref<=gap<=SETTINGS['join_gap_ref']*native_ref:
            return False
        endpoints=[x-y for x in ea for y in eb]
        offset=min(endpoints,key=np.linalg.norm)
        if abs(offset@v)>SETTINGS['join_perpendicular_ref']*native_ref:
            return False
    return True


def expand(prob, high, roi, low, shapes):
    """Grow from oriented high seeds through observed low pixels, with exclusive ownership."""
    native_ref=max(roi['box'][2]-roi['box'][0],1)/REF_WIDTH
    hi_lab,_=ndimage.label(high,structure=CONNECT)
    low_lab,n=ndimage.label(prob>low,structure=CONNECT)
    out=[]
    audit=dict(seeds_joined=0,weak_pixels_added=0,conflict_pixels=0)
    for sid, box in enumerate(ndimage.find_objects(low_lab,n),start=1):
        if box is None:
            continue
        low_mask=low_lab[box]==sid
        high_labels=np.where(low_mask,hi_lab[box],0)
        ids=np.unique(high_labels)
        ids=[int(i) for i in ids if i]
        if not ids:
            continue
        compatible={a:[b for b in ids if b!=a and likely_join(shapes[a],shapes[b],native_ref)]
                    for a in ids}
        groups=[]
        seen=set()
        for a in ids:
            if a in seen:
                continue
            pair=[b for b in compatible[a] if compatible[b]==[a]] if len(compatible[a])==1 else []
            group=[a,*pair]
            groups.append(group)
            seen.update(group)
        weak=low_mask & (high_labels==0)
        yy,xx=np.nonzero(weak)
        owner=np.full(len(xx),-1,int)
        best=np.full(len(xx),np.inf)
        second=best.copy()
        for i,group in enumerate(groups):
            seed=np.isin(high_labels,group)
            if not len(xx) or max(shapes[a]['length_px'] for a in group)<8*native_ref:
                continue
            dist,nearest=ndimage.distance_transform_edt(~seed,return_indices=True)
            dy=yy-nearest[0,yy,xx]
            dx=xx-nearest[1,yy,xx]
            source_ids=high_labels[nearest[0,yy,xx],nearest[1,yy,xx]]
            perp=np.zeros(len(xx))
            for anchor in group:
                u,v=vectors(shapes[anchor]['direction_deg'])
                source=source_ids==anchor
                perp[source]=abs(dx[source]*v[0]+dy[source]*v[1])
            d=dist[yy,xx]
            supported=(d<=SETTINGS['extension_ref']*native_ref) & (
                perp<=max(1.1,SETTINGS['perpendicular_ref']*native_ref)+.12*d)
            cost=np.where(supported,d+1.5*perp,np.inf)
            better=cost<best
            second=np.minimum(second,np.where(better,best,cost))
            best=np.where(better,cost,best)
            owner[better]=i
        conflict=np.isfinite(second) & (second<=best+SETTINGS['ambiguity_ref']*native_ref)
        owner[conflict]=-1
        audit['conflict_pixels']+=int(conflict.sum())
        offset=np.array([box[1].start,box[0].start])
        for i,group in enumerate(groups):
            mask=np.isin(high_labels,group)
            take=owner==i
            mask[yy[take],xx[take]]=True
            for _,p in component_pixels(mask):
                anchors={int(v) for v in high_labels[p[:,1],p[:,0]] if v}
                if not anchors:
                    continue
                audit['seeds_joined']+=len(anchors)>1
                audit['weak_pixels_added']+=int((high_labels[p[:,1],p[:,0]]==0).sum())
                out.append((min(anchors),p+offset))
    # Different strong seeds remain different candidates unless an unambiguous observed bridge joined them.
    assert len({i for i,_ in out})==len(out)
    return out,audit


def path_geometry(points, seg, native_ref, diagnostics=False):
    """Fit a quadratic centerline to cross-section medians, with split-section checks."""
    u,v=vectors(seg['direction_deg'])
    origin=np.asarray(seg['centroid'])
    x=(points-origin)@u
    y=(points-origin)@v
    span=float(np.ptp(x))
    if span < 18*native_ref:
        return None
    n=max(5,min(120,round(span/max(1.6,3*native_ref))))
    bins=np.minimum(np.floor((x-x.min())/span*n).astype(int),n-1)
    centers=[]
    split_sections=0
    for i in range(n):
        use=bins==i
        if not use.any():
            continue
        yy=np.sort(y[use])
        if len(yy)>=6:
            split=np.arange(1,len(yy))
            separate=(np.diff(yy)>max(1.55,2.*native_ref))
            size=np.minimum(split,len(yy)-split)
            split_sections+=bool(np.any(separate & (size>=max(2,.18*len(yy)))))
        centers.append([float(np.median(x[use])),float(np.median(yy))])
    if len(centers)<6:
        return None
    z,c=np.asarray(centers).T
    zscaled=z/span
    losses=[[],[]]
    for residue in range(3):
        test=np.arange(len(z))%3==residue
        for order in (1,2):
            fit=np.polyfit(zscaled[~test],c[~test],order)
            losses[order-1].extend((np.polyval(fit,zscaled[test])-c[test])**2)
    linear,curve=np.sqrt(np.mean(losses,axis=1))
    q=np.polyfit(zscaled,c,2)
    path_x=np.linspace(z[0],z[-1],len(z))
    path_y=np.polyval(q,path_x/span)
    slope=np.polyval(np.polyder(q),path_x/span)/span
    turn=abs(math.degrees(math.atan(slope[-1])-math.atan(slope[0])))
    bow=float(np.max(abs(path_y-np.linspace(path_y[0],path_y[-1],len(path_y)))))
    yhat=np.polyval(q,x/span)
    norm=(y-yhat)/np.hypot(1,np.polyval(np.polyder(q),x/span)/span)
    widths=[float(np.ptp(norm[bins==i])) for i in range(n) if np.any(bins==i)]
    width_ref=float(np.quantile(widths,.95)/native_ref)
    over_width=max(widths)/native_ref
    path=origin+path_x[:,None]*u+path_y[:,None]*v
    reasons=[]
    if split_sections:
        reasons.append('split_cross_sections')
    if width_ref>SETTINGS['local_width_ref'] or over_width>1.5*SETTINGS['local_width_ref']:
        reasons.append('locally_wide')
    if turn>SETTINGS['max_turn_deg']:
        reasons.append('excessive_turn')
    if bow<SETTINGS['min_bow_ref']*native_ref:
        reasons.append('weak_curvature')
    if linear-curve<max(.15*native_ref,SETTINGS['holdout_relative_gain']*linear):
        reasons.append('no_heldout_gain')
    if len(z)/n < .92:
        reasons.append('poor_support')
    length=path_length(path)
    return dict(path=path.tolist(),medians=(origin+z[:,None]*u+c[:,None]*v).tolist(),
                accepted=not reasons,reasons=reasons,arc_px=length,local_width_ref=width_ref,
                bow_ref=bow/native_ref,turn_deg=turn,linear_cv_ref=float(linear/native_ref),
                curve_cv_ref=float(curve/native_ref))


def path_length(path):
    return float(np.linalg.norm(np.diff(np.asarray(path),axis=0),axis=1).sum())


def decode_shapes(components, roi, cfg, details=False, original_shapes=None):
    straight, curved, audit=[],[],[]
    ref=max(roi['box'][2]-roi['box'][0],1)/REF_WIDTH
    for sid,points in components:
        s=original_shapes[sid] if original_shapes is not None else segment(points,sid)
        allowed,g=gate(s,roi,cfg['reference_filter'])
        base=dict(id=sid,path=[s['p0'],s['p1']],arc_px=s['length_px'],mode='straight')
        if allowed:
            straight.append(base)
        curve=None
        excluded=[flag for flag in g['flags'] if flag not in ('wide_or_branch_suspected','low_linearity')]
        if not excluded and g['reference_length']>=45:
            curve=path_geometry(points,s,ref)
        if curve and curve['accepted']:
            curved.append(dict(id=sid,path=curve['path'],arc_px=curve['arc_px'],mode='curve',
                               bow_ref=curve['bow_ref'],local_width_ref=curve['local_width_ref']))
        elif allowed:
            curved.append(base)
        if details:
            audit.append(dict(id=sid,gate=g,curve=curve,area=s['area_px']))
    return straight,curved,audit


def decode(prob, high, roi, cfg, low, details=False):
    assert np.array_equal(prob>SETTINGS['high'],high)
    components=list(component_pixels(high))
    shapes={sid:segment(points,sid) for sid,points in components}
    a,c,ac=decode_shapes(components,roi,cfg,details,shapes)
    expanded,ha=expand(prob,high,roi,low,shapes)
    b,d,bd=decode_shapes(expanded,roi,cfg,details)
    return dict(A=a,B=b,C=c,D=d),dict(high_components=len(components),hysteresis=ha,base=ac,expanded=bd)


def annotation_paths(binary, span):
    """Trace label center medians independently of the proposed quadratic model."""
    paths=[]
    ref=span/REF_WIDTH
    for sid,p in component_pixels(binary):
        s=segment(p,sid)
        if s['length_px']/ref<45 or s['area_px']<15:
            continue
        u,v=vectors(s['direction_deg'])
        o=np.asarray(s['centroid'])
        x=(p-o)@u
        y=(p-o)@v
        bins=np.minimum(49,np.floor((x-x.min())/np.ptp(x)*50).astype(int))
        n=max(7,min(40,int(s['length_px']/ref/4)))
        bins=bins*n//50
        z=[]
        wide=0
        for i in np.unique(bins):
            m=bins==i
            z.append([np.median(x[m]),np.median(y[m])])
            wide+=np.ptp(y[m])>8*ref
        if len(z)<.9*n or wide:
            continue
        z=np.asarray(z)
        lin=np.interp(z[:,0],[z[0,0],z[-1,0]],[z[0,1],z[-1,1]])
        bow=float(np.quantile(abs(z[:,1]-lin),.8)/ref)
        path=o+z[:,0,None]*u+z[:,1,None]*v
        paths.append(dict(id=sid,path=path.tolist(),arc_px=path_length(path),
                          mode='label_arc' if bow>=2.5 and path_length(path)/ref>=80 else 'label_line',
                          bow_ref=bow))
    return paths


def resample(path,step):
    p=np.asarray(path,float)
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    n=max(3,min(240,int(s[-1]/max(step,.2))+1))
    at=np.linspace(0,s[-1],n)
    xy=np.column_stack([np.interp(at,s,p[:,i]) for i in (0,1)])
    delta=np.gradient(xy,axis=0)
    tangent=np.degrees(np.arctan2(delta[:,1],delta[:,0]))%180
    return xy,tangent


def matching(refs, predictions, span):
    """One-to-one spatial matching with independent reference coverage and local direction."""
    pitch=span/REF_WIDTH
    radius=max(1.,3.0*pitch)
    rs=[resample(a['path'],2*pitch) for a in refs]
    ps=[resample(a['path'],2*pitch) for a in predictions]
    trees=[cKDTree(p) for p,_ in ps]
    score=np.zeros((len(rs),len(ps)))
    details={}
    hits=np.zeros(score.shape,bool)
    for i,(a,da) in enumerate(rs):
        alo,ahi=a.min(0)-radius,a.max(0)+radius
        for k,(b,db) in enumerate(ps):
            if np.any(b.max(0)<alo) or np.any(b.min(0)>ahi):
                continue
            dist,idx=trees[k].query(a)
            angle=abs((db[idx]-da+90)%180-90)
            good=(dist<=radius) & (angle<=20)
            coverage=float(good.mean())
            if coverage<.2:
                continue
            backward,rev=cKDTree(a).query(b)
            agreement=abs((da[rev]-db+90)%180-90)<=20
            precision=float(((backward<=radius)&agreement).mean())
            if precision<.18:
                continue
            hits[i,k]=coverage>=.4 and precision>=.2
            score[i,k]=2*coverage*precision/(coverage+precision)
            ix=np.flatnonzero(good)
            consecutive=max(map(len,np.split(ix,np.flatnonzero(np.diff(ix)>1)+1)))
            details[i,k]=dict(coverage=coverage, precision=precision,
                              continuous=consecutive/len(a),
                              distance=float(np.mean(dist[good])/pitch),
                              angle=float(np.mean(angle[good])),matched=True)
    assignments={i:k for i,k in zip(*linear_sum_assignment(score,maximize=True)) if score[i,k]>0}
    empty=dict(coverage=0.,precision=None,continuous=0.,distance=None,angle=None,matched=False)
    records=[]
    for i,ref in enumerate(refs):
        k=assignments.get(i)
        q=details.get((i,k),empty)
        records.append(dict(**q,reference=ref['id'],mode=ref['mode'],
                            candidate=predictions[k]['id'] if k is not None else None,
                            candidate_mode=predictions[k]['mode'] if k is not None else None))
    return dict(references=records, predictions=len(ps), likely_joins=int((hits.sum(0)>1).sum()),
                multi_candidates=int((hits.sum(1)>1).sum()))
