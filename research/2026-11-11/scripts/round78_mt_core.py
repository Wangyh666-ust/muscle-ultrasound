"""MT-only opt-in geometry adapter. No fitting, model call, labels or PA/FL changes."""
import numpy as np

def qualified_at(path,x,native_step):
    xs=np.asarray(path['x'],float);q=np.asarray(path['qualified'],bool)
    if not len(xs) or x<xs[0]-1e-7 or x>xs[-1]+1e-7:return False,'outside_completed_path'
    i=int(np.searchsorted(xs,x));i=min(i,len(xs)-1)
    if abs(xs[i]-x)<1e-7:return bool(q[i]),'qualified_node' if q[i] else 'unqualified_node'
    if i==0:return False,'outside_completed_path'
    if xs[i]-xs[i-1]>native_step*(1+1e-6):return False,'missing_curve_samples'
    passed=bool(q[i-1] and q[i]);return passed,'qualified_bracket' if passed else 'unqualified_bracket'

def measure_mt(pair,curve,positions,width,height,px_per_mm,native_step,geometry='old',support='old'):
    if curve is None or pair is None or len(positions)!=3:return dict(value=None,reasons=['no_apo_boundaries'],positions=positions,points=[])
    reasons=[];points=[];raw=np.asarray(curve['paired_x']);threshold=curve['max_sample_gap'];gaps=[]
    for x in positions:
        rec=dict(x=float(x),sides={});raw_distance=float(np.min(abs(raw-x))) if len(raw) else float('inf');rec['raw_joint_distance']=raw_distance;rec['raw_threshold']=threshold
        if support=='old':good=raw_distance<=threshold;gaps.append(not good)
        else:good=True
        values={}
        for role,side in [('sup','upper'),('deep','lower')]:
            path=pair[side];passed,why=qualified_at(path,x,native_step)
            yy=float(np.interp(x,path['x'],path['y'])) if geometry=='complete' else float(np.interp(x,curve[role]['knots_x'],curve[role]['knots_y']))
            values[side]=yy;rec['sides'][side]=dict(y=yy,qualified=passed,qualification_reason=why)
            if support=='qualified':good=good and passed
        rec['support_passed']=bool(good);rec['gap_px']=values['lower']-values['upper'];points.append(rec)
    if (support=='old' and any(gaps)) or (support=='qualified' and not all(p['support_passed'] for p in points)):
        reasons.append('measurement_in_support_gap' if support=='old' else 'measurement_without_qualified_boundary_support')
    if any(not 0<=p['sides'][s]['y']<height for p in points for s in ('upper','lower')):reasons.append('fitted_band_outside_image')
    if any(p['gap_px']<=0 for p in points):reasons.append('nonpositive_thickness')
    value=None if reasons else float(np.mean([p['gap_px'] for p in points])/(px_per_mm or 1.))
    return dict(value=value,reasons=reasons,positions=positions,points=points,geometry=geometry,support_policy=support,
                units='mm' if px_per_mm else 'px',raw_and_qualified_flags_unchanged=True)
