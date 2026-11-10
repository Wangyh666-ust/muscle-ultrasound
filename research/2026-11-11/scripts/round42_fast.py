"""Optional process-local scalar arithmetic acceleration, not a decoder change.

Never edit frozen modules on disk. The same knot choice, multiplication/addition
and distance minimum are used. Adoption requires full-output replay verification.
"""
import bisect,sys
import numpy as np

def line_at(fit,x):
    xs,ys=fit['knots_x'],fit['knots_y']
    if x<xs[0]:return fit['left_ray']
    if x>xs[-1]:return fit['right_ray']
    i=min(max(bisect.bisect_right(xs,x)-1,0),len(xs)-2)
    m=(ys[i+1]-ys[i])/(xs[i+1]-xs[i]);return [m,ys[i]-m*xs[i]]

def value_at(fit,x):
    m,b=line_at(fit,x)
    return float(np.float64(m)*x+np.float64(b))

def nearest_paired_distance(fits,x):
    if fits is None:return None
    paired=fits.get('paired_x',[])
    if not len(paired):return float('inf')
    # The frozen geometry constructs paired_x as increasing observed columns.
    x=float(x);i=bisect.bisect_left(paired,x)
    if i==0:return abs(float(paired[0])-x)
    if i==len(paired):return abs(float(paired[-1])-x)
    return min(abs(float(paired[i])-x),abs(float(paired[i-1])-x))

def install():
    import scripts.round29_core as r29
    import scripts.candidate_validity as cv
    from scripts.round38_speed import install as oldinstall
    oldinstall()
    mapping={r29.line_at:line_at,r29.value_at:value_at,cv.nearest_paired_distance:nearest_paired_distance}
    for name,module in list(sys.modules.items()):
        if not name.startswith('scripts.') or module is None:continue
        for field,obj in list(vars(module).items()):
            if callable(obj):
                try:replacement=mapping.get(obj)
                except TypeError:continue
                if replacement is not None:setattr(module,field,replacement)

