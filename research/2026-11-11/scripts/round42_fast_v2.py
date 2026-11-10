"""Equivalent boundary perturbations with immutable support sharing.

Profiling found repeated deep copies of observed support columns dominate CPU.
Only the modified side's ordinates/ray are copied. All consumers are read-only.
Full-output exact equality and input immutability are tested before adoption.
"""
import math,sys
from scripts.round42_fast import install as scalar_install

def perturb_boundary(cv,side,offset=0.,ray=None,angle=0.):
    q=dict(cv);f=dict(cv[side]);q[side]=f
    if offset:
        f['knots_y']=[y+offset for y in f['knots_y']]
        for r in ('left_ray','right_ray'):f[r]=[f[r][0],f[r][1]+offset]
    if ray:
        at=0 if ray=='left_ray' else -1;x=f['knots_x'][at];y=f['knots_y'][at]
        slope=math.tan(math.atan(f[ray][0])+math.radians(angle));f[ray]=[slope,y-slope*x]
    return q

def install():
    scalar_install()
    import scripts.round40_core as r40
    original=r40.perturb_boundary
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and module is not None:
            for field,obj in list(vars(module).items()):
                if obj is original:setattr(module,field,perturb_boundary)

