"""Replace only bounded weak-tail seam windows; never move distant anchors."""
import numpy as np
from scripts.round63_paths import pack_path,bridge,path_quality,sample


def crop_path(p,mask):
    return pack_path(p['x'][mask],p['y'][mask],p['confidence'][mask],p['observed'][mask],p['ids'],
                     kind=p['kind'][mask],events=[e for e in p['events'] if e['interval'][0]>=p['x'][mask][0] and e['interval'][1]<=p['x'][mask][-1]])


def reconstruct(paths,field,gray,valid,width,cfg,base):
    candidates=[];audits=[];seen=set()
    for a in paths:
        for b in paths:
            if set(a['ids'])&set(b['ids']):continue
            if b['x'][0]<=a['x'][0] or b['x'][-1]<=a['x'][-1]+.005*width:continue
            if b['x'][0]-a['x'][-1]>.20*width:continue
            proposals=[]
            for left in cfg['seam_trim_fractions']:
                for right in cfg['seam_trim_fractions']:
                    if not left and not right:continue
                    ma=a['x']<=a['x'][-1]-left*width;mb=b['x']>=b['x'][0]+right*width
                    if ma.sum()<8 or mb.sum()<8:continue
                    aa,bb=crop_path(a,ma),crop_path(b,mb)
                    if aa['confidence'][-1]<.5 or bb['confidence'][0]<.5:continue
                    p,reason=bridge(aa,bb,field,gray,valid,width,base)
                    if p is None:audits.append(reason);continue
                    middle=(p['x']>aa['x'][-1])&(p['x']<bb['x'][0])
                    support=(sample(field,p['x'][middle],p['y'][middle],max(1,round(width/1046.)))>=.1)|(
                        sample(gray,p['x'][middle],p['y'][middle],max(1,round(width/1046.)))>=.6)
                    cuts=np.flatnonzero(np.diff(np.r_[False,~support,False]))
                    longest=int(np.max(cuts[1::2]-cuts[::2],initial=0))
                    if len(support) and (support.mean()<.8 or longest>.02*width):
                        audits.append('unreliable_seam_corridor');continue
                    p['events'].append(dict(type='seam_window',interval=[float(aa['x'][-1]),float(bb['x'][0])],
                         trimmed_native=[float(a['x'][-1]-aa['x'][-1]),float(bb['x'][0]-b['x'][0])],
                         outside_window_unchanged=True))
                    proposals.append(p)
            if proposals:
                p=max(proposals,key=lambda p:path_quality(p,width));key=(p['ids'],p['x'][0],p['x'][-1])
                if key not in seen:candidates.append(p);seen.add(key)
    return sorted(candidates,key=lambda p:-path_quality(p,width))[:10],audits
