"""Conservative original-pixel MT. PA/FL are unavailable, never inferred from MT.
Default positions are common-support quartiles, not assumed expert locations.
"""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
from PIL import Image

def extract_apo_bands(mask, roi=None):
    if mask.ndim!=2 or not np.isfinite(mask).all():return None,'invalid_mask'
    H,W=mask.shape
    x0,y0,x1,y1=roi if roi is not None else (0,0,W-1,H-1)
    if not (0<=x0<x1<W and 0<=y0<y1<H):return None,'invalid_roi'
    xs=[]; sup=[]; deep=[]; ambiguous=0; occupied=0
    for x in range(int(x0),int(x1)+1):
        ys=np.flatnonzero(mask[int(y0):int(y1)+1,x])+int(y0)
        if not len(ys):continue
        occupied+=1
        runs=np.split(ys,np.flatnonzero(np.diff(ys)>max(2,round(H*.002)))+1)
        if len(runs)>2:ambiguous+=1;continue
        if len(runs)!=2:continue
        a,b=[float(np.median(v)) for v in runs]
        if b-a<=max(2,H*.005):continue
        xs.append(x);sup.append(a);deep.append(b)
    if not occupied:return None,'no_mask'
    if ambiguous/occupied>.15:return None,'ambiguous_multiple_bands'
    if len(xs)<max(8,.3*(x1-x0+1)):return None,'insufficient_paired_support'
    xs=np.array(xs);sup=np.array(sup);deep=np.array(deep)
    if xs[-1]-xs[0]<.3*(x1-x0):return None,'short_common_support'
    fits={'sup':np.polyfit(xs,sup,1),'deep':np.polyfit(xs,deep,1)}
    residuals=[np.abs(np.polyval(fits[k],xs)-v) for k,v in [('sup',sup),('deep',deep)]]
    if max(np.quantile(v,.9) for v in residuals)>max(3,.02*H):
        return None,'nonlinear_or_unstable_bands'
    gap=np.polyval(fits['deep']-fits['sup'],[xs[0],xs[-1]])
    if (gap<=0).any():return None,'crossed_or_inverted_lines'
    fits.update(support=(int(xs[0]),int(xs[-1])),paired_x=xs,
                max_sample_gap=max(2,.015*W),image_shape=(H,W),
                roi_source='explicit' if roi is not None else 'mask_paired_support')
    return fits,None

def mt_at_positions(fits,positions):
    x=np.asarray(positions,dtype=float);lo,hi=fits['support']
    if len(x)!=3 or not np.isfinite(x).all():return None,'invalid_measurement_positions'
    if (x<lo).any() or (x>hi).any():return None,'measurement_outside_support'
    if any(np.min(np.abs(fits['paired_x']-v))>fits['max_sample_gap'] for v in x):
        return None,'measurement_in_support_gap'
    mt=np.polyval(fits['deep']-fits['sup'],x)
    if (mt<=0).any() or not np.isfinite(mt).all():return None,'nonpositive_thickness'
    H,_=fits['image_shape']
    for key in ('sup','deep'):
        y=np.polyval(fits[key],x)
        if (y<0).any() or (y>=H).any():return None,'fitted_band_outside_image'
    return mt.tolist(),None

def measure_mt(mask,positions=None,roi=None,px_per_mm=None):
    fits,fail=extract_apo_bands(mask,roi)
    if fail:return {'status':'rejected','reason':fail},None
    lo,hi=fits['support']
    xs=positions if positions is not None else [lo+f*(hi-lo) for f in (.25,.5,.75)]
    mt,fail=mt_at_positions(fits,xs)
    if fail:return {'status':'rejected','reason':fail},fits
    scale=float(px_per_mm) if px_per_mm is not None else None
    if scale is not None and (not np.isfinite(scale) or scale<=0):
        return {'status':'rejected','reason':'invalid_scale'},fits
    return {'status':'candidate','x_positions':list(map(float,xs)),'mt_px':mt,
            'mt_mean_px':float(np.mean(mt)),
            'mt_mm':[v/scale for v in mt] if scale else None,
            'mt_mean_mm':float(np.mean(mt))/scale if scale else None,
            'px_per_mm':scale,'support':list(fits['support']),
            'position_rule':'explicit' if positions is not None else 'paired_support_quartiles',
            'roi_source':fits['roi_source'],'pa_deg':None,'fl_mm':None,
            'pa_fl_status':'not_implemented','reference_comparison':'not_evaluated'},fits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['apo'],default='apo')
    ap.add_argument('--masks_dir',required=True);ap.add_argument('--candidates',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True);(out/'sketches').mkdir(exist_ok=True)
    from .data import UmudTrainData,ROOT,read_image_gray3
    samples={s['image']['filename']:s for s in UmudTrainData('apo','dev').samples}
    rows=list(csv.DictReader(open(args.candidates,encoding='utf-8')));results=[]
    for r in rows:
        fn=r['filename'];p=Path(args.masks_dir)/Path(fn).with_suffix('.png')
        if fn not in samples:raise ValueError('Not an apo dev image: '+fn)
        mask=np.asarray(Image.open(p).convert('L'))>127
        if mask.shape!=(int(r['height']),int(r['width'])):raise ValueError('Shape mismatch: '+fn)
        result,fits=measure_mt(mask);result.update(filename=fn,group_id=r['group_id']);results.append(result)
        rgb=read_image_gray3(os.path.join(ROOT,samples[fn]['image']['path']))
        if fits:
            xs=np.arange(fits['support'][0],fits['support'][1]+1)
            for k,color in [('sup',[255,255,0]),('deep',[255,0,255])]:
                yy=np.rint(np.polyval(fits[k],xs)).astype(int)
                keep=(yy>=0)&(yy<mask.shape[0]);rgb[yy[keep],xs[keep]]=color
        Image.fromarray(rgb).save(out/'sketches'/Path(fn).with_suffix('.png'))
    (out/'geometry.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    good=sum(r['status']=='candidate' for r in results)
    summary={'n_candidates':len(results),'n_outputs':good,'n_rejected':len(results)-good,
             'meaning':'Output availability, not anatomical correctness',
             'units':'pixels; no manual scale supplied','expert_accuracy':'not_evaluated',
             'failures':[r for r in results if r['status']=='rejected']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
