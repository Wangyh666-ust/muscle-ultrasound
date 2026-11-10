"""Freeze the image-only target masks and audit them; never repair masks using GT."""
import cv2,time
from scripts.round71_common import *
from scripts.round71_roi import locate

def view_mask(mask,rec,shape):
    matrix=np.asarray(rec['matrix'],float).copy();matrix[:,2]-=np.asarray(rec['input_box'][:2])
    return cv2.warpAffine(mask.astype(np.uint8),matrix,(shape[1],shape[0]),flags=cv2.INTER_NEAREST,borderValue=0).astype(bool)

def main():
    cv2.setNumThreads(1);md=j(R63/'metadata.json');inputs=[r for r in j(R63/'inputs.json')['records'] if r['view']=='identity'];rows=[];tic=time.monotonic()
    for i,rec in enumerate(inputs):
        key=rec['key'];image=cv2.imread(str(ROOT/md[key]['row']['image_path']),cv2.IMREAD_GRAYSCALE)
        assert image is not None and list(image.shape)==[md[key]['row']['height'],md[key]['row']['width']]
        mask,meta=locate(image);file=OUT/'domains'/rec['dataset']/(Path(rec['image_id']).stem+'.npz');file.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(file,bits=np.packbits(mask),shape=np.asarray(mask.shape),image_sha=array_sha(image),mask_sha=array_sha(mask))
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea).reshape(-1,2)[::max(1,len(max(contours,key=cv2.contourArea))//180)] if contours else np.empty((0,2),int)
        row=dict(key=key,dataset=rec['dataset'],image_id=rec['image_id'],meta=meta,path=file.relative_to(ROOT).as_posix(),sha=sha(file),contour=contour,manual=rec['manual'])
        if rec['manual']:
            old=readgz(R68/'records'/rec['dataset']/Path(rec['image_id']).stem/'identity.json.gz');native=view_mask(mask,rec,old['shape']);scores={}
            for s,ref in zip(SIDES,old['referenceLines']):
                x,y=np.rint(ref['x']).astype(int),np.rint(ref['y']).astype(int)
                valid=(x>=0)&(x<native.shape[1])&(y>=0)&(y<native.shape[0]);inside=np.zeros(len(x),bool);inside[valid]=native[y[valid],x[valid]]
                scores[s]=dict(points=len(x),clipped=int((~inside).sum()),clipped_fraction=float((~inside).mean()))
            row['reference_audit_only']=scores
        rows.append(row)
        if (i+1)%60==0:print('R71_DOMAIN',i+1,len(inputs),round(time.monotonic()-tic),flush=True)
    manual=[r for r in rows if r['manual']]
    summary=dict(originals=len(rows),manual=len(manual),ambiguities=sum(r['meta']['ambiguous'] for r in rows),
        sides={s:dict(total_points=sum(r['reference_audit_only'][s]['points'] for r in manual),
             clipped_points=sum(r['reference_audit_only'][s]['clipped'] for r in manual),
             images_any_clip=sum(r['reference_audit_only'][s]['clipped']>0 for r in manual),
             images_clip_over1pct=sum(r['reference_audit_only'][s]['clipped_fraction']>.01 for r in manual)) for s in SIDES},
        changed_using_reference=False,semantic_ROI_validated=False,identity_ROI_affine_mapped_to_views=True)
    write(OUT/'domain_manifest.json',dict(summary=summary,records=rows,locator_sha=sha(ROOT/'scripts/round71_roi.py')))
    print('R71_DOMAIN_COMPLETE',summary,flush=True)

if __name__=='__main__':main()
