"""Read-only label reconstruction and training-domain fit, no optimizer or gradients."""
from scripts.boundary_weak_response_infer_20261111 import (
    ROOT,OUT,j,sha,write,stats,near,np,cv2,torch,time,GuidedBoundaryUNet,predict,
)
from PIL import Image
from src.umud_baseline.data import read_mask,resize_long_side,read_image_gray3
from src.umud_baseline.boundary_identity import extract_identity,align_identity


def main():
    m=j(ROOT/'runs/round60_boundary_objective_v1/payload/training_manifest.json');p=j(OUT/'protocol.json')
    plan=dict(purpose='Read-only follow-up of target transfer and source-domain model fit, not new training',
        registered_before_inference=True,model_sha256=p['model_sha256'],source_manifest_sha256=sha(ROOT/'runs/round60_boundary_objective_v1/payload/training_manifest.json'),
        images=672,variants=673,raw_label_check='Rebuild both identity masks from source union using frozen extraction/alignment/512 transfer; exact compare saved image,target,known,canvas; source and input SHA',
        inference='One single512 evaluation per original training image; no augmentation. Score each variant, primary variant0 per image for aggregate.',
        primary_metrics='Upper/lower reference-midpoint max within2 network pixels, role/union/fused>.18/.5 coverage; known-band Dice; unknown regions never background negatives',
        limitation='In-sample fit, not validation. Broad grayscale anatomy and target muscle semantics cannot be certified by mask topology. No held-out data added to training.',
        default_changes=False,training=False,optimizer_steps=0,subagents=False)
    path=OUT/'training_source_protocol.json'
    if path.exists():assert j(path)==plan
    else:write(path,plan)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    model=GuidedBoundaryUNet();assert sha(ROOT/p['model'])==p['model_sha256']
    model.load_state_dict(torch.load(ROOT/p['model'],map_location='cpu',weights_only=True)['model'],strict=True);model.cuda().eval().requires_grad_(False)
    t0=time.monotonic();records=[]
    for n,s in enumerate(m['samples']):
        dest=OUT/f"training_records/{s['image_id']}.json"
        if dest.exists():records.extend(j(dest)['records']);continue
        with Image.open(ROOT/s['image_path']) as im:wh=im.size
        image,_=resize_long_side(read_image_gray3(ROOT/s['image_path']),512,False);h,w=image.shape[:2]
        base=np.pad(image[:,:,0],((0,512-h),(0,512-w)));prob=predict(model,np.repeat(base[:,:,None],3,2),True);local=[]
        for vi,v in enumerate(s['variants']):
            source=ROOT/v['label_source'];input_path=ROOT/'runs/round60_boundary_objective_v1/payload'/v['path']
            assert sha(source)==v['label_sha256'];assert sha(input_path)==v['sha256']
            native=read_mask(source);target,valid,meta=extract_identity(native)
            assert meta['components']==2 and meta['reason'] is None,(s['image_id'],meta)
            target,valid=align_identity(target,valid,wh)
            parts=np.stack([np.pad(resize_long_side(a,512,True)[0],((0,512-h),(0,512-w))) for a in (*target,valid)]).astype(np.uint8)
            identity=parts[:2]&parts[2];common=identity.any(1).all(0);canvas=np.zeros((512,512),np.uint8);canvas[:h,:w]=1
            known=parts[2]&canvas&common[None];identity &= known
            with np.load(input_path) as z:actual=tuple(z[k].copy() for k in ('image','target','known','canvas'))
            assert all(np.array_equal(a,b) for a,b in zip(actual,(base,identity,known,canvas))),s['image_id']
            assert identity.any((1,2)).all();assert not (identity[0]&identity[1]).any()
            ends=511-identity[0,::-1].argmax(0);starts=identity[1].argmax(0);assert (ends[common]<starts[common]).all()
            sides={}
            for i,side in enumerate(('upper','lower')):
                a=identity[i];xx=np.flatnonzero(a.any(0));yy=(a.argmax(0)[xx]+511-a[::-1].argmax(0)[xx])/2
                gt=np.column_stack([xx,yy]);fields=dict(role=prob[i+1],union=prob[0],fused=prob[i+1]*(.4+.6*prob[0]))
                coverage={k:stats(near(f,gt,2.)) for k,f in fields.items()}
                pred=(prob[i+1]>=.5)&known.astype(bool)
                sides[side]=dict(columns=len(xx),label_gap=int(np.diff(xx).max(initial=1)-1),heads=coverage,
                    dice=float((2*(pred&a.astype(bool)).sum()+1)/(pred.sum()+a.sum()+1)),
                    source_columns=int(target[i].any(0).sum()))
            local.append(dict(id=s['image_id'],group=s['group_id'],variant=v['variant'],primary=vi==0,source_file_sha256=v['label_sha256'],
                input_sha256=v['sha256'],source_to512_exact=True,side_order_preserved=True,both_labels_retained=True,sides=sides))
        write(dest,dict(records=local,model_sha256=p['model_sha256']));records.extend(local)
        if (n+1)%80==0:print('TRAINING_SOURCE_FIT_AUDIT',n+1,672,round(time.monotonic()-t0,2),flush=True)
    primary=[r for r in records if r['primary']];assert len(primary)==672 and len(records)==673
    summary={}
    for side in ('upper','lower'):
        rr=[r['sides'][side] for r in primary]
        summary[side]=dict(images=len(rr),mean_dice=float(np.mean([r['dice'] for r in rr])),
            mean_role_over05=float(np.mean([r['heads']['role']['over05'] for r in rr])),
            mean_fused_over018=float(np.mean([r['heads']['fused']['over018'] for r in rr])),
            role_below80=[r['id'] for r in primary if r['sides'][side]['heads']['role']['over05']<.8],
            fused_below80=[r['id'] for r in primary if r['sides'][side]['heads']['fused']['over018']<.8])
    write(OUT/'training_source_summary.json',dict(images=672,variants=673,exact_rebuilds=len(records),
        both_side_labels_retained=673,side_order_preserved=673,max_512_label_gap=max(r['sides'][s]['label_gap'] for r in records for s in ('upper','lower')),
        metrics=summary,seconds=time.monotonic()-t0,training=False,optimizer_steps=0,model_sha256=p['model_sha256']))
    print('TRAINING_SOURCE_AUDIT_DONE',json_safe(summary),flush=True)


def json_safe(summary):return {s:d for s,d in summary.items()}


if __name__=='__main__':main()
