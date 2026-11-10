"""Freeze diagnostic inputs and image-only matched controls before new inference."""
from collections import Counter
from pathlib import Path
import json

from scripts.round80_recognition_triage import ROOT,j,write,readgz,sha,np,load_context,references
from scripts.round80_common import verify_default

OUT=ROOT/'review/boundary_weak_response_investigation_20261111'


def image_features(image,box):
    l,t,r,b=box;pixels=image[t:b,l:r].astype(float)
    q=np.percentile(pixels,[10,50,90])
    return np.array([np.log((r-l)/(b-t)),(r-l)/max(image.shape),q[1]/128.,(q[2]-q[0])/128.])


def main():
    if (OUT/'protocol.json').exists():
        print('PROTOCOL_EXISTS',sha(OUT/'protocol.json'));return
    verify_default();rows=j(ROOT/'runs/round77_subject_layer_v1/protocol.json')['rows']
    triage=ROOT/'review/recognition_chain_triage_20261110';stats={}
    for p in sorted((triage/'records').glob('*.json.gz')):
        r=readgz(p);stats[(r['dataset'],r['id'])]=r
    weak=[k for k,r in stats.items() if any(s['flags']['response_below_low_over20pct'] for s in r['sides'].values())]
    healthy=[k for k,r in stats.items() if all(s['heads']['fused']['over018']>=.98 and s['heads']['role']['over05']>=.95 for s in r['sides'].values())]
    assert len(weak)==12
    bykey={(r['dataset'],r['image_id']):r for r in rows};candidates={}
    for n,key in enumerate(sorted(set(weak+healthy))):
        row=bykey[key];c=load_context(row,'identity');yy,xx=np.where(c['native_mask']);box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        candidates[key]=dict(row=row,features=image_features(c['image'],box),box=box,shape=list(c['image'].shape))
        if n%40==0:print('PREPARE_IMAGE_FEATURES',n,len(set(weak+healthy)),flush=True)
    pairs=[];used=set();weakgroups={bykey[k]['group'] for k in weak}
    for key in weak:
        a=candidates[key];eligible=[k for k in healthy if k[0]==key[0] and k not in used and bykey[k]['group'] not in weakgroups]
        assert eligible,key
        control=min(eligible,key=lambda k:(float(np.linalg.norm(candidates[k]['features']-a['features'])),k))
        used.add(control);pairs.append(dict(weak=list(key),control=list(control),distance=float(np.linalg.norm(candidates[control]['features']-a['features']))))
    selected=[(key,'weak') for key in weak]+[(tuple(p['control']),'matched_control') for p in pairs];items=[]
    for key,cohort in selected:
        row=bykey[key];c=load_context(row,'identity');gt,refclass=references(row);l,t,r,b=candidates[key]['box'];mx=round((r-l)*.03);my=round((b-t)*.03)
        crop=[max(0,l-mx),max(0,t-my),min(c['image'].shape[1],r+mx),min(c['image'].shape[0],b+my)]
        # GT is retained in original coordinates; only image-derived box affects input.
        gt_native={s:g-np.asarray(c['frame']['input_box'][:2]) for s,g in gt.items()}
        retained={s:float(((g[:,0]>=crop[0])&(g[:,0]<crop[2])&(g[:,1]>=crop[1])&(g[:,1]<crop[3])).mean()) for s,g in gt_native.items()}
        dst=OUT/f'inputs/{key[0]}__{key[1]}.npz';dst.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dst,image=c['image'],upper=gt_native['upper'],lower=gt_native['lower'])
        meta=j(ROOT/f"runs/round75_density_measurement_v1/inference/{key[0]}/{key[1]}/identity/boundary.json")
        assert sha(ROOT/meta['data_path'])==meta['sha']
        items.append(dict(dataset=key[0],id=key[1],cohort=cohort,row=row,input=dst.relative_to(ROOT).as_posix(),input_sha256=sha(dst),
            cached_probability=meta['data_path'],cached_sha256=meta['sha'],frame=c['frame'],shape=list(c['image'].shape),
            image_only_tissue_box=candidates[key]['box'],crop_box=crop,crop_reference_retention=retained,reference_class=refclass,
            baseline_radius_px=2*max(c['image'].shape)/512,baseline_triage=stats[key]['sides']))
    model=ROOT/'runs/round62_augmentation_quality_v1/training/20261023/C/epoch_06_model.pt'
    train=j(ROOT/'runs/round62_augmentation_quality_v1/training_probe_summary.json')['metrics']['C']
    controls={k:dict(inputs=train[k]['inputs'],images=train[k]['images'],coverage_mean=train[k]['coverage_mean'],below80=train[k]['below80'],band_dice=train[k]['band_dice']) for k in ('sup','deep')}
    protocol=dict(date='2026-11-11',purpose='Investigate low source-boundary response; fixed model, no training, no decoder or default changes',
        training=False,subagents=False,new_inference='24 diagnostic inputs only; no full measurement tuning or submission',
        model=model.relative_to(ROOT).as_posix(),model_sha256=sha(model),selection='All12 prior weak-response images plus12 deterministic image-only matched healthy controls, within dataset and outside weak groups',
        feature_distance='L2 on log tissue aspect ratio, tissue width / original input longside, median/128, interdecile range/128; no target boundary geometry features',
        rows=items,pairs=pairs,
        arms=['F512_single','F512_TTA5','F768_single','M512_single','C512_single'],
        F512_single='Exact frozen input, single identity view at longside512',
        F512_TTA5='Exact frozen input, five native translations; verify against consumed cache',
        F768_single='Same pixels and crop, longside768; changes grid sampling and object scale, not a pure information test',
        M512_single='Same full input dimensions and tissue scale, zero pixels outside image-only tissue box +3% margin; context sensitivity test',
        C512_single='Crop to the same image-only margin box then longside512; context + tissue-scale intervention, not isolated from each other',
        fourth_channel='All zero in existing training AND inference; not an omitted trained C1 input',
        scoring='Same baseline-native radius (2 original512 grid units) across ALL arms, on original source-supported references. Report upper/lower role/union/fused continuous values, >.18 and >.5 coverage, both-side controls, GT retention. Never use GT in input cropping.',
        rescue='Descriptive flagged-side gain>=.10 fused>.18 coverage with opposite side coverage loss<=.02 and complete source GT retention; no optimization/promotion gate',
        hazards='Mask introduces artificial context, cropping changes sampling, upresolution is not training-distribution-equivalent. No generalization/causal training claim from sensitivity alone.',
        historical_training_probe=controls,
        protected=j(ROOT/'runs/round80_connection_fidelity_v1/protocol.json')['protected'],
        references_used_only_for_scoring=True,blind_validation=False,
        code_binding={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),ROOT/'scripts/round41_native.py',ROOT/'scripts/round57_finetune.py',ROOT/'scripts/round62_train.py')})
    write(OUT/'protocol.json',protocol);print('DIAGNOSTIC_PROTOCOL_FROZEN',len(items),len(pairs),sha(OUT/'protocol.json'),flush=True)


if __name__=='__main__':main()
