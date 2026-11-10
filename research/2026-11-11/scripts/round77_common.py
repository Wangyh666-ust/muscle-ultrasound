"""R77 isolated registration and cache context. Never edits frozen/default sources."""
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import numpy as np
from PIL import Image
from scripts.round75_common import ROOT,OUT as R75,j,write,sha,readgz,records
OUT=ROOT/'runs/round77_subject_layer_v1'
R76=ROOT/'runs/round76_adaptive_density_prior_v1'
VIEWS=('identity','translate2_-1','brightness105','scale1015')
NEW='combo098_translate-1_1'
SEEDS=('20261023','20261024')

def register():
    if (OUT/'protocol.json').exists():return
    old=j(R75/'protocol.json');rows=old['rows'];fr={r['key']:r for r in j(ROOT/'runs/frozen_boundary_20261110/manifest.json')['rows'] if r['view']=='identity'}
    natural=readgz(ROOT/'runs/round47_boundary_selection_v1/evaluation/natural_pairs.json.gz');parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for r in natural:union(r['a'],r['b'])
    controls=[]
    for r in rows:
        key=r['dataset']+'|'+r['image_id']
        if r['dataset']!='apo153' or not fr[key]['manual']:continue
        group=find(key);phase='confirmation' if int(sha256(group.encode()).hexdigest()[:8],16)%3==0 else 'calibration'
        controls.append(dict(key=key,id=r['image_id'],group=group,phase=phase))
    assert len(controls)==143
    grids=[dict(min_role=p,min_contrast=c,min_span=s,dominance=.60,dominance_gap=.15,dedup_roi=.006,match_roi=.015)
           for p in (.55,.70) for c in (0.,.15) for s in (.12,.20)]
    write(OUT/'protocol.json',dict(round=77,status='registered_before_controls',training=False,default_changed=False,subagents=False,rows=rows,
        models=old['models'],protected={**old['protected'], 'configs/boundary_frozen_20261110.json':sha(ROOT/'configs/boundary_frozen_20261110.json')},
        prior=dict(maximum_penalty=.4,shoulder_mm=2.,outside=[10.,31.],mode='additional_to_old_prior_not_replacement'),
        L_grids=grids,controls=controls,control_groups='natural-similarity connected components kept together; subject identity unconfirmed',
        controls_are='real143 expert boundary-reference images, calibration first; synthetic functional checks only supplementary',
        gates=dict(L='at least one wrong-layer recovery on calibration and confirmation; no loss of baseline correct sides/pairs; no global support or boundary MAE degradation',
                   P='exact unchanged interval scores; smooth finite outside penalties; no scale guessed; baseline geometry parity'),
        baseline='R76 BA original formal adapter plus locked adaptive contributor policy, C1 B23 chooses boundary, B23/B24 measure; no Q substitution',
        arms=['B','P','L_if_admitted','PL_if_admitted'],views=VIEWS,extra_confirmation_view=NEW,
        extra_transform='brightness0.98 followed by native translation(-1,+1), never used to choose parameters',
        original_scope=547,scalar_reference_scope=112,new_confirmation_scope=112,seeds=SEEDS,boundary_seed=20261023,
        primary='same-cohort Kaggle formula/errors, missing/recovered, boundary upper/lower, layer switches, FL severe jumps, 117 natural pairs and39 reference-change pairs',
        independence='all historical data reused; this is regression, not a new blind or clinical validation',
        no_GT_in_ranking=True,failure='failed L does not enter full arms or combination, report failure and retain diagnostics',
        source_plan_sha=sha(ROOT/'thinking/ROUND77_LAYER_SELECTION_PLAN_20261110.txt')))
    print('R77_REGISTERED',len(controls),'controls',flush=True)

def sourcefile(row,view):return R75/f"results/{row['dataset']}/{row['image_id']}/{view}.json.gz"

def load_context(row,view):
    from scripts.round75_evaluate import ready,load_infer,view_gray
    from scripts.round46_frame import input_crop
    from scripts.round71_roi import locate as tissue_locate
    ready()
    if view==NEW:
        p=OUT/f"inference/{row['dataset']}/{row['image_id']}/{view}/boundary.json";meta=j(p)
        with np.load(ROOT/meta['data_path']) as z:prob=z['probability'].copy()
    else:prob,meta=load_infer(row,view,'boundary')
    matrix=np.asarray(meta['matrix']);frame=meta['frame'];gray=np.array(Image.open(ROOT/row['image_path']).convert('L'))
    if view==NEW:
        import cv2
        brightness=np.rint(gray.astype(float)*.98).clip(0,255).astype(np.uint8)
        viewim=cv2.warpAffine(brightness,matrix,(row['width'],row['height']),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    else:viewim=view_gray(gray,view,matrix)
    image=input_crop(np.repeat(viewim[:,:,None],3,2),frame)[:,:,0];mask,dm=tissue_locate(viewim)
    native_mask=input_crop(np.repeat(mask.astype(np.uint8)[:,:,None],3,2),frame)[:,:,0].astype(bool)
    yy,xx=np.where(native_mask);domain=[float(xx.min()),float(xx.max())] if len(xx) else [0.,float(image.shape[1]-1)]
    ev=np.zeros_like(viewim,np.uint8);l,t,r,b=frame['evidence_box'];ev[t:b,l:r]=1
    valid=input_crop(np.repeat(ev[:,:,None],3,2),frame)[:,:,0].astype(bool)
    if view!=NEW:
        source=readgz(sourcefile(row,view));segments=source['segments'];counts=source['counts']
    else:
        from scripts.round48_length_replay import extract
        from scripts.round53_experiment import settings
        cfg,_=settings();segments={};counts={};a,b,c,d=frame['input_box']
        roi=dict(box=frame['scan_box'],fallback=frame['fallback'],reason=frame['reason'])
        for seed in SEEDS:
            m=j(OUT/f"inference/{row['dataset']}/{row['image_id']}/{view}/B_{seed}.json")
            with np.load(ROOT/m['data_path']) as z:pr=z['probability'][0]
            full=np.zeros(gray.shape,bool);l,t,r,bt=max(0,a),max(0,b),min(row['width'],c),min(row['height'],d);full[t:bt,l:r]=pr[t-b:bt-b,l-a:r-a]>.5
            evidence=np.zeros_like(full);l,t,r,bt=frame['evidence_box'];evidence[t:bt,l:r]=full[t:bt,l:r]
            raw,kept,_,_,_=extract(evidence,roi,cfg)
            for s in kept:
                for k in ('p0','p1','centroid'):s[k]=(np.asarray(s[k])-matrix[:,2])/matrix[0,0]
            segments[seed]=kept;counts[seed]=dict(raw=len(raw),kept=len(kept))
        source=None
    mapped=[];box=np.asarray(frame['input_box'][:2])
    for s in segments['20261023']:
        q=deepcopy(s)
        for k in ('p0','p1','centroid'):q[k]=matrix[:,:2]@np.asarray(q[k])+matrix[:,2]-box
        mapped.append(q)
    return dict(row=row,view=view,prob=prob,frame=frame,matrix=matrix,image=image,mask=mask,native_mask=native_mask,valid=valid,domain=domain,
                source=source,segments=segments,counts=counts,mapped=mapped,width=frame['reference_width'],scale=row['px_per_mm']*matrix[0,0] if row['px_per_mm'] else None)
