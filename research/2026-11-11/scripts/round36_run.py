"""Frozen two-seed Round36 replay. Writes only Round36 outputs."""
import argparse, copy, gzip, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.round35_common import inputs,source_prediction,j,sha,binary,require,clean
from scripts.round35_io import write
from scripts.round27_experiment import check_equal
from src.umud_baseline.data import restore_original_coords
from scripts.round36_core import *

BASE=ROOT/'runs/round36_candidate_consensus_v1'
CFG=ROOT/'configs/round36_candidate_consensus.json'
OLD=ROOT/'runs/round35_boundary_continuity_v1'
ARMS=('D0','D1','M','C','P','F','CP','CF','CPF')


def save_gzip(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf8',compresslevel=5) as f:json.dump(clean(value),f,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)


def code_binding():
    # Snapshot current code, including the existing dirty workspace, rather than resetting it.
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs').glob('*.json'))+list((ROOT/'manifest').glob('round*.json'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths if not p.name.startswith('round36')}


def freeze():
    BASE.mkdir(parents=True,exist_ok=True);cfg=j(CFG)
    files={CFG.relative_to(ROOT).as_posix():sha(CFG),'scripts/round36_core.py':sha(ROOT/'scripts/round36_core.py'),'scripts/round36_run.py':sha(Path(__file__))}
    for seed in cfg['seeds']:
        src=source_prediction(seed,'T3');require(sha(src/'manifest.json')==j(src/'success.json')['manifest_sha256'],'source manifest hash')
        files[(src/'manifest.json').relative_to(ROOT).as_posix()]=sha(src/'manifest.json')
        for ds in cfg['datasets']:
            p=OLD/f'chain/{seed}/T3/{ds}/identity.json'
            require(sha(p)==p.with_suffix('.sha256').read_text().strip(),'retained identity chain hash')
            files[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel in ['runs/round34_boundary_guidance_v1/c1_guidance/manifest.json','runs/round26_1_direction_aux_corrected_v1/training/C1_corrected/checkpoints/epoch_017.pt']:
        files[rel]=sha(ROOT/rel)
    binding={'config':cfg,'files':files,'protected_code':code_binding(),'training':False,'new_inference':False,
             'evaluation_references_read_by_predictor':False,'candidate_scope':'existing D0/D1 pairs; no new cross-side pair synthesis',
             'prior_scope':'hypothesis sensitivity only; external34 only has physical scales; no GT fitted',
             'datasets_per_seed':469,'total_image_seed_records':938}
    path=BASE/'binding.json'
    if path.exists():require(j(path)==binding,'Round36 binding changed')
    else:write(path,binding)
    return binding


def run(seed=None,limit=None):
    binding=freeze();cfg=binding['config'];ins={(r['dataset'],r['image_id']):r for r in inputs()};files={};count=0
    for sd in ([seed] if seed else cfg['seeds']):
        src=source_prediction(sd,'T3');pred={(p['dataset'],p['image_id']):p for p in j(src/'manifest.json')['rows'] if p['transform']['name']=='identity'}
        for ds,n in cfg['datasets'].items():
            dest=BASE/f'predictions/{sd}/{ds}.json.gz';stamp=dest.with_suffix('.sha256')
            if dest.exists() and stamp.exists() and limit is None:
                require(sha(dest)==stamp.read_text().strip(),'resume output hash');print('EXISTS',sd,ds,flush=True);continue
            rows=j(OLD/f'chain/{sd}/T3/{ds}/identity.json');require(len(rows)==n,'cohort denominator')
            expected={iid for dataset,iid in ins if dataset==ds};require({r['image_id'] for r in rows}==expected,'cohort identities')
            out=[];tic=time.time()
            for ri,row in enumerate(rows[:limit] if limit else rows):
                iid=row['image_id'];r=ins[ds,iid];pinfo=pred[ds,iid]
                for key in ('width','height','group_id','pixel_sha256'):require(row[key]==r[key],'input '+key)
                require(sha(ROOT/r['mask_path'])==r['mask_sha256'],'C1 mask hash')
                pp=src/pinfo['path'];require(sha(pp)==pinfo['sha256'],'T3 probability hash')
                with np.load(pp,allow_pickle=False) as z:
                    small=z['prob'];meta=json.loads(str(z['meta']))
                    probs=np.stack([restore_original_coords(q,meta['original_wh'],meta['scale'],meta['pad'],'nearest') for q in small])
                    require(np.array_equal(probs>.5,z['masks']),'probability exact replay')
                segments=[c['segment'] for c in row['arms']['D1']['candidates']]
                check_equal(segments,[c['segment'] for c in row['arms']['D0']['candidates']],'D0/D1 same frozen C1')
                # Replay numeric geometry and candidate selection before introducing new controls.
                for d in ('D0','D1'):
                    replay,_=measure(row['arms'][d]['curve'],segments,row,row['arms'][d]['selected_pair_id'])
                    for k in ('values','selection','PA_ids','PA_invalid','candidates'):check_equal(replay[k],row['arms'][d][k],'Round35 '+d+' '+k)
                ev=prepare_evidence(binary(ROOT/r['mask_path']),segments,row['width'],row['height'],cfg['c1'])
                pool=merge_pools(row,cfg['merge']);scores=candidate_scores(pool,ev,probs,row,cfg)
                arms={d:copy.deepcopy(row['arms'][d]) for d in ('D0','D1')};measurement_cache={}
                for arm,c1,prior in [('M',False,False),('C',True,False),('P',False,True),('CP',True,True)]:
                    p=choose(pool,scores,row,c1,prior);pid=p['pair_id'] if p else None
                    if pid not in measurement_cache:measurement_cache[pid]=measure(p['curve'] if p else None,segments,row,pid)[0]
                    arms[arm]=copy.deepcopy(measurement_cache[pid])
                    arms[arm]['selection_reason']='C1_gate_rejected_all' if p is None and pool and c1 else 'no_valid_pair' if p is None else 'ranked'
                    arms[arm]['prior_requested']=prior;arms[arm]['prior_applied']=bool(prior and row.get('px_per_mm') and pool)
                    arms[arm]['prior_status']=cfg['mt_prior']['status'] if prior and row.get('px_per_mm') else 'not_requested' if not prior else 'no_physical_scale'
                for target,source in [('F','M'),('CF','C'),('CPF','CP')]:arms[target]=consensus_measure(arms[source],segments,row,cfg)
                for a in arms.values():a['curve']=strip_curve(a['curve'])
                record={k:row[k] for k in ('dataset','image_id','group_id','width','height','image_path','pixel_sha256','seed','px_per_mm','component_counts')}
                record.update(arms=arms,pool=pool,scores=scores,evidence={k:ev[k] for k in ('all_count','independent_count','reliable_count')},
                              input_sha={'probability':pinfo['sha256'],'C1':r['mask_sha256']},baseline_replayed=True)
                if not row.get('px_per_mm'):
                    for a,b in [('M','P'),('C','CP'),('CF','CPF')]:check_equal(arms[a]['values'],arms[b]['values'],'no scale no prior effect')
                for a,b in [('M','F'),('C','CF'),('CP','CPF')]:
                    check_equal(arms[a]['curve'],arms[b]['curve'],'direction cannot change boundary')
                    check_equal(arms[a]['values']['MT'],arms[b]['values']['MT'],'direction cannot change MT')
                out.append(record);count+=1
                if (ri+1)%10==0:print('PROGRESS',sd,ds,ri+1,'/',n,'seconds',round(time.time()-tic),flush=True)
            if limit:dest=BASE/f'smoke/{sd}/{ds}.json.gz'
            save_gzip(dest,out)
            if not limit:stamp.write_text(sha(dest),encoding='ascii')
            files[dest.relative_to(BASE).as_posix()]=sha(dest);print('SAVED',sd,ds,len(out),flush=True)
    require(code_binding()==binding['protected_code'],'default code changed')
    if limit:return
    allfiles={}
    for sd in cfg['seeds']:
        for ds,n in cfg['datasets'].items():
            p=BASE/f'predictions/{sd}/{ds}.json.gz'
            if not p.exists():return
            require(sha(p)==p.with_suffix('.sha256').read_text().strip(),'output integrity');allfiles[p.relative_to(BASE).as_posix()]=sha(p)
    write(BASE/'predictions/success.json',{'files':allfiles,'binding_sha256':sha(BASE/'binding.json'),'baseline_replays':938*2,'image_seed_records':938,'arms':list(ARMS),'reference_read':False})
    print('PREDICTIONS_COMPLETE 938 images x 9 arms',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--limit',type=int);a=p.parse_args();run(a.seed,a.limit)
