"""Frozen selected S layers, independent local repairs, unchanged formal measurement."""
from copy import deepcopy
import time
from scripts.round65_common import ROOT,R63,OUT as OLD,np,j,sha,write,readgz,array_sha,Path,SIDES
from scripts.round63_paths import compact_pair,compact_path,image_evidence
from scripts.round63_measure import reference_score,original_pair
from scripts.round67_geometry import process_path

OUT=ROOT/'runs/round67_local_evidence_v1'


def destination(rec,pilot=False):
    return OUT/('case_records' if pilot else 'records')/rec['dataset']/Path(rec['image_id']).stem/(rec['view']+'.json.gz')


def check_pair(reference,paths):
    if reference is None:return None,False
    pair=deepcopy(reference)
    for side,p in zip(SIDES,paths):pair[side]=p
    lo=max(p['x'][0] for p in paths);hi=min(p['x'][-1] for p in paths)
    if hi<=lo:return reference,True
    x=np.linspace(lo,hi,max(101,int(hi-lo)+1))
    if np.any(np.interp(x,paths[0]['x'],paths[0]['y'])>=np.interp(x,paths[1]['x'],paths[1]['y'])):
        return reference,True
    pair['interval']=[float(lo),float(hi)];pair['layer_selection_frozen']=True
    return pair,False


def work(rec,probability,pilot=False):
    from scripts.round65_measure import measure
    tic=time.monotonic();lock=j(OUT/'parameter_lock.json');dest=destination(rec,pilot)
    if dest.exists():
        old=readgz(dest);assert old['probability_sha']==array_sha(probability) and old['lock_sha']==sha(OUT/'parameter_lock.json')
        return dict(key=rec['key'],view=rec['view'],resumed=True)
    assert sha(ROOT/rec['path'])==rec['sha256']
    historical=readgz(R63/'records'/rec['dataset']/Path(rec['image_id']).stem/(rec['view']+'.json.gz'))
    assert array_sha(probability)==historical['probability_sha256']
    original=readgz(OLD/'records'/rec['dataset']/Path(rec['image_id']).stem/(rec['view']+'.json.gz'))
    assert original['probability_sha']==array_sha(probability)
    baseline=original['arms']['S'];reference=deepcopy(baseline['pair']);cfg=lock['config'];shape=lock['shape_config'];base=j(R63/'config.json')
    if reference is not None:
        for side in SIDES:
            for axis in ('x','y','kind','observed'):reference[side][axis]=np.asarray(reference[side][axis])
    with np.load(ROOT/rec['path']) as z:
        image=z['image'];valid=z['evidence'].astype(bool);width=float(z['reference_width']);gray=image_evidence(image,width)
        results=[]
        if reference is not None:
            for i,side in enumerate(SIDES):
                src=deepcopy(reference[side]);src['x']=np.asarray(src['x']);src['y']=np.asarray(src['y'])
                src['kind']=np.asarray(src['kind']);src['observed']=np.asarray(src['observed'],bool)
                field=probability[i+1]*(.4+.6*probability[0])
                effective_gain=lock['selected_gain'] if lock['repair_enabled'] else float('inf')
                result=process_path(src,field,gray,valid,width,cfg,shape,base,effective_gain)
                if not lock['repair_enabled']:
                    result['repaired']=src;result['audit']['global_repair_disabled']=True
                results.append(result)
        pairs={'S':reference};fallback={}
        if reference is None:pairs.update(R67=None,RX67=None)
        else:
            for arm,key in [('R67','repaired'),('RX67','extended')]:
                pairs[arm],fallback[arm]=check_pair(reference,[r[key] for r in results])
        row={k:original[k] for k in ('key','dataset','image_id','view','manual','partition','group','input','width','shape')}
        row.update(probability_sha=array_sha(probability),lock_sha=sha(OUT/'parameter_lock.json'),
            baseline_record_sha=sha(OLD/'records'/rec['dataset']/Path(rec['image_id']).stem/(rec['view']+'.json.gz')),
            arms={},audit={side:results[i]['audit'] for i,side in enumerate(SIDES)} if results else dict(reason='no_selected_baseline_pair'),
            ordering_fallback=fallback,estimates={side:[compact_path(p) for p in results[i]['estimates']] for i,side in enumerate(SIDES)} if results else {})
        for arm,pair in pairs.items():
            if arm=='S':out=deepcopy(baseline)
            else:
                measured,ma=measure(original_pair(pair,rec),rec['key'])
                out=dict(pair=compact_pair(pair),formal=measured,measurement_audit=ma)
                if rec['manual']:out['reference']=reference_score(pair,z)
            row['arms'][arm]=out
        if rec['manual']:row['referenceLines']=original['referenceLines']
    row['seconds']=time.monotonic()-tic;write(dest,row)
    return dict(key=rec['key'],view=rec['view'],seconds=row['seconds'],resumed=False)
