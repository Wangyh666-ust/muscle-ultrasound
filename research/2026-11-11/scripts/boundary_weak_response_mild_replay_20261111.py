"""Read existing five-condition caches for all78 source-referenced external images."""
from scripts.round80_recognition_triage import ROOT,j,write,sha,np
from scripts.audit_boundary_connection_20261110 import references
from scipy.ndimage import map_coordinates
import gzip,json,time

OUT=ROOT/'review/boundary_weak_response_investigation_20261111'
VIEWS=('identity','translate2_-1','brightness105','scale1015','combo098_translate-1_1')


def near(field,points,radius):
    n=int(np.ceil(radius/.5));offset=np.arange(-n,n+1)*.5
    offset=np.unique(np.r_[offset[abs(offset)<=radius+1e-8],-radius,radius])
    return np.max([map_coordinates(field,[points[:,1]+dy,points[:,0]],order=1,mode='constant',cval=0.) for dy in offset],axis=0)


def stat(a):return dict(median=float(np.median(a)),mean=float(np.mean(a)),over018=float((a>=.18).mean()),over05=float((a>=.5).mean()))


def main():
    proto=j(OUT/'protocol.json');rows=[r for r in j(ROOT/'runs/round77_subject_layer_v1/protocol.json')['rows'] if r['dataset']=='young78'];assert len(rows)==78
    plan=dict(scope='All78 expert source-stroke reference images x5 existing cache conditions; references never feed predictions',views=VIEWS,
        training=False,new_inference=False,scoring='Reference native coordinates transformed by bound matrix, original2grid tolerance scaled by transform, head/union/opposite/fused separately',
        model_sha256=proto['model_sha256'],causal_limit='Small-transform replay, not held-out validation. Physical target reference is preserved rather than recomputed from image. Annotation and tissue-target semantics require expert review.')
    write(OUT/'mild_replay_protocol.json',plan);t0=time.monotonic();results=[]
    weak={r['id'] for r in proto['rows'] if r['dataset']=='young78' and r['cohort']=='weak'}
    for n,row in enumerate(rows):
        gt,_=references(row);base_meta=j(ROOT/f"runs/round75_density_measurement_v1/inference/young78/{row['image_id']}/identity/boundary.json")
        a,b,c,d=base_meta['frame']['input_box'];radius=2*max(d-b,c-a)/512
        for view in VIEWS:
            source='round77_subject_layer_v1' if view.startswith('combo') else 'round75_density_measurement_v1'
            meta=j(ROOT/f"runs/{source}/inference/young78/{row['image_id']}/{view}/boundary.json")
            path=ROOT/meta['data_path'];assert sha(path)==meta['sha']
            if meta.get('model_sha'):assert meta['model_sha']==proto['model_sha256']
            elif path.with_suffix('.json').exists():
                binding=j(path.with_suffix('.json'));assert binding['model_sha']==proto['model_sha256']
            else:raise AssertionError(('missing model binding',path))
            with np.load(path) as z:prob=z['probability']
            matrix=np.asarray(meta['matrix']);box=meta['frame']['input_box'];H,W=prob.shape[-2:];sides={}
            for i,side in enumerate(('upper','lower')):
                mapped=gt[side]@matrix[:,:2].T+matrix[:,2]-np.asarray(box[:2]);r=radius*matrix[0,0]
                retained=(mapped[:,0]>=0)&(mapped[:,0]<W)&(mapped[:,1]>=0)&(mapped[:,1]<H)
                fields=dict(role=prob[i+1],opposite=prob[2-i],union=prob[0],fused=prob[i+1]*(.4+.6*prob[0]))
                profiles={k:near(field,mapped,r) for k,field in fields.items()}
                missing=profiles['fused']<.18
                sides[side]=dict(metrics={k:stat(v) for k,v in profiles.items()},columns=len(mapped),retained=float(retained.mean()),
                    raw_union_and_role_both_weak=int((missing&(profiles['union']<.18)&(profiles['role']<.18)).sum()),
                    role_weak_union_strong=int((missing&(profiles['role']<.18)&(profiles['union']>=.5)).sum()),
                    role_weak_opposite_strong=int((missing&(profiles['role']<.18)&(profiles['opposite']>=.5)).sum()),
                    missing_columns=int(missing.sum()))
            results.append(dict(id=row['image_id'],group=row['group'],view=view,cohort='previous_weak' if row['image_id'] in weak else 'other69',
                sides=sides,cache_sha256=meta['sha'],model_sha256=proto['model_sha256']))
        if (n+1)%15==0:print('MILD_CACHE_REPLAY',n+1,78,round(time.monotonic()-t0,2),flush=True)
    groups={}
    for cohort in ('previous_weak','other69','all78'):
        rr=[r for r in results if cohort=='all78' or r['cohort']==cohort];groups[cohort]={}
        for side in ('upper','lower'):
            values={r['image_id']:[] for r in rows if cohort=='all78' or (r['image_id'] in weak)==(cohort=='previous_weak')}
            for r in rr:values[r['id']].append(r['sides'][side]['metrics']['fused']['over018'])
            groups[cohort][side]=dict(mean_coverage_by_view={v:float(np.mean([r['sides'][side]['metrics']['fused']['over018'] for r in rr if r['view']==v])) for v in VIEWS},
                image_range_median=float(np.median([np.ptp(x) for x in values.values()])),image_range_over10pp=[k for k,v in values.items() if np.ptp(v)>.10],
                low_in_all5=[k for k,v in values.items() if max(v)<.8])
    with gzip.open(OUT/'mild_replay_records.json.gz','wt',encoding='utf-8') as f:json.dump(results,f,ensure_ascii=False)
    write(OUT/'mild_replay_summary.json',dict(images=78,conditions=390,source_reference_retention_min=min(r['sides'][s]['retained'] for r in results for s in ('upper','lower')),
        groups=groups,seconds=time.monotonic()-t0,training=False,new_inference=False,default_changed=False))
    print('MILD_REPLAY_DONE',390,flush=True)


if __name__=='__main__':main()
