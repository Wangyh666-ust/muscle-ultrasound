"""Full replay only for admitted factors; failed controls never enter combinations."""
from scripts.round80_common import *
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from scripts.round77_common import load_context, NEW
from scripts.round77_layer import rank as prior_rank, same
from scripts.round80_core import connect, fair_rank, fidelity_curve
from scripts.round75_evaluate import ready, baseline_measure, slim
from scripts.round53_experiment import settings
from scripts.round76_core import adaptive
from scripts.round78_mt_core import measure_mt
from scripts.round64_measure_adapter import adapt
from scripts.round63_measure import original_pair
from scripts.round29_core import extended_fl
from scripts import validation112_measure as engine

# The original full experiment must remain replayable after the documented
# post-hoc safety guard. G diagnostics intentionally use the current bracket fix.
_initial_core=OUT/'implementation/round80_core_G0_before_bracket_fix.py'
if _initial_core.exists():
    import importlib.util
    _spec=importlib.util.spec_from_file_location('_round80_initial_core',_initial_core)
    _module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
    connect=_module.connect;fair_rank=_module.fair_rank


class Unchanged(Exception):pass


def replay(c,arm,lock,reuse=True):
    original_choose=engine.choose_pair;original_seams=engine.seams;original_keep=engine.keep;au={};ca=[]
    def choose(paths,width,cfg,segments,scale,c1,prior):
        _,pairs=original_choose(paths,width,cfg,segments,scale,c1,prior)
        baseline,old=prior_rank(pairs,paths,c['prob'],segments,width,c['domain'],None,bool(scale))
        selected,a=(fair_rank(pairs,paths,c['prob'],segments,width,c['domain'],bool(scale)) if 'R' in arm else (baseline,old))
        au.update(a,selected_ids=None if selected is None else [list(selected[s]['ids']) for s in ('upper','lower')])
        if reuse and 'C' not in arm and same(baseline,selected):raise Unchanged()
        return selected,pairs
    def seam(paths,field,gray,valid,width,scfg,bcfg):
        extra,a=original_seams(paths,field,gray,valid,width,scfg,bcfg)
        new,b=connect(paths,field,gray,valid,width,bcfg,lock['parameters']['C'],j(OUT/'protocol.json')['fixed_C'],extra);ca.append(b)
        return extra+new,a
    def keep(paths,extra,width):
        old=[p for p in extra if not any(e['type']=='R80_C' for e in p['events'])];new=[p for p in extra if any(e['type']=='R80_C' for e in p['events'])]
        return original_keep(paths,old,width)+new
    try:
        engine.choose_pair=choose
        if 'C' in arm:engine.seams=seam;engine.keep=keep
        pair,_,completion=engine.replay(c['prob'],c['image'],c['valid'],c['width'],c['mapped'],c['scale'],c['domain'])
        if pair:engine.qualify(pair,c['native_mask'],c['image'].shape)
        pair=original_pair(pair,dict(input_box=c['frame']['input_box'],matrix=c['matrix']))
        return pair,dict(ranking=au,C=ca,completion=completion),False
    except Unchanged:return None,dict(ranking=au,C=ca),True
    finally:engine.choose_pair=original_choose;engine.seams=original_seams;engine.keep=original_keep


def measured(pair,curve,c):
    row=c['row'];m=c['matrix'];l,t,r,b=c['frame']['scan_box'];roi=dict(box=[(l-m[0,2])/m[0,0],(t-m[1,2])/m[0,0],(r-m[0,2])/m[0,0],(b-m[1,2])/m[0,0]],fallback=c['frame']['fallback'],reason=c['frame']['reason'])
    cfg,cal=settings();policy=j(R77/'execution_lock.json')['A_policy'];outputs={};audits={}
    for seed in SEEDS:
        out=baseline_measure(curve,c['segments'][seed],c['counts'][seed]['raw'],roi,row);out,aa=adaptive(out,pair,row,cfg,cal,policy)
        mt=measure_mt(pair,curve,out['mt_positions'] or [],row['width'],row['height'],row['px_per_mm'],1/m[0,0],'complete','qualified')
        out['values']['MT']=mt['value'];out['MT_reasons']=mt['reasons'];out['stage']='output' if all(v is not None for v in out['values'].values()) else 'partial_or_missing'
        outputs[seed]=slim(out);audits[seed]=dict(adaptive=aa,MT=mt)
    return outputs,audits


def pair_error(a,b):
    if a is None or b is None:return 0. if a is None and b is None else float('inf')
    error=0.
    for side in ('upper','lower'):
        if not np.array_equal(a[side]['x'],b[side]['x']):return float('inf')
        error=max(error,float(np.max(abs(np.asarray(a[side]['y'])-np.asarray(b[side]['y'])))))
        for flag in ('observed','qualified'):
            if not np.array_equal(a[side][flag],b[side][flag]):return float('inf')
    return error


def g_values(base,pair,curve,c):
    g,audit=fidelity_curve(pair,curve,c['row']['width'],c['row']['height']);outputs=deepcopy(base);details={};scale=c['row']['px_per_mm'] or 1.
    for seed in SEEDS:
        weights=base[seed]['FL_weights'] or {};by={str(s['label_id']):s for s in c['segments'][seed]};tracks=[]
        for sid,w in weights.items():
            if w<=0:continue
            f=extended_fl(by[sid],g,c['row']['width'],c['row']['height']);px=f.get('fl_px_raw')
            tracks.append(dict(id=sid,weight=w,FL=None if px is None else px/scale,roots=f.get('roots'),failure=f.get('fl_failure_conditions')))
        complete=bool(tracks) and all(t['FL'] is not None for t in tracks)
        outputs[seed]['values']['FL']=sum(t['weight']*t['FL'] for t in tracks) if complete else None
        outputs[seed]['stage']='output' if all(v is not None for v in outputs[seed]['values'].values()) else 'partial_or_missing'
        details[seed]=tracks
        assert outputs[seed]['values']['MT']==base[seed]['values']['MT'] and outputs[seed]['values']['PA']==base[seed]['values']['PA']
    return outputs,g,dict(geometry=audit,tracks=details,weights_frozen=True,PA_MT_fixed=True)


def fixtures():
    ready();lock=j(OUT/'execution_lock.json');rows=[]
    for ds,sid in [('young78','ANON_66'),('young78','ANON_68'),('young78','ANON_78'),('external34','im_20_arch')]:
        old=readgz(R77/f'results/{ds}/{sid}/identity.json.gz');baseline=readgz(R78/f'results/{ds}/{sid}/identity.json.gz')
        c=load_context(old['row'],'identity');pair,_,_=replay(c,'B',lock,False);error=pair_error(pair,old['pairs']['P']);assert error<1e-7,(sid,error)
        fit,_=adapt(pair,c['row']['width'],c['row']['height']);actual,_=measured(pair,fit['curve'] if fit else None,c)
        maximum=0.
        for seed in SEEDS:
            for m in ('MT','PA','FL'):
                a,b=actual[seed]['values'][m],baseline['outputs'][seed]['M']['values'][m];assert (a is None)==(b is None)
                if a is not None:maximum=max(maximum,abs(a-b));assert abs(a-b)<1e-7,(sid,seed,m,a,b)
        rows.append(dict(id=sid,pair_error=error,scalar_error=maximum))
    write(OUT/'baseline_fixtures.json',dict(passed=True,cases=rows))


def work(row,view):
    ready();dst=OUT/f"results/{row['dataset']}/{row['image_id']}/{view}.json.gz"
    if dst.exists():return dict(id=row['image_id'],resumed=True)
    tic=time.monotonic();lock=j(OUT/'execution_lock.json');assert all(a=='B' or all(lock['admitted'][f] for f in a) for a in lock['arms'])
    source=readgz(R77/f"results/{row['dataset']}/{row['image_id']}/{view}.json.gz");mt=readgz(R78/f"results/{row['dataset']}/{row['image_id']}/{view}.json.gz")
    c=load_context(row,view);base={s:deepcopy(mt['outputs'][s]['M']) for s in SEEDS};outputs={'B':base};pairs={'B':source['pairs']['P']};curves={'B':source['curves']['P']};audits={};baseline_checks=[]
    for arm in lock['arms'][1:]:
        if 'G' in arm:
            parent=arm.replace('G','') or 'B';outputs[arm],curves[arm],audits[arm]=g_values(outputs[parent],pairs[parent],curves[parent],c);pairs[arm]=pairs[parent];continue
        pair,audit,reuse=replay(c,arm,lock)
        if reuse:outputs[arm]=deepcopy(base);pairs[arm]=pairs['B'];curves[arm]=curves['B'];audits[arm]=dict(**audit,unchanged=True);continue
        control,_,_=replay(c,'B',lock,False);err=pair_error(control,pairs['B']);baseline_checks.append(dict(arm=arm,error=err if np.isfinite(err) else None,passed=err<1e-7))
        if err>=1e-7:
            # A historical reconstruction mismatch is not an R/C effect; retain the sealed result and disclose it.
            outputs[arm]=deepcopy(base);pairs[arm]=pairs['B'];curves[arm]=curves['B'];audits[arm]=dict(**audit,not_applied='historical_baseline_geometry_or_qualification_mismatch');continue
        fit,adapter=adapt(pair,row['width'],row['height']);curve=fit['curve'] if fit else None;out,ma=measured(pair,curve,c)
        outputs[arm]=out;pairs[arm]=pair;curves[arm]=curve;audits[arm]=dict(**audit,measurement=ma,adapter=adapter,unchanged=False)
    out={s:{a:outputs[a][s] for a in lock['arms']} for s in SEEDS}
    write(dst,dict(row=row,view=view,frame=c['frame'],matrix=c['matrix'],outputs=out,pairs=pairs,curves=curves,audits=audits,baseline_checks=baseline_checks,
                   r77_source=f"runs/round77_subject_layer_v1/results/{row['dataset']}/{row['image_id']}/{view}.json.gz",source_sha=sha(R78/f"results/{row['dataset']}/{row['image_id']}/{view}.json.gz"),seconds=time.monotonic()-tic))
    return dict(id=row['image_id'],seconds=time.monotonic()-tic)


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--fixtures',action='store_true');args=ap.parse_args()
    if args.fixtures:fixtures();return
    assert j(OUT/'baseline_fixtures.json')['passed'];lock=j(OUT/'execution_lock.json');rows=j(R77/'protocol.json')['rows'];tasks=[];start=time.monotonic();done=0
    for row in rows:tasks.extend((row,v) for v in (('identity','translate2_-1','brightness105','scale1015',NEW) if row['reference'] else ('identity',)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,*task) for task in tasks]
        for f in as_completed(futures):
            f.result();done+=1
            if done%50==0 or done==len(tasks):print('R80_EVALUATE',done,len(tasks),round(time.monotonic()-start),flush=True)
    verify_default();write(OUT/'evaluation_done.json',dict(views=done,arms=lock['arms'],seeds=SEEDS,training=False,inference=False,default_changed=False,seconds=time.monotonic()-start))


if __name__=='__main__':main()
