"""Recalculate the Figure 11 weight grid, with a checkpoint for each ion pair.

Run from the archive root: python weight_validation/run_scan.py
Completed results are written to output/data. Interrupted runs resume by default.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from core.data import available_cases, load_pair_from_h5, load_cfg_model, paper_parameters
from core.similarity import compute_xcos_parts
from core.matching import paper_rules, threshold_results, acceptable_ranges

MODEL_CACHE = {}


def input_fingerprints(cases):
    """Identify the retained fields used in matching, models and vocabularies."""
    result = {}
    for seq, ion, db1, db2 in cases:
        with h5py.File(ROOT/'clean_data'/f'{ion}.h5', 'r') as f:
            for db in [db1, db2]:
                key=f'{ion}/lev_{db}'
                if key in result:
                    continue
                digest=hashlib.sha256()
                for field in ['idx','cfg','2S+1','L','J','pi','eV']:
                    a=f['lev_'+db][field]
                    digest.update(field.encode());digest.update(str(a.dtype).encode());digest.update(a.tobytes())
                result[key]=digest.hexdigest()
        for name in ['cfg_ts_model.pth','cfg_vocab.pkl']:
            key=f'models/{seq}/{name}'
            if key not in result:
                result[key]=hashlib.sha256((ROOT/key).read_bytes()).hexdigest()
    return result


def weight_grid(step):
    if not np.isfinite(step) or step <= 0:
        raise ValueError('The grid step must be positive and finite.')
    n=round(1/step)
    if n<1 or not np.isclose(n*step,1):
        raise ValueError('The grid step must divide 1 exactly.')
    rows=[]
    for i in range(n+1):
        for j in range(n+1-i):
            weights=[i/n,j/n,(n-i-j)/n]
            rows.append(dict(weight_id=f'grid_{i}_{j}_{n-i-j}',w_cfg=weights[0],w_term=weights[1],w_energy=weights[2],is_default=False))
    adopted=paper_parameters()['weights']
    rows.append(dict(weight_id='default',w_cfg=adopted[0],w_term=adopted[1],w_energy=adopted[2],is_default=True))
    return rows


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def run_case(task):
    import torch
    torch.set_num_threads(1)
    seq,ion,db1,db2=task['case']
    checkpoint=Path(task['checkpoint'])
    if checkpoint.exists():
        try:
            previous=json.loads(checkpoint.read_text())
        except (ValueError, OSError):
            previous={}
        valid_rows = hashlib.sha256(json.dumps(previous.get('rows',[]),sort_keys=True).encode()).hexdigest() == previous.get('rows_sha256')
        if valid_rows and previous.get('signature')==task['signature'] and len(previous.get('rows',[]))==len(task['weights']):
            return previous['rows']
    if seq not in MODEL_CACHE:
        MODEL_CACHE[seq]=load_cfg_model(seq)
    pair=load_pair_from_h5(ion,db1,db2,seq=seq,cfg_model=MODEL_CACHE[seq])
    _,parts=compute_xcos_parts(pair,keep=('cfg','term','energy'),cfg_weight=1.,term_weight=1.,energy_weight=1.)
    parts={key:np.nan_to_num(np.asarray(parts[key],dtype=np.float32),nan=-1e6,posinf=1.,neginf=-1e6)
           for key in ['cfg_cos','term_score','energy_cos']}
    rules=paper_rules(pair,seq);theoretical=int(rules['theoretical_overlap'])
    rows=[]
    for spec in task['weights']:
        wc,wt,we=spec['w_cfg'],spec['w_term'],spec['w_energy']
        sim=wc*parts['cfg_cos']+wt*parts['term_score']+we*parts['energy_cos']
        sim=np.nan_to_num(sim,nan=-1e6,posinf=1.,neginf=-1e6).astype(np.float32,copy=False)
        frame=threshold_results(sim,rules,task['thresholds'])
        ranges=acceptable_ranges(frame,theoretical)
        rows.append(dict(ion_type=seq,ion=ion,db1=db1,db2=db2,**spec,theoretical=theoretical,success=bool(ranges),success_threshold_ranges=ranges))
    atomic_json(checkpoint,dict(signature=task['signature'],rows=rows,rows_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--grid-step',type=float,default=.1)
    parser.add_argument('--output-dir',type=Path,default=HERE/'output')
    parser.add_argument('--max-cases',type=int,default=None,help='Run a small check; not the full paper experiment.')
    parser.add_argument('--keep-checkpoints',action='store_true')
    args=parser.parse_args()
    if args.workers<1 or (args.max_cases is not None and args.max_cases<1):
        raise ValueError('workers and max-cases must be positive')
    cases=list(available_cases())
    if args.max_cases is not None:
        cases=cases[:args.max_cases]
    specs=weight_grid(args.grid_step)
    settings=paper_parameters();grid=settings['threshold_grid']
    thresholds=np.round(np.arange(grid['start'],grid['stop']+grid['step']/2,grid['step']),6).tolist()
    inputs=input_fingerprints(cases)
    code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'core').glob('*.py'))}
    code['weight_validation/run_scan.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config=dict(settings=settings,input_fingerprints=inputs,code_fingerprints=code,weights=specs,thresholds=thresholds,cases=cases)
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    out=args.output_dir.resolve();data=out/'data';check=out/'checkpoints'
    data.mkdir(parents=True,exist_ok=True);check.mkdir(parents=True,exist_ok=True)
    record=dict(status='running',run_signature=signature,n_cases_requested=len(cases),n_grid_specs=len(specs),configuration=config)
    atomic_json(data/'run_config.json',record)
    tasks=[dict(case=case,weights=specs,thresholds=thresholds,signature=signature,checkpoint=str(check/('_'.join(case)+'.json'))) for case in cases]
    all_rows=[]
    try:
        if args.workers==1:
            for number,task in enumerate(tasks,1):
                all_rows.extend(run_case(task));print(f'{number}/{len(tasks)} cases completed',flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(run_case,task):task for task in tasks}
                for number,future in enumerate(as_completed(futures),1):
                    all_rows.extend(future.result());print(f'{number}/{len(tasks)} cases completed',flush=True)
        if input_fingerprints(cases)!=inputs:
            raise RuntimeError('Input data changed during the scan; checkpoints retained.')
        if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in code.items()):
            raise RuntimeError('Calculation code changed during the scan; checkpoints retained.')
        details=pd.DataFrame(all_rows).sort_values(['ion_type','ion','db1','db2','weight_id'])
        keys=['weight_id','w_cfg','w_term','w_energy','is_default']
        summary=details.groupby(keys,sort=False,dropna=False).agg(n_eval=('success','size'),n_success=('success','sum')).reset_index()
        if len(details)!=len(cases)*len(specs) or not (summary.n_eval==len(cases)).all():
            raise RuntimeError('Incomplete weight scan; checkpoints retained.')
        details.to_csv(data/'weight_grid_case_details.csv',index=False)
        summary.to_csv(data/'weight_grid_results.csv',index=False)
        pd.DataFrame(cases,columns=['ion_type','ion','db1','db2']).to_csv(data/'case_list.csv',index=False)
        record.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),paper_run=(len(cases)==314 and np.isclose(args.grid_step,.1)),data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*.csv')})
        atomic_json(data/'run_config.json',record)
        from plot_fig11 import plot_ternary
        import matplotlib.pyplot as plt
        with plt.rc_context():
            plt.rcdefaults();plot_ternary(summary,out/'figures/fig11_weight_sensitivity.pdf')
        if not args.keep_checkpoints:
            for task in tasks:
                Path(task['checkpoint']).unlink(missing_ok=True)
            if not any(check.iterdir()):check.rmdir()
        print(out,flush=True)
    except BaseException:
        record['status']='incomplete';atomic_json(data/'run_config.json',record)
        raise


if __name__=='__main__':
    main()
