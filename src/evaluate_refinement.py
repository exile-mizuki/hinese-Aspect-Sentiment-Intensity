"""Freeze the system choice on development predictions, then evaluate the fixed comparison on test."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from main import O, inputs, truth
from bert_experiment import scored_predictions, write_json
from window_experiment import extract_scores
from refine_experiment import OUT, infer, load_model, tokenizer_for

SEEDS = (42, 17, 2026)
METRICS = ('Valance MAE','Valance PCC','Arousal MAE','Arousal PCC')


def baseline_folder(seed):
    return O/'bert_aspect' if seed==42 else O/'repeats'/f'seed_{seed}'/'bert_aspect'


def read_values(path, rows):
    values = extract_scores(pd.read_csv(path, sep=' ', keep_default_na=False), rows)
    if not np.isfinite(values).all() or not ((values>=1)&(values<=9)).all():
        raise ValueError(f'Invalid predictions: {path}')
    return values


def systems(base, new):
    return dict(baseline_single=base[0], baseline_ensemble=np.mean(base,axis=0),
                candidate_single=new[0], candidate_ensemble=np.mean(new,axis=0))


def main():
    decision=json.loads((OUT/'candidate_selection.json').read_text(encoding='utf-8'))
    variant=decision['selected_variant']
    dev=inputs('Validation'); dev_gold=truth('Validation')
    base=[]; new=[]; source_hashes={}; seed_metrics=[]
    for seed in SEEDS:
        for kind,folder,collector in [('baseline',baseline_folder(seed),base),
                                      ('candidate',OUT/variant/f'seed_{seed}',new)]:
            file=folder/'dev_predictions.txt'
            values=read_values(file,dev)
            collector.append(values)
            scores=scored_predictions(dev,values,dev_gold)[1]
            seed_metrics.append(dict(system=kind,seed=seed,split='dev',**scores))
            source_hashes[str(file.relative_to(O))]=hashlib.sha256(file.read_bytes()).hexdigest()
    scores_rows=[]
    for name,values in systems(base,new).items():
        pred,scores=scored_predictions(dev,values,dev_gold)
        (OUT/'systems'/name).mkdir(parents=True,exist_ok=True)
        pred.to_csv(OUT/'systems'/name/'dev_predictions.txt',sep=' ',index=False)
        scores_rows.append(dict(system=name,split='dev',mean_MAE=(scores['Valance MAE']+scores['Arousal MAE'])/2,**scores))
    selected=min(scores_rows,key=lambda r:(r['mean_MAE'],'ensemble' in r['system'],'candidate' in r['system']))['system']
    frozen=dict(selected_system=selected,selected_variant=variant,seeds=list(SEEDS),
                criterion='minimum development mean MAE across original/new single and 3-seed equal-weight ensemble',
                ensemble='mean of clipped, rounded-to-2-decimal member scores; then official output rounding',
                frozen_before_test=True,dev_results=scores_rows,development_sources_sha256=source_hashes,
                limitation='The old test set informed prior analysis. This is exploratory comparison, not a new untouched test set.')
    write_json(OUT/'final_selection.json',frozen)
    print('Frozen development choice:',selected,flush=True)
    # No Testing inputs or truth are read above this boundary.
    test=inputs('Testing'); gold=truth('Testing')
    tok=tokenizer_for(variant); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base=[]; new=[]
    for seed in SEEDS:
        base_values=read_values(baseline_folder(seed)/'test_predictions.txt',test)
        base.append(base_values)
        seed_metrics.append(dict(system='baseline',seed=seed,split='test',**scored_predictions(test,base_values,gold)[1]))
        model=load_model(variant,seed,device)
        raw=infer(model,test,variant,tok,device)
        pred,scores=scored_predictions(test,raw,gold)
        values=extract_scores(pred,test)
        new.append(values)
        folder=OUT/variant/f'seed_{seed}'
        pred.to_csv(folder/'test_predictions.txt',sep=' ',index=False)
        write_json(folder/'test_scores.json',scores)
        seed_metrics.append(dict(system='candidate',seed=seed,split='test',**scores))
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    cases=test.copy();cases[['true_V','true_A']]=extract_scores(gold,test)
    for name,values in systems(base,new).items():
        pred,scores=scored_predictions(test,values,gold)
        pred.to_csv(OUT/'systems'/name/'test_predictions.txt',sep=' ',index=False)
        write_json(OUT/'systems'/name/'test_scores.json',scores)
        cases[[name+'_V',name+'_A']]=extract_scores(pred,test)
        scores_rows.append(dict(system=name,split='test',mean_MAE=(scores['Valance MAE']+scores['Arousal MAE'])/2,**scores))
    pd.DataFrame(scores_rows).to_csv(OUT/'final_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(seed_metrics).to_csv(OUT/'per_seed_metrics.csv',index=False,encoding='utf-8-sig')
    cases.to_csv(OUT/'test_cases.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(scores_rows).to_string(index=False),flush=True)
    write_json(OUT/'evaluation_log.json',dict(selected_system=selected,test_rows=len(test),
               selection_sha256=hashlib.sha256((OUT/'final_selection.json').read_bytes()).hexdigest(),
               test_truth_sha256=hashlib.sha256(next((O.parent/'data').glob('*Testing*truth.txt')).read_bytes()).hexdigest(),
               evaluation='test evaluated after freezing dev choice; test not used to select the deployed system'))


if __name__=='__main__': main()
