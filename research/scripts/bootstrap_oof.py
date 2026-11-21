#!/usr/bin/env python3
"""Paired cluster bootstrap of EXISTING grouped OOF predictions; no new training."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def main(project,out,seed=20261120,B=3000):
    folder=project/'results'/'round2'
    b=pd.read_csv(folder/'marked_l1'/'oof_predictions.csv',keep_default_na=False)
    c=pd.read_csv(folder/'arousal_mse_aux'/'oof_predictions.csv',keep_default_na=False)
    manifest=pd.read_csv(folder/'fold_assignments.csv',keep_default_na=False)
    cols=['id','text','aspect'];assert b[cols].equals(c[cols]) and b[cols].equals(manifest[cols])
    assert set(manifest.fold)=={0,1,2}
    by=np.abs(np.round(np.clip(b[['pred_V','pred_A']].values,1,9),2)-b[['V','A']].values)
    cy=np.abs(np.round(np.clip(c[['pred_V','pred_A']].values,1,9),2)-c[['V','A']].values)
    gain=by-cy
    high=(b.A.values>=6.5)
    ids,inv=np.unique(manifest.group.values,return_inverse=True)
    ns=np.bincount(inv,minlength=len(ids))
    summed={
      'V_MAE':np.bincount(inv,weights=gain[:,0],minlength=len(ids)),
      'A_MAE':np.bincount(inv,weights=gain[:,1],minlength=len(ids)),
      'mean_MAE':np.bincount(inv,weights=gain.mean(axis=1),minlength=len(ids)),
      'high_A_MAE':np.bincount(inv,weights=gain[:,1]*high,minlength=len(ids))}
    den={'V_MAE':ns,'A_MAE':ns,'mean_MAE':ns,'high_A_MAE':np.bincount(inv,weights=high,minlength=len(ids))}
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(ids),size=(B,len(ids)))
    rows=[]
    for metric in summed:
        vals=summed[metric][samples].sum(axis=1)/den[metric][samples].sum(axis=1)
        observed=summed[metric].sum()/den[metric].sum()
        lo,hi=np.quantile(vals,[.025,.975])
        rows.append({'metric':metric,'baseline_minus_candidate_positive_is_better':observed,
                     'cluster_bootstrap_95CI_low':lo,'cluster_bootstrap_95CI_high':hi,
                     'n_review_connected_groups':len(ids),'n_resamples':B,
                     'evidence_type':'existing single-seed OOF paired group bootstrap; not independent multi-seed experiment'})
    outdir=out/'validation';outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir/'existing_oof_group_bootstrap.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    (outdir/'bootstrap_limits.json').write_text(json.dumps({
     'seed':seed,'iterations':B,'groups':len(ids),'validation_data':'saved OOF train predictions only',
     'interpretation':'Descriptive paired uncertainty; overlapping CV training sets and one training seed limit inference. No proof of significant model improvement.',
     'test_used':False},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--seed',type=int,default=20261120);p.add_argument('--bootstraps',type=int,default=3000)
    a=p.parse_args();main(a.project,a.out,a.seed,a.bootstraps)
