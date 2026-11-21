#!/usr/bin/env python3
"""Training-fold OOF fixed-weight ensemble, diagnostics and GROUP bootstrap.

IMPORTANT: Fixed 0.5/0.5 *exploratory* ensemble of two historical OOF models.
This script does not fit or tune a blender, load official dev/test, or assert
that the original 3-epoch OOF base equals the 2-epoch deployed model.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=('marked_l1','arousal_mse_aux')

def metrics(y,p):
    p=np.round(np.clip(np.asarray(p,dtype=float),1,9),2)
    e=np.abs(y-p)
    high=y[:,1]>=6.5
    ans={'n':int(len(y)), 'V_MAE':float(e[:,0].mean()),
         'A_MAE':float(e[:,1].mean()),'mean_MAE':float(e.mean()),
         'high_A_n':int(high.sum()),
         'high_A_MAE':float(e[high,1].mean()) if high.any() else None}
    for j,key in enumerate(['V_PCC','A_PCC']):
        x,z=y[:,j],p[:,j]
        ans[key]=float(np.corrcoef(x,z)[0,1]) if np.std(x)>0 and np.std(z)>0 else None
    return ans

def bootstrap(y,base,candidate,group,draws=3000,seed=20261120):
    rng=np.random.default_rng(seed)
    _,codes=np.unique(group,return_inverse=True)
    n=int(codes.max())+1
    bygroup=[np.where(codes==i)[0] for i in range(n)]
    rows=[]
    high=y[:,1]>=6.5
    for label,mask,dim in [('V_MAE',np.ones(len(y),dtype=bool),0),
                           ('A_MAE',np.ones(len(y),dtype=bool),1),
                           ('mean_MAE',np.ones(len(y),dtype=bool),None),
                           ('high_A_MAE',high,1)]:
        olderr=np.abs(np.round(np.clip(base,1,9),2)-y)
        newerr=np.abs(np.round(np.clip(candidate,1,9),2)-y)
        delta=olderr-newerr
        if dim is not None:delta=delta[:,dim]
        else:delta=delta.mean(axis=1)
        values=[]
        for _ in range(draws):
            chosen=rng.integers(0,n,size=n)
            ix=np.concatenate([bygroup[i] for i in chosen])
            ix=ix[mask[ix]]
            if ix.size:values.append(delta[ix].mean())
        lo,hi=np.percentile(values,[2.5,97.5])
        rows.append({'metric':label,'mae_reduction':float(delta[mask].mean()),
                     'ci95_lower':float(lo),'ci95_upper':float(hi),'n':int(mask.sum()),
                     'draws':draws,'seed':seed,'ci_excludes_zero':bool(lo>0 or hi<0)})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--draws',type=int,default=3000)
    a=ap.parse_args()
    files=[a.project/'results'/'round2'/m/'oof_predictions.csv' for m in METHODS]
    frames=[pd.read_csv(p,encoding='utf-8-sig',keep_default_na=False) for p in files]
    left,right=frames
    keys=['id','text','aspect','fold','V','A']
    if not left[keys].equals(right[keys]):raise ValueError('Paired OOF rows/fold/gold do not match')
    if len(left)!=7476 or left.fold.nunique()!=3:raise ValueError('Unexpected historic OOF: inspect before reuse')
    y=left[['V','A']].to_numpy(dtype=float)
    p=left[['pred_V','pred_A']].to_numpy(dtype=float)
    q=right[['pred_V','pred_A']].to_numpy(dtype=float)
    avg=(p+q)/2
    if not np.isfinite(y).all() or not np.isfinite(avg).all():raise ValueError('NaN')
    metrics_rows=[]
    for name,pred in [('round2_marked_l1_epoch3',p),('round2_arousal_mse_aux_epoch3',q),('fixed_equal_weight_ensemble',avg)]:
        metrics_rows.append({'method':name,'origin':'historical 3-fold OOF; no new model trained',**metrics(y,pred)})
    manifest=pd.read_csv(a.project/'results'/'round2'/'fold_assignments.csv',encoding='utf-8-sig',keep_default_na=False)
    if not manifest[['id','text','aspect','fold']].equals(left[['id','text','aspect','fold']]):
        raise ValueError('Original CV connected-group manifest does not match OOF rows')
    group=manifest['group'].to_numpy()
    ci=bootstrap(y,p,avg,group,draws=a.draws)
    out=a.out/'offline_ensemble';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(out/'oof_fixed_blend_metrics.csv',index=False,encoding='utf-8-sig')
    ci.to_csv(out/'group_bootstrap_ci.csv',index=False,encoding='utf-8-sig')
    per_fold=[]
    for fold in sorted(left.fold.unique()):
        ix=left.fold.eq(fold).to_numpy()
        for name,pred in [('base',p),('fixed_half',avg)]:
            per_fold.append({'fold':int(fold),'method':name,**metrics(y[ix],pred[ix])})
    pd.DataFrame(per_fold).to_csv(out/'per_fold_metrics.csv',index=False,encoding='utf-8-sig')
    blended=left[keys].copy()
    blended[['pred_V','pred_A']]=np.round(np.clip(avg,1,9),2)
    blended.to_csv(out/'oof_fixed_blend_predictions.csv',index=False,encoding='utf-8-sig')
    (out/'status.json').write_text(json.dumps({'status':'exploratory offline recalculation; not a new training result',
      'blend_weight':0.5,'sources':[str(p) for p in files], 'official_dev_test_used':False,
      'cannot_deploy_without_new_matching_full_checkpoints_and_development_gate':True,
      'important':'OOF source marked_l1 is epoch3, not deployed refinement marked seed42 epoch2',
      'bootstrap_unit':'original training CV connected groups (union of review_id and exact sentence)',
      'metrics':metrics_rows,'bootstrap':ci.to_dict(orient='records')},ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print(ci.to_string(index=False))

if __name__=='__main__':main()
