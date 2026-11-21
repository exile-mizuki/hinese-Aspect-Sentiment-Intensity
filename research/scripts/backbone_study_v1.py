"""Bounded backbone study: two seed-42 screens, at most one multi-seed followup.
Uses only frozen TRAIN folds. This is exploratory after prior model selection.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_target_balance_v1 import training_rows, assert_rows, sha, cluster_ci

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent/'中文方面情感强度预测'
OUT=ROOT/'backbone_v1'
MODELS={'macbert':'chinese-macbert-base','roberta':'chinese-roberta-wwm-ext'}
PLAN={'models':MODELS,'screen_seed':42,'confirmation_seeds':[43,44],
      'screen':'overall mean MAE strictly lower; MULTI mean delta <= .001; HIGH_A A_MAE delta <= .005',
      'selection':'lowest overall mean MAE among eligible seed42 models, name tie-break',
      'confirmation':'3-seed average overall delta < 0, >=2/3 overall wins, MULTI mean delta <= .001, HIGH_A A delta <= .005, overall paired-group CI upper < 0',
      'epochs':3,'batch_size':8,'learning_rate':2e-5,'official_dev_test_read':False,
      'purpose':'bounded exploratory comparison, no automatic deployment; previous Dev/Test already consulted'}

def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def manifest():
    m=pd.read_csv(ROOT/'validation/proposed_3fold_manifest.csv',keep_default_na=False)
    rows,hashes,_=training_rows(PROJECT)
    assert_rows(m,rows,'frozen manifest')
    assert np.array_equal(m.row_index,np.arange(len(m)))
    assert set(m.fold)=={0,1,2} and m.groupby('group').fold.nunique().max()==1
    return m,hashes

def load(name,seed,m,hashes):
    parent=ROOT if name=='baseline' else OUT/name
    external=ROOT/'backbone_comparison_v1'/name
    if name!='baseline' and (external/'training_v2/marked_l1'/f'seed_{seed}'/'fold_2/config.json').is_file():
        parent=external
    frames=[]
    for fold in (0,1,2):
        d=parent/'training_v2/marked_l1'/f'seed_{seed}'/f'fold_{fold}'
        cfg=json.loads((d/'config.json').read_text(encoding='utf-8'))
        expected={'seed':seed,'fold':fold,'epochs':3,'batch_size':8,'learning_rate':2e-5,'max_length':128,'method':'marked_l1','method_config':['l1',0.,False],'dev_test_access':False,'manifest_sha256':sha(ROOT/'validation/proposed_3fold_manifest.csv')}
        for k,v in expected.items():
            assert cfg[k]==v,(name,seed,fold,k)
        recorded=cfg['training_source_file_sha256']
        assert not recorded or recorded==hashes
        if not recorded:
            print('PROVENANCE_NOTE',str(d),'original source hashes empty; frozen rows independently verified',flush=True)
        if name!='baseline':
            assert Path(cfg['model_path']).resolve()==(ROOT.parent/'模型缓存'/MODELS[name]).resolve()
        p=pd.read_csv(d/'heldout_predictions.csv',keep_default_na=False)
        true=m.loc[m.fold==fold].reset_index(drop=True)
        assert_rows(p,true,f'{name} seed{seed} fold{fold}')
        assert np.array_equal(p.group,true.group) and np.array_equal(p.fold,true.fold)
        p['row_index']=true.row_index.to_numpy();frames.append(p)
    p=pd.concat(frames).sort_values('row_index')
    a=p[['pred_V','pred_A']].to_numpy(float)
    assert np.isfinite(a).all() and ((a>=1)&(a<=9)).all()
    return np.abs(a-m[['V','A']].to_numpy(float))

def metrics(error,m):
    multi=m.groupby(['id','text']).aspect.transform('nunique').to_numpy()>1
    masks={'OVERALL':np.ones(len(m),bool),'SINGLE':~multi,'MULTI':multi,'HIGH_A':m.A.to_numpy()>=6.5,
           'EXTREME_A':m.A.to_numpy()>=7.5,'LOW_V':m.V.to_numpy()<3,'MULTI_EXTREME_A':multi&(m.A.to_numpy()>=7.5)}
    return {k:{'n':int(s.sum()),'V_MAE':float(error[s,0].mean()),'A_MAE':float(error[s,1].mean()),'mean_MAE':float(error[s].mean())} for k,s in masks.items()}

def compare(name,seeds,m,hashes):
    b=np.stack([load('baseline',s,m,hashes) for s in seeds])
    c=np.stack([load(name,s,m,hashes) for s in seeds])
    bm,cm=metrics(b.mean(0),m),metrics(c.mean(0),m)
    delta={k:{v:cm[k][v]-bm[k][v] for v in ('V_MAE','A_MAE','mean_MAE')} for k in bm}
    d=(c-b).mean(0);ci=cluster_ci(np.column_stack([d,d.mean(1)]),m.group.to_numpy(),2000,20261120)
    gains=[float(c[i].mean()-b[i].mean()) for i in range(len(seeds))]
    eligible=delta['OVERALL']['mean_MAE']<0 and delta['MULTI']['mean_MAE']<=.001 and delta['HIGH_A']['A_MAE']<=.005
    return {'model':name,'seeds':seeds,'baseline':bm,'candidate':cm,'delta_candidate_minus_baseline':delta,
            'per_seed_overall_delta':dict(zip(map(str,seeds),gains)),'overall_group_bootstrap_95ci':ci,
            'screen_pass':eligible,'cv_confirmation_pass':len(seeds)==3 and eligible and sum(x<0 for x in gains)>=2 and ci['mean_MAE'][1]<0,
            'deployment_accepted':False,'ci_limit':'conditional on saved seeds/folds; does not account for retraining, screening selection, or repeated exploration'}

def run_model(name,seeds):
    external=ROOT/'backbone_comparison_v1'/name/'training_v2/marked_l1'
    completed=[s for s in seeds if all((external/f'seed_{s}'/f'fold_{f}'/'config.json').is_file() for f in [0,1,2])]
    if completed:
        m,h=manifest()
        for s in completed:load(name,s,m,h)
        print('REUSE_EXTERNAL_VERIFIED',name,completed,flush=True)
    seeds=[s for s in seeds if s not in completed]
    if not seeds:return
    args=[sys.executable,'-u',str(ROOT/'scripts/train_backbone_v1.py'),'--project',str(PROJECT),
          '--out',str(OUT/name),'--model-path',str(ROOT.parent/'模型缓存'/MODELS[name]),
          '--seeds',*map(str,seeds),'--epochs','3','--batch-size','8']
    log=OUT/f'{name}_seeds_{"_".join(map(str,seeds))}.log'
    with log.open('a',encoding='utf-8') as stream:
        for action in ('run','aggregate'):
            print('START',name,action,seeds,flush=True)
            subprocess.run(args+['--action',action],stdout=stream,stderr=subprocess.STDOUT,
                           env={**os.environ,'PYTHONIOENCODING':'utf-8','TOKENIZERS_PARALLELISM':'false'},check=True)
    print('COMPLETE',name,seeds,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--analyze-only',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    plan=OUT/'frozen_plan.json'
    if plan.exists():assert json.loads(plan.read_text(encoding='utf-8'))==PLAN
    else:save('frozen_plan.json',PLAN)
    m,h=manifest();screen=[]
    for name in MODELS:
        if not args.analyze_only:run_model(name,[42])
        r=compare(name,[42],m,h);screen.append(r);save(f'{name}_seed42_comparison.json',r)
        print('SCREEN',name,r['delta_candidate_minus_baseline']['OVERALL'],'eligible',r['screen_pass'],flush=True)
    selected=sorted([r for r in screen if r['screen_pass']],key=lambda r:(r['candidate']['OVERALL']['mean_MAE'],r['model']))
    result={'plan':PLAN,'screen':screen,'selected':selected[0]['model'] if selected else None,'deployment_accepted':False}
    save('study_status.json',result)
    if selected:
        name=selected[0]['model']
        if not args.analyze_only:run_model(name,[43,44])
        if all((OUT/name/'training_v2/marked_l1'/f'seed_{s}'/'fold_2/config.json').is_file() for s in [43,44]):
            result['confirmation']=compare(name,[42,43,44],m,h)
            save('study_status.json',result)
            print('CONFIRMATION',json.dumps(result['confirmation'],ensure_ascii=False),flush=True)
    print('STUDY FINISHED',flush=True)

if __name__=='__main__':main()
