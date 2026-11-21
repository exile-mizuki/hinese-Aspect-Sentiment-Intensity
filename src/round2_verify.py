"""Recompute saved results and audit grouped validation, without training."""
import hashlib
import json
import numpy as np
import pandas as pd
from main import D, O, inputs, train, truth
from scoring import task1_score
from window_experiment import extract_scores
from round2_experiment import metrics, METHODS
from round2_errors import check_rules
from bert_experiment import write_json

OUT=O/'round2'


def main():
    checks=[]
    plan=json.loads((OUT/'experiment_plan.json').read_text(encoding='utf-8'))
    source=__import__('pathlib').Path(__file__).with_name('round2_experiment.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_sha256']
    checks.append('training source matches frozen experiment plan')
    for name,digest in plan['training_sha256'].items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    checks.append('training input hashes unchanged')
    rows,_=train()
    manifest=pd.read_csv(OUT/'fold_assignments.csv',keep_default_na=False)
    assert rows[['id','text','aspect']].equals(manifest[['id','text','aspect']])
    assert np.allclose(rows[['V','A']],manifest[['V','A']])
    assert set(manifest.fold)=={0,1,2}
    for fold in range(3):
        a,b=manifest[manifest.fold!=fold],manifest[manifest.fold==fold]
        for column in ['review_id','text','group']:
            assert not(set(a[column])&set(b[column])),(fold,column)
    checks.append('all three folds have disjoint reviews, exact text and connected groups')
    pooled=pd.read_csv(OUT/'cv_pooled_metrics.csv').set_index('method')
    foldmetrics=pd.read_csv(OUT/'cv_fold_metrics.csv')
    def match(expected,actual):
        for key,value in actual.items():
            if value is not None: assert np.isclose(float(expected[key]),value,atol=1e-6),(key,expected[key],value)
    for method in METHODS:
        oof=pd.read_csv(OUT/method/'oof_predictions.csv',keep_default_na=False)
        assert oof[['id','text','aspect']].equals(rows[['id','text','aspect']])
        assert oof.fold.equals(manifest.fold)
        assert np.isfinite(oof[['pred_V','pred_A']]).all().all()
        match(pooled.loc[method],metrics(oof[['V','A']].to_numpy(),oof[['pred_V','pred_A']].to_numpy()))
        for fold in range(3):
            subset=oof[oof.fold==fold]
            rec=foldmetrics[(foldmetrics.method==method)&(foldmetrics.fold==fold)].iloc[0]
            match(rec,metrics(subset[['V','A']].to_numpy(),subset[['pred_V','pred_A']].to_numpy()))
    checks.append('all 9 fold metrics and 3 pooled OOF metrics reproduced')
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    baseline=pooled.loc['marked_l1']
    eligible=[m for m in METHODS[1:] if pooled.loc[m,'A_MAE']<baseline.A_MAE and pooled.loc[m,'high_A_MAE']<baseline.high_A_MAE and pooled.loc[m,'mean_MAE']<=baseline.mean_MAE+.003]
    chosen=min(eligible,key=lambda m:(pooled.loc[m,'A_MAE'],m)) if eligible else 'marked_l1'
    assert selection['selected']==chosen and selection['eligible']==eligible
    checks.append('selection matches predeclared training-only rule')
    official=pd.read_csv(OUT/'official_metrics.csv')
    for r in official.itertuples():
        split='Validation' if r.split=='dev' else 'Testing'
        predictions=pd.read_csv(OUT/r.method/'full'/f'{r.split}_predictions.txt',sep=' ',keep_default_na=False)
        gold=truth(split)
        got=task1_score(predictions,gold,{})
        saved=json.loads((OUT/r.method/'full'/f'{r.split}_scores.json').read_text(encoding='utf-8'))
        for key,value in got.items():
            assert np.isclose(value,saved['official'][key]),(r.method,r.split,key)
        expected=inputs(split)
        match(r._asdict(),metrics(extract_scores(gold,expected),extract_scores(predictions,expected)))
        cases=pd.read_csv(OUT/r.method/'full'/f'{r.split}_cases.csv',keep_default_na=False)
        assert expected.equals(cases[['id','text','aspect']])
        assert np.allclose(cases[['true_V','true_A']],extract_scores(gold,expected))
        assert np.allclose(cases[['pred_V','pred_A']],extract_scores(predictions,expected))
    checks.append('all 4 official development/test score files reproduced')
    deployment=json.loads((OUT/'deployment.json').read_text(encoding='utf-8'))
    dev=official[official.split=='dev'].set_index('method')
    c,b=dev.loc[deployment['candidate']],dev.loc['marked_l1']
    passed=chosen!='marked_l1' and c.A_MAE<b.A_MAE and c.high_A_MAE<b.high_A_MAE and c.mean_MAE<=b.mean_MAE+.003
    assert deployment['accepted']==passed
    checks.append('deployment decision uses CV and development only')
    check_rules()
    for split in ['dev','test']:
        diagnostics=pd.read_csv(OUT/f'{split}_diagnostic_cases.csv',keep_default_na=False)
        for r in diagnostics.itertuples(index=False,name=None):
            data=dict(zip(diagnostics.columns,r))
            for e in json.loads(data['原文线索_位置从0开始']):
                assert data['text'][e['start']:e['end']]==e['text']
    checks.append('Chinese exclusion rules and all evidence offsets checked')
    write_json(OUT/'verification.json',dict(passed=True,checks=checks,count=len(checks)))
    print('全部复核通过。')
    print('已核对：数据划分、折外预测、官方评分、模型选择及错误线索位置。')
    print('本命令没有重新训练。')


if __name__=='__main__':main()
