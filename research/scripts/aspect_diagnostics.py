#!/usr/bin/env python3
"""Aspect-marker diagnostics using Train-only OOF. Character length is NOT tokenizer length."""
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd
from training_utils_v2 import scored_metrics

def run(project,out):
    sys.path.insert(0,str(project/'src'))
    from main import train
    rows,_=train()
    base=pd.read_csv(project/'results/round2/marked_l1/oof_predictions.csv',encoding='utf-8-sig',keep_default_na=False)
    alt=pd.read_csv(out/'offline_ensemble/oof_fixed_blend_predictions.csv',encoding='utf-8-sig',keep_default_na=False)
    for d in [base,alt]:
        if not rows[['id','text','aspect']].equals(d[['id','text','aspect']]):
            raise ValueError('OOF rows do not align with source labels')
    part=rows[['id','text','aspect','V','A']].copy()
    part['review_id']=part.id.str.split(':').str[0]
    part['aspect_occurrences']=[0 if a=='NULL' else s.count(a) for s,a in zip(part.text,part.aspect)]
    part['aspect_case']=np.select([part.aspect.eq('NULL'),part.aspect_occurrences.eq(0),part.aspect_occurrences.eq(1),part.aspect_occurrences.ge(2)],
                                  ['NULL','aspect_absent','one_occurrence','repeated_occurrences'],default='other')
    part['text_chars']=part.text.str.len()
    part['chars_gt_128']=part.text_chars.gt(128)
    same=part.groupby(['id','text'],sort=False).aspect.transform('size')
    part['multiple_aspects']=same.ge(2)
    part[['base_pred_V','base_pred_A']]=base[['pred_V','pred_A']]
    part[['blend_pred_V','blend_pred_A']]=alt[['pred_V','pred_A']]
    metrics=[]
    def append(name,mask):
        if not mask.any():return
        block=part.loc[mask]
        y=block[['V','A']].to_numpy()
        b=block[['base_pred_V','base_pred_A']].to_numpy()
        c=block[['blend_pred_V','blend_pred_A']].to_numpy()
        x=scored_metrics(y,b);z=scored_metrics(y,c)
        metrics.append({'slice':name,'n':len(block),'baseline_A_MAE':x['A_MAE'],'blended_A_MAE':z['A_MAE'],
                        'A_mae_reduction':x['A_MAE']-z['A_MAE'],
                        'baseline_mean_MAE':x['mean_MAE'],'blended_mean_MAE':z['mean_MAE'],
                        'mean_mae_reduction':x['mean_MAE']-z['mean_MAE']})
    for k in ['NULL','aspect_absent','one_occurrence','repeated_occurrences']:
        append('aspect_case='+k,part.aspect_case.eq(k).to_numpy())
    for name,mask in [('multi_aspect',part.multiple_aspects),('single_aspect',~part.multiple_aspects),
                      ('text_chars_gt_128',part.chars_gt_128),('text_chars_le_128',~part.chars_gt_128),
                      ('A_ge_7_5',part.A.ge(7.5))]:append(name,mask.to_numpy())
    folder=out/'training_v2';folder.mkdir(parents=True,exist_ok=True)
    part.to_csv(folder/'aspect_marker_train_oof_detail.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(metrics).to_csv(folder/'aspect_marker_slices.csv',index=False,encoding='utf-8-sig')
    stats={'train_rows':len(part),'aspect_cases':part.aspect_case.value_counts().to_dict(),
           'multi_aspect_rows':int(part.multiple_aspects.sum()),
           'text_longer_than_128_chars':int(part.chars_gt_128.sum()),
           'max_text_chars':int(part.text_chars.max()),
           'marker_implementation':'all occurrences replaced by str.replace; NULL not marked',
           'important_limit':'Character count is NOT tokenizer sequence length. Exact truncation cannot be tested without transformers tokenizer.',
           'source':'Only training OOF (no Dev/Test)'}
    (folder/'aspect_marker_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2));print(pd.DataFrame(metrics).to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();run(a.project,a.out)
