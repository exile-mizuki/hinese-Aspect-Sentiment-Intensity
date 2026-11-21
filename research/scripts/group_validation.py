#!/usr/bin/env python3
"""Prepare LEAKAGE-AWARE 3- or 5-fold manifests; does not train anything.
Grouping constraints: same review_id, exact text; normalized text >=12 chars.
NOT claiming near-duplicate/semantic equivalence grouping or independent blind evaluation.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase1_audit import read_raw, normalize, save, write_json

def build(project,out,n_splits,seed):
    _,rows,_,_=read_raw(project)
    rows['norm']=rows.text.map(normalize)
    n=len(rows); parent=np.arange(n)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for col,filter_fn in [('review_id',lambda x:True),('text',lambda x:True),('norm',lambda x:len(x)>=12)]:
        first={}
        for i,value in enumerate(rows[col].tolist()):
            if not filter_fn(value):continue
            if value in first:union(i,first[value])
            else:first[value]=i
    groups=np.array([find(i) for i in range(n)])
    folds=np.full(n,-1,dtype=int)
    audit=[]
    try: splitter=GroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    except TypeError:
        raise RuntimeError('Please upgrade scikit-learn: GroupKFold shuffle requires a recent version')
    for fold,(a,b) in enumerate(splitter.split(rows,groups=groups)):
        folds[b]=fold
        split1,split2=rows.iloc[a],rows.iloc[b]
        shared_reviews=set(split1.review_id)&set(split2.review_id)
        shared_texts=set(split1.text)&set(split2.text)
        norm1=set(x for x in split1.norm if len(x)>=12)
        norm2=set(x for x in split2.norm if len(x)>=12)
        assert not shared_reviews and not shared_texts and not(norm1&norm2)
        audit.append({'fold':fold,'train_n':len(a),'heldout_n':len(b),'train_groups':len(set(groups[a])),
                     'heldout_groups':len(set(groups[b])),'shared_reviews':len(shared_reviews),
                     'shared_exact_texts':len(shared_texts),'shared_long_normalized_texts':len(norm1&norm2),
                     'high_A_heldout':int((split2.A>=6.5).sum())})
    assert min(folds)>=0
    rows.insert(0,'row_index',np.arange(n))
    rows['group']=groups;rows['fold']=folds
    dest=out/'validation'/f'proposed_{n_splits}fold_manifest.csv'
    save(rows[['row_index','id','review_id','text','aspect','V','A','group','fold']],dest)
    write_json({'created_from_training_only':True,'seed':seed,'n_splits':n_splits,'connected_groups':int(len(set(groups))),
      'constraints':['review ID','exact identical sentences','normalized identical sentences if length >=12'],
      'normalization':'Unicode NFKC, lowercase, remove whitespace and punctuation',
      'not_evaluated':'No model trained or evaluated on these folds; high cosine near-duplicate pairs are NOT clustered',
      'folds':audit},out/'validation'/f'proposed_{n_splits}fold_audit.json')
    print(f'{n_splits}-fold split saved; groups={len(set(groups))} folds:',json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--folds',type=int,choices=[3,5],default=5);p.add_argument('--seed',type=int,default=20261119)
    a=p.parse_args();build(a.project,a.out,a.folds,a.seed)
