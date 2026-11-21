"""固定预测的误差诊断；这些测试集分析不用于选择模型。"""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

def analyze(out):
    df=pd.read_csv(out/'cases.csv',keep_default_na=False)
    df['multi']=df.groupby('id').aspect.transform('nunique')>1
    slices=[]; pairs=[]
    for m in ['baseline','improved','sentence_only']:
        for flag,part in df.groupby('multi'):
            row=dict(method=m,group='多方面' if flag else '单方面',sentences=part.id.nunique(),aspects=len(part))
            for k in 'VA':
                row[k+'_MAE']=(part[m+'_'+k]-part['true_'+k]).abs().mean()
            slices.append(row)
        for k in 'VA':
            counts=[0,0,0]
            for _,g in df[df.multi].groupby('id'):
                for (a,b),(c,d) in combinations(g[['true_'+k,m+'_'+k]].to_numpy(),2):
                    if a==c: continue
                    sign=(a-c)*(b-d)
                    counts[0 if sign>0 else 1 if sign==0 else 2]+=1
            n=sum(counts)
            pairs.append(dict(method=m,dimension=k,eligible_pairs=n,correct=counts[0],tied=counts[1],reversed=counts[2],strict_correct_rate=counts[0]/n))
    s=pd.DataFrame(slices); q=pd.DataFrame(pairs)
    s.to_csv(out/'single_multi_metrics.csv',index=False,encoding='utf-8-sig')
    q.to_csv(out/'aspect_order.csv',index=False,encoding='utf-8-sig')
    print(s.to_string(index=False)); print(q.to_string(index=False))
if __name__=='__main__': analyze(Path(__file__).resolve().parents[1]/'results')
