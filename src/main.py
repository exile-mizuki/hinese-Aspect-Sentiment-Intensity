"""SIGHAN dimABSA Subtask 1: 训练/开发选参/冻结后的测试。"""
from pathlib import Path
import json,re,sys,platform,hashlib
from collections import defaultdict
import numpy as np,pandas as pd,sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scoring import task1_score
P=Path(__file__).resolve().parents[1]; D=P/'data'; O=P/'results'; O.mkdir(exist_ok=True)
def path(pattern): return next(D.glob(pattern))
def numeric(s): return [float(i) for i in s.split('#')]
def train():
 rows=[]
 for f in sorted(D.glob('*Training*.json')):
  for r in json.loads(f.read_text(encoding='utf-8-sig')):
   assert len(r['Aspect'])==len(r['Intensity'])
   for a,y in zip(r['Aspect'],r['Intensity']):
    rows.append([r['ID'],r['Sentence'],a,*numeric(y)])
 df=pd.DataFrame(rows,columns=['id','text','aspect','V','A'])
 return df.groupby(['id','text','aspect'],sort=False,as_index=False)[['V','A']].mean(),len(df)
def inputs(split):
 rows=[]
 for line in path('*'+split+'*Simplified.txt').read_text(encoding='utf-8-sig').splitlines()[1:]:
  i,tail=line.split(', ',1); s,aa=tail.rsplit(', ',1)
  rows.extend([i,s,a] for a in aa.split('#'))
 return pd.DataFrame(rows,columns=['id','text','aspect'])
def truth(split):
 return pd.read_csv(path('*'+split+'*truth.txt'),sep=' ',encoding='utf-8-sig')
def local(s,a):
 clauses=re.split('[，,。；;！？!?]',s)
 found=[v for v in clauses if a!='NULL' and a in v]
 return ' '.join(found) if found else s
def cols(df):
 return [df.text.tolist(),df.aspect.tolist(),[local(s,a) for s,a in zip(df.text,df.aspect)]]
def features(t,d,e):
 xc=[]
 for traincol,devcol,testcol in zip(cols(t),cols(d),cols(e)):
  v=TfidfVectorizer(analyzer='char',ngram_range=(1,3),sublinear_tf=True,min_df=1)
  xc.append([v.fit_transform(traincol),v.transform(devcol),v.transform(testcol)])
 return xc
def save_pred(d,y,name):
 d=d.copy(); d[['V','A']]=np.round(np.clip(y,1,9),2)
 lines=[(i,''.join(f'({r.aspect},{r.V:.2f}#{r.A:.2f})' for r in g.itertuples())) for i,g in d.groupby('id',sort=False)]
 out=pd.DataFrame(lines,columns=['ID','Intensity'])
 if name: out.to_csv(O/(name+'.txt'),sep=' ',index=False,lineterminator='\n')
 return out
def main():
 t,nraw=train(); d,e=inputs('Validation'),inputs('Testing')
 gold=truth('Validation')
 assert len(t)>0 and np.isfinite(t[['V','A']]).all().all()
 xc=features(t,d,e); summary=[]; details={}; cases=e.copy()
 for name,blocks in [('sentence_only',[0]),('baseline',[0,1]),('improved',[0,1,2])]:
  x,z,u=[hstack([xc[b][k] for b in blocks]).tocsr() for k in range(3)]
  import time,shutil
  start=time.perf_counter(); folder=O/name; folder.mkdir(exist_ok=True); candidate_log=[]
  best=None
  for alpha in (.1,1,5,15):
   reg=Ridge(alpha=alpha,solver='lsqr',tol=1e-5).fit(x,t[['V','A']])
   scores=task1_score(save_pred(d,reg.predict(z),None),gold,{})
   loss=(scores['Valance MAE']+scores['Arousal MAE'])/2
   candidate_log.append(dict(alpha=alpha,dev_mean_MAE=loss,**scores))
   if best is None or loss<best[0]:best=loss,alpha,reg
  _,alpha,reg=best
  for split,rows,mat,ref in [('dev',d,z,gold),('test',e,u,truth('Testing'))]:
   y=reg.predict(mat); pred=save_pred(rows,y,name+'_'+split)
   assert pred.ID.tolist()==ref.ID.tolist()
   sc=task1_score(pred,ref,{})
   summary.append(dict(method=name,split=split,alpha=alpha,**sc))
   shutil.copyfile(O/(name+'_'+split+'.txt'),folder/(split+'_predictions.txt'))
   (folder/(split+'_scores.json')).write_text(json.dumps(sc,indent=2),encoding='utf-8')
   if split=='test':
    cases[[name+'_V',name+'_A']]=np.clip(y,1,9).round(2)
  (folder/'run_log.json').write_text(json.dumps(dict(method=name,channels=blocks,train_rows=len(t),dev_rows=len(d),test_rows=len(e),selection='minimum development mean MAE',selected_alpha=alpha,solver='lsqr',elapsed_seconds=round(time.perf_counter()-start,3),candidates=candidate_log),indent=2),encoding='utf-8')
 for row in truth('Testing').itertuples():
  for a,v,a2 in re.findall(r'\(([^,]*),([\d.]+)#([\d.]+)\)',row.Intensity):
   flag=(cases.id==row.ID)&(cases.aspect==a); cases.loc[flag,['true_V','true_A']]=[float(v),float(a2)]
 assert cases[['true_V','true_A']].notna().all().all()
 cases.to_csv(O/'cases.csv',index=False,encoding='utf-8-sig')
 stats={k:{'sentences':int(v.id.nunique()),'aspect_rows':len(v),'null_aspect':int((v.aspect=='NULL').sum()),'blank_text':int(v.text.str.strip().eq('').sum())} for k,v in [('train',t),('dev',d),('test',e)]}
 (O/'run_log.json').write_text(json.dumps(dict(stats=stats,raw_train_tuples=nraw,python=sys.version,sklearn=sklearn.__version__,hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in D.iterdir() if f.is_file()}),ensure_ascii=False,indent=2),encoding='utf-8')
 pd.DataFrame(summary).to_csv(O/'metrics.csv',index=False,encoding='utf-8-sig');print(stats);print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':
 main()
 from diagnostics import analyze
 analyze(O)
