#!/usr/bin/env python3
"""Independent, read-only audit for SIGHAN2024 dimABSA; NEVER changes source files.
Usage: python scripts/phase1_audit.py --project <original unpacked project> --out <output directory>
This script inspects test data only for split integrity and dataset descriptives;
TEST labels are NOT used for model tuning or candidate selection.
"""
from __future__ import annotations
import argparse, collections, datetime as dt, hashlib, json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

RATINGS=re.compile(r'\(([^,]*),([0-9.]+)#([0-9.]+)\)')
TAGS = {
    '否定线索': r'没有|并非|不是|不太|不够|不会|不能|没|不|未|无',
    '转折线索': r'但是|不过|然而|可是|虽然|却|但',
    '程度强调线索': r'非常|超级|极其|十分|完全|实在|特别|太|超|最|[!！]+',
    '隐含评价候选': r'排队|等了|等很久|放很久|放了很久|吃不完|吃完|剩下|见底|回访|再来|第二次|第三次|不会再|不再|没再|冷掉|凉掉',
    '指代线索': r'这个|那个|这些|那些|它们|他们|前者|后者|这家|那家',
}
EXCLUSIONS = re.compile(r'不但|不仅|不只|不得不|没想到|无论|不管')
def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,index=False,encoding='utf-8-sig')
def write_json(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def review_id(s): return str(s).split(':',1)[0]
def normalize(s):
    s=unicodedata.normalize('NFKC',str(s)).casefold()
    return ''.join(c for c in s if not c.isspace() and not unicodedata.category(c).startswith('P'))
def label_cues(text,aspect,multi):
    excluded=[(m.start(),m.end()) for m in EXCLUSIONS.finditer(text)]
    labels=[]
    for k,pat in TAGS.items():
        for m in re.finditer(pat,text):
            if k in ('否定线索','转折线索') and any(x<=m.start()<y for x,y in excluded): continue
            labels.append(k)
    if multi: labels.append('同句多方面')
    if aspect=='NULL': labels.append('隐式方面NULL')
    elif aspect not in text: labels.append('方面未字面匹配')
    elif text.count(aspect)>1: labels.append('方面重复出现')
    return sorted(set(labels))
def read_raw(root):
    issues=[]; annotations=[]; source_summary=[]
    for source in sorted((root/'data').glob('*Training*.json')):
        objs=json.loads(source.read_text(encoding='utf-8-sig'))
        source_summary.append({'file':source.name,'raw_sentence_records':len(objs),'sha256':sha(source)})
        for idx,r in enumerate(objs):
            iid=r.get('ID',''); txt=r.get('Sentence','')
            fields=('Aspect','Intensity','Opinion','Category')
            lengths={k:len(r.get(k,[])) if isinstance(r.get(k),list) else None for k in fields}
            if len(set(lengths.values()))!=1 or None in lengths.values():
                issues.append({'source':source.name,'record_index':idx,'id':iid,'issue':'array_length_mismatch','detail':str(lengths)})
            n=min([v for v in lengths.values() if isinstance(v,int)],default=0)
            if not str(iid).strip() or not str(txt).strip():
                issues.append({'source':source.name,'record_index':idx,'id':iid,'issue':'blank_id_or_sentence','detail':repr(txt)})
            for j in range(n):
                y=r['Intensity'][j]
                try: v,a=[float(x) for x in y.split('#')]
                except (ValueError,AttributeError):
                    v,a=np.nan,np.nan
                annotations.append({'source':source.name,'id':str(iid),'review_id':review_id(iid),'text':str(txt),
                   'aspect':str(r['Aspect'][j]),'category':str(r['Category'][j]),'opinion':str(r['Opinion'][j]),
                   'V':v,'A':a,'raw_position':j})
    raw=pd.DataFrame(annotations)
    if raw.empty: raise RuntimeError('No training annotations read')
    train=raw.groupby(['id','text','aspect'],sort=False,dropna=False,as_index=False)[['V','A']].mean()
    train['review_id']=train.id.map(review_id)
    return raw,train,source_summary,issues

def read_official(root, tag):
    root=root/'data'
    fi=next(root.glob(f'*{tag}*Simplified.txt'))
    truth=next(root.glob(f'*{tag}*truth.txt'))
    rows=[]
    for lineno,line in enumerate(fi.read_text(encoding='utf-8-sig').splitlines()[1:],2):
        if not line.strip():continue
        rid,rem=line.split(', ',1); txt,asp=rem.rsplit(', ',1)
        for a in asp.split('#'):
            rows.append({'id':rid,'text':txt,'aspect':a,'review_id':review_id(rid)})
    data=pd.DataFrame(rows)
    labels={}
    for line in truth.read_text(encoding='utf-8-sig').splitlines()[1:]:
        if not line.strip():continue
        rid, packed=line.split(' ',1)
        vals=RATINGS.findall(packed)
        for a,v,ar in vals:
            key=(rid,a)
            if key in labels: raise ValueError('Duplicated truth key: '+str(key))
            labels[key]=(float(v),float(ar))
    if len(labels)!=len(data): raise ValueError(f'{tag} truth count mismatch: {len(labels)} vs {len(data)}')
    if not all((r.id,r.aspect) in labels for r in data.itertuples()):raise ValueError(f'{tag} truth keys do not match')
    data[['V','A']]=[labels[(r.id,r.aspect)] for r in data.itertuples()]
    return data,{'input_file':fi.name,'input_sha256':sha(fi),'truth_file':truth.name,'truth_sha256':sha(truth)}

def split_stats(name,df):
    multi=df.groupby('id').aspect.nunique()
    is_multi=df.id.isin(multi[multi>1].index)
    return {'split':name,'sentence_ids':int(df.id.nunique()),'unique_review_ids':int(df.review_id.nunique()),
        'aspect_instances':len(df),'unique_aspects':int(df.aspect.nunique()),'null_aspects':int((df.aspect=='NULL').sum()),
        'blank_sentences':int(df.text.str.strip().eq('').sum()),'blank_aspects':int(df.aspect.str.strip().eq('').sum()),
        'missing_labels':int(df[['V','A']].isna().any(axis=1).sum()),
        'invalid_V':int((~np.isfinite(df.V)|~df.V.between(1,9)).sum()),
        'invalid_A':int((~np.isfinite(df.A)|~df.A.between(1,9)).sum()),
        'duplicate_id_aspect':int(df.duplicated(['id','aspect']).sum()),
        'duplicate_text_aspect':int(df.duplicated(['text','aspect']).sum()),
        'multi_aspect_sentences':int((multi>1).sum()),'multi_aspect_instances':int(is_multi.sum()),
        'A_ge_6_5':int((df.A>=6.5).sum()),'A_ge_7_5':int((df.A>=7.5).sum())}

def overlap_diagnostics(frames):
    summaries=[]; details=[]
    for ai,left in enumerate(frames):
        for right in list(frames)[ai+1:]:
            x,y=frames[left],frames[right]
            sets=lambda df,col: set(df[col].tolist())
            rev=sets(x,'review_id')&sets(y,'review_id')
            ex=sets(x,'text')&sets(y,'text')
            nx=sets(x,'normalized_text')&sets(y,'normalized_text')
            ids=sets(x,'id')&sets(y,'id')
            pairs=set(zip(x.text,x.aspect))&set(zip(y.text,y.aspect))
            both=x[x.review_id.isin(rev)]
            bothy=y[y.review_id.isin(rev)]
            summaries.append({'left':left,'right':right,'same_sentence_IDs':len(ids),'shared_review_IDs':len(rev),
             'left_instances_in_shared_reviews':len(both),'right_instances_in_shared_reviews':len(bothy),
             'shared_exact_texts':len(ex),'shared_normalized_texts':len(nx),'shared_exact_text_aspects':len(pairs),
             'left_instances_same_exact_text':int(x.text.isin(ex).sum()),'right_instances_same_exact_text':int(y.text.isin(ex).sum()),
             'left_instances_same_normalized_text':int(x.normalized_text.isin(nx).sum()),'right_instances_same_normalized_text':int(y.normalized_text.isin(nx).sum())})
            for label, keys,col in [('same_review',rev,'review_id'),('exact_text',ex,'text'),('normalized_text',nx,'normalized_text')]:
                for v in sorted(keys):
                    a=x.loc[x[col]==v,['id','aspect','text']]; b=y.loc[y[col]==v,['id','aspect','text']]
                    details.append({'left':left,'right':right,'relation':label,'key':v,
                         'left_ids':';'.join(a.id.unique()[:20]),'right_ids':';'.join(b.id.unique()[:20]),
                         'left_count':len(a),'right_count':len(b),'sample_left':str(a.iloc[0].text)[:160],
                         'sample_right':str(b.iloc[0].text)[:160]})
    return pd.DataFrame(summaries),pd.DataFrame(details)

def near_duplicate_search(frames):
    """TFIDF char n-gram cosine, cross-split; EXPLORATORY heuristic (not exhaustive semantic duplication)."""
    df=pd.concat([x[['text','normalized_text']].assign(split=name) for name,x in frames.items()],ignore_index=True)
    df=df.drop_duplicates(['split','text']).reset_index(drop=True)
    x=TfidfVectorizer(analyzer='char',ngram_range=(2,4),sublinear_tf=True).fit_transform(df.normalized_text)
    out=[]
    for i,left in enumerate(frames):
        for right in list(frames)[i+1:]:
            ia=np.flatnonzero(df.split.values==left); ib=np.flatnonzero(df.split.values==right)
            if not len(ia) or not len(ib): continue
            mat=x[ib]
            for block in np.array_split(ia,max(1,(len(ia)+149)//150)):
                sims=(x[block]@mat.T).tocsr()
                for pos,idx in enumerate(block):
                    a=df.iloc[idx]
                    if len(a.normalized_text)<12:continue
                    row=sims.getrow(pos)
                    candidates=[(j,v) for j,v in zip(row.indices,row.data) if .92<=v<.999999]
                    for j,score in sorted(candidates,key=lambda x:-x[1])[:8]:
                        b=df.iloc[ib[j]]
                        if len(b.normalized_text)<12 or a.normalized_text==b.normalized_text:continue
                        out.append({'left':left,'right':right,'left_text':a.text,'right_text':b.text,
                                    'cosine_similarity':round(float(score),5),'needs_human_review':True})
    out.sort(key=lambda r:-r['cosine_similarity'])
    return pd.DataFrame(out,columns=['left','right','left_text','right_text','cosine_similarity','needs_human_review'])

def assign_bins(frame):
    bins=[1,2,3,4,5,6,7,8,9.00001]
    out=[]
    for dim in ('V','A'):
        y=frame[dim]
        for j in range(8):
            lo,hi=bins[j],bins[j+1]
            out.append({'dimension':dim,'interval':f'[{j+1},{j+2}{"]" if j==7 else ")"}',
                        'count':int(((y>=lo)&(y<hi)).sum()),'pct':round(float(((y>=lo)&(y<hi)).mean()*100),3)})
    return pd.DataFrame(out)

def matched_analysis(before,after,split):
    key=['id','text','aspect']; assert before[key].equals(after[key]),f'{split} pair alignment failed'
    assert np.allclose(before[['V','A']],after[['V','A']]), f'{split} truth mismatch'
    d=before[key+['V','A']].copy()
    d[['baseline_V','baseline_A']]=before[['pred_V','pred_A']]
    d[['candidate_V','candidate_A']]=after[['pred_V','pred_A']]
    for m in ['baseline','candidate']:
        d[f'{m}_V_error']=(d[f'{m}_V']-d['V']).abs()
        d[f'{m}_A_error']=(d[f'{m}_A']-d['A']).abs()
        d[f'{m}_mean_error']=(d[f'{m}_V_error']+d[f'{m}_A_error'])/2
    d['gain_positive_better']=d.baseline_mean_error-d.candidate_mean_error
    d['review_id']=d.id.map(review_id)
    d['multi']=d.groupby('id').aspect.transform('size')>1
    d['rule_tags']=[('|'.join(label_cues(r.text,r.aspect,bool(r.multi)))) for r in d.itertuples()]
    return d

def slice_report(df,split):
    cats=['全部样本','高A>=6.5','高A>=7.5']+list(TAGS)+['同句多方面','隐式方面NULL','方面未字面匹配','方面重复出现']
    out=[]
    for c in cats:
        if c=='全部样本': sub=df
        elif c=='高A>=6.5':sub=df[df.A>=6.5]
        elif c=='高A>=7.5':sub=df[df.A>=7.5]
        else: sub=df[df.rule_tags.map(lambda t:c in t.split('|'))]
        if sub.empty:continue
        gain=sub.gain_positive_better
        out.append({'split':split,'machine_defined_slice':c,'n':len(sub),'reviews':sub.review_id.nunique(),
                    'baseline_V_MAE':round(sub.baseline_V_error.mean(),5),'candidate_V_MAE':round(sub.candidate_V_error.mean(),5),
                    'baseline_A_MAE':round(sub.baseline_A_error.mean(),5),'candidate_A_MAE':round(sub.candidate_A_error.mean(),5),
                    'baseline_mean_MAE':round(sub.baseline_mean_error.mean(),5),'candidate_mean_MAE':round(sub.candidate_mean_error.mean(),5),
                    'improved_n':int((gain>1e-9).sum()),'regressed_n':int((gain<-1e-9).sum()),'tied_n':int((gain.abs()<=1e-9).sum()),
                    'status':'自动规则统计，未完成人工原因确认'})
    return out

def select_review(df,n=24):
    d=df.copy(); d['reason']=''; d['abs_remaining']=d.candidate_mean_error
    # reserve both improvement and regression, plus hard cases and rare high A
    groups=[('严重退步',d.sort_values('gain_positive_better').head(12)),
            ('显著改善',d.sort_values('gain_positive_better',ascending=False).head(12)),
            ('仍然难例',d.sort_values('abs_remaining',ascending=False).head(14)),
            ('高A长尾',d[d.A>=7.5].sort_values('abs_remaining',ascending=False).head(8)),
            ('同句多方面',d[d.multi].sort_values('abs_remaining',ascending=False).head(8))]
    chosen=[]; seen=set()
    for reason,part in groups:
        for _,r in part.iterrows():
            key=(r.id,r.aspect)
            if key in seen:continue
            if sum(x['抽样原因']==reason for x in chosen)>=max(1,n//len(groups)):continue
            chosen.append({'id':r.id,'text':r.text,'aspect':r.aspect,'true_V':r.V,'true_A':r.A,
                'baseline_V':r.baseline_V,'baseline_A':r.baseline_A,'candidate_V':r.candidate_V,'candidate_A':r.candidate_A,
                'baseline_mean_error':round(r.baseline_mean_error,4),'candidate_mean_error':round(r.candidate_mean_error,4),
                'gain_positive_better':round(r.gain_positive_better,4),'规则标签_待确认':r.rule_tags,
                '抽样原因':reason,'人工确认标签':'','人工解释':'','是否原标注有歧义':'','复核人':'','复核日期':''})
            seen.add(key)
    for _,r in d.sort_values('abs_remaining',ascending=False).iterrows():
        if len(chosen)>=n:break
        key=(r.id,r.aspect)
        if key in seen:continue
        chosen.append({'id':r.id,'text':r.text,'aspect':r.aspect,'true_V':r.V,'true_A':r.A,
                'baseline_V':r.baseline_V,'baseline_A':r.baseline_A,'candidate_V':r.candidate_V,'candidate_A':r.candidate_A,
                'baseline_mean_error':round(r.baseline_mean_error,4),'candidate_mean_error':round(r.candidate_mean_error,4),
                'gain_positive_better':round(r.gain_positive_better,4),'规则标签_待确认':r.rule_tags,
                '抽样原因':'高误差补足','人工确认标签':'','人工解释':'','是否原标注有歧义':'','复核人':'','复核日期':''})
        seen.add(key)
    return pd.DataFrame(chosen[:n])

def run(project,out):
    for name in ['data_audit','leakage_check','data_diagnostics','error_analysis','validation','docs']:(out/name).mkdir(exist_ok=True,parents=True)
    raw,tr,srcs,issues=read_raw(project)
    dev,ds=read_official(project,'Validation'); test,ts=read_official(project,'Testing')
    frames={'train':tr,'dev':dev,'test':test}
    for d in frames.values(): d['normalized_text']=d.text.map(normalize)
    write_json({'sources':srcs,'validation':ds,'testing':ts,'pipeline_note':'Training rows collapse all opinion-level intensity records sharing id,text,aspect by mean. Original data are untouched.','audited_at_UTC':dt.datetime.now(dt.timezone.utc).isoformat()},out/'data_audit'/'source_manifest.json')
    save(pd.DataFrame(issues,columns=['source','record_index','id','issue','detail']),out/'data_audit'/'structural_issues.csv')
    bad=raw[~np.isfinite(raw[['V','A']]).all(axis=1) | ~raw[['V','A']].apply(lambda c:c.between(1,9)).all(axis=1)]
    save(bad,out/'data_audit'/'invalid_training_labels.csv')
    summary=pd.DataFrame([split_stats(name,d) for name,d in frames.items()]); save(summary,out/'data_audit'/'dataset_summary.csv')
    # distinguish legitimate multiple opinion ratings from duplicate keyed labels
    groups=raw.groupby(['id','text','aspect'],sort=False,dropna=False)
    ann=groups.agg(opinion_records=('id','size'),unique_ratings=('V',lambda x:len(set(zip(x,raw.loc[x.index,'A'])))),
          unique_opinions=('opinion','nunique'),unique_categories=('category','nunique'),
          min_V=('V','min'),max_V=('V','max'),min_A=('A','min'),max_A=('A','max')).reset_index()
    multi=ann[ann.opinion_records>1].copy()
    multi['different_ratings']=(multi.unique_ratings>1)
    multi['review_flag']='多评价词同一方面，不等于标注错误；请核查聚合规则'
    save(multi,out/'data_audit'/'multi_opinion_same_aspect.csv')
    cross=tr.groupby(['text','aspect'],sort=False).agg(ids=('id',lambda x:';'.join(sorted(set(x)))),
          count=('id','size'),min_V=('V','min'),max_V=('V','max'),min_A=('A','min'),max_A=('A','max')).reset_index()
    cross=cross[cross['count']>1].copy()
    cross['has_label_disagreement']=(cross.max_V-cross.min_V>0.000001)|(cross.max_A-cross.min_A>0.000001)
    cross['review_flag']='不同训练实例出现相同正文和方面，标签差异需人工判断，不直接删'
    save(cross,out/'data_audit'/'duplicate_text_aspect_review.csv')
    categorical=raw.groupby(['category','aspect'],dropna=False).agg(raw_annotations=('id','size'),unique_sentences=('id','nunique')).reset_index()
    save(categorical.sort_values('raw_annotations',ascending=False),out/'data_diagnostics'/'category_aspect_counts.csv')
    for name,d in frames.items():
        save(assign_bins(d),out/'data_diagnostics'/f'{name}_label_bins.csv')
        save(d.groupby('aspect').agg(n=('id','size'),reviews=('review_id','nunique'),V_mean=('V','mean'),A_mean=('A','mean')).reset_index().sort_values('n',ascending=False),out/'data_diagnostics'/f'{name}_aspect_frequency.csv')
    o,detail=overlap_diagnostics(frames)
    save(o,out/'leakage_check'/'cross_split_overlaps.csv');save(detail,out/'leakage_check'/'cross_split_overlap_details.csv')
    near=near_duplicate_search(frames)
    save(near,out/'leakage_check'/'near_duplicates_cross_split_candidates.csv')
    # Check saved CV predictions and metrics independently (no model training)
    r=project/'results'/'round2'
    cv={}
    files={'baseline':r/'marked_l1'/'oof_predictions.csv','candidate':r/'arousal_mse_aux'/'oof_predictions.csv'}
    for k,p in files.items():
        if not p.exists(): raise FileNotFoundError(p)
        cv[k]=pd.read_csv(p,keep_default_na=False)
        assert tr[['id','text','aspect']].equals(cv[k][['id','text','aspect']]),'OOF order or identities mismatch'
    folds=pd.read_csv(r/'fold_assignments.csv',keep_default_na=False)
    assert tr[['id','text','aspect']].equals(folds[['id','text','aspect']]),'manifest mismatch'
    fold_checks=[]
    for fold in sorted(folds.fold.unique()):
        a,b=folds[folds.fold!=fold],folds[folds.fold==fold]
        shared_rev=set(a.review_id)&set(b.review_id)
        shared_text=set(a.text)&set(b.text)
        shared_norm=set(a.text.map(normalize))&set(b.text.map(normalize))
        fold_checks.append({'fold':int(fold),'train_n':len(a),'heldout_n':len(b),
            'shared_reviews':len(shared_rev),'shared_exact_texts':len(shared_text),
            'shared_normalized_texts':len(shared_norm)})
    save(pd.DataFrame(fold_checks),out/'validation'/'existing_3fold_integrity.csv')
    pd.DataFrame(fold_checks).to_json(out/'validation'/'existing_3fold_integrity.json',orient='records',force_ascii=False,indent=2)
    actual=[]; reports=[]
    for split in ('oof_train','dev'):
        if split=='oof_train':
            bs,cs=cv['baseline'],cv['candidate']
            for k,f in [('baseline',bs),('candidate',cs)]:
                error=np.abs(np.round(np.clip(f[['pred_V','pred_A']].values,1,9),2)-f[['V','A']].values)
                actual.append({'split':split,'method':k,'n':len(f),'V_MAE':error[:,0].mean(),'A_MAE':error[:,1].mean(),'mean_MAE':error.mean(),'high_A_MAE':error[f.A.values>=6.5,1].mean()})
        else:
            bs=pd.read_csv(r/'marked_l1'/'full'/'dev_cases.csv',keep_default_na=False).rename(columns={'true_V':'V','true_A':'A'})
            cs=pd.read_csv(r/'arousal_mse_aux'/'full'/'dev_cases.csv',keep_default_na=False).rename(columns={'true_V':'V','true_A':'A'})
            for k,f in [('baseline',bs),('candidate',cs)]:
                error=np.abs(np.round(np.clip(f[['pred_V','pred_A']].values,1,9),2)-f[['V','A']].values)
                actual.append({'split':split,'method':k,'n':len(f),'V_MAE':error[:,0].mean(),'A_MAE':error[:,1].mean(),'mean_MAE':error.mean(),'high_A_MAE':error[f.A.values>=6.5,1].mean()})
        paired=matched_analysis(bs,cs,split)
        save(paired,out/'error_analysis'/f'{split}_paired_errors.csv')
        reports+=slice_report(paired,split)
        if split=='oof_train':save(select_review(paired,24),out/'error_analysis'/'新增训练折外24例_待人工复核.csv')
    save(pd.DataFrame(actual),out/'validation'/'independent_metric_recalculation.csv')
    save(pd.DataFrame(reports),out/'error_analysis'/'machine_rule_slices.csv')
    existing=pd.read_csv(r/'人工复核案例_待填写.csv',keep_default_na=False)
    assert len(existing)==24
    assert not existing['人工确认标签'].astype(str).str.strip().any(), 'Existing review unexpectedly has user-entered tags; check before rewriting'
    existing.insert(0,'来源与用途','原项目测试集：仅展示与人工分析，不用于后续选模调参')
    save(existing,out/'error_analysis'/'原项目24例_保留待人工复核.csv')
    # No file modifications in input project.
    meta={'raw_annotations':len(raw),'aggregated_train_instances':len(tr),'multi_opinion_groups':len(multi),
          'multi_opinion_groups_with_different_ratings':int(multi.different_ratings.sum()),
          'duplicate_text_aspect_groups_across_training_IDs':len(cross),
          'duplicate_text_aspect_groups_label_differences':int(cross.has_label_disagreement.sum()),
          'cross_split_near_duplicate_candidates':len(near),
          'review_count_existing':len(existing),'review_count_new_oof':24,
          'source_original_untouched':True,
          'warning':'Official test labels and scores were examined in earlier experiments. No fresh blind-test performance claim permitted.'}
    write_json(meta,out/'data_audit'/'audit_findings.json')
    print(json.dumps({'summary':summary.to_dict(orient='records'),'overlaps':o.to_dict(orient='records'),
             'cv_integrity':fold_checks,'metrics':actual,'findings':meta},ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.project,args.out)
