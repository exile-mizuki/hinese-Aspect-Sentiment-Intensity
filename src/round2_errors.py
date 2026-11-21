"""Auditable rule-based diagnostic slices; tags are cues, never human annotations."""
import json
import re
from itertools import combinations

import numpy as np
import pandas as pd
from main import O
from bert_experiment import write_json

OUT = O/'round2'
RULES = {
    '否定线索': r'没有|并非|不是|不太|不够|不会|不能|没|不|未|无',
    '转折线索': r'但是|不过|然而|可是|虽然|却|但',
    '程度强调线索': r'非常|超级|极其|十分|完全|实在|特别|太|超|最|[!！]+',
    '隐含评价候选': r'排队|等了|等很久|放很久|放了很久|吃不完|吃完|剩下|见底|回访|再来|第二次|第三次|不会再|不再|没再|冷掉|凉掉',
    '指代线索': r'这个|那个|这些|那些|它们|他们|前者|后者|这家|那家',
}
EXCLUSIONS = r'不但|不仅|不只|不得不|没想到|无论|不管'


def cues(text, aspect, multi):
    evidence = []
    excluded = [(m.start(),m.end()) for m in re.finditer(EXCLUSIONS,text)]
    for tag, pattern in RULES.items():
        for m in re.finditer(pattern,text):
            if tag=='否定线索' and any(a <= m.start() < b for a,b in excluded): continue
            if tag=='转折线索' and m.group()=='但' and any(a <= m.start() < b for a,b in excluded): continue
            evidence.append(dict(tag=tag, start=m.start(), end=m.end(), text=m.group()))
    tags = sorted({e['tag'] for e in evidence})
    if multi: tags.append('同句多方面')
    if aspect=='NULL': tags.append('隐式方面NULL')
    elif aspect not in text: tags.append('方面未字面匹配')
    elif text.count(aspect)>1: tags.append('方面重复出现')
    for e in evidence: assert text[e['start']:e['end']]==e['text']
    return tags, evidence


def load_comparison(split):
    deployment = json.loads((OUT/'deployment.json').read_text(encoding='utf-8'))
    candidate = deployment['candidate']
    before = pd.read_csv(OUT/'marked_l1'/'full'/f'{split}_cases.csv', keep_default_na=False)
    after = pd.read_csv(OUT/candidate/'full'/f'{split}_cases.csv', keep_default_na=False)
    keys = ['id','text','aspect']
    assert before[keys].equals(after[keys])
    assert np.array_equal(before[['true_V','true_A']], after[['true_V','true_A']])
    frame = before.rename(columns={'pred_V':'before_V','pred_A':'before_A'})
    frame[['after_V','after_A']] = after[['pred_V','pred_A']]
    return frame


def annotate(frame):
    frame = frame.copy()
    multi = frame.groupby('id').aspect.transform('size').gt(1)
    tags, evidence = [], []
    for r, m in zip(frame.itertuples(), multi):
        t, e = cues(r.text, r.aspect, m)
        tags.append('|'.join(t))
        evidence.append(json.dumps(e, ensure_ascii=False))
    frame['规则标签'] = tags
    frame['原文线索_位置从0开始'] = evidence
    frame['高A真值切片'] = frame.true_A >= 6.5
    for method in ('before','after'):
        for dim in ('V','A'):
            frame[f'{method}_{dim}_abs_error'] = (frame[f'{method}_{dim}']-frame['true_'+dim]).abs()
        frame[method+'_mean_abs_error'] = frame[[method+'_V_abs_error',method+'_A_abs_error']].mean(axis=1)
    frame['平均误差减少_正数为改善'] = frame.before_mean_abs_error-frame.after_mean_abs_error
    return frame


def slices(frame, split):
    labels = sorted({s for t in frame['规则标签'] for s in t.split('|') if s})
    masks = [('全部样本',pd.Series(True,index=frame.index))]
    masks += [(label,frame['规则标签'].map(lambda s:label in s.split('|'))) for label in labels]
    masks += [('高A真值>=6.5',frame.true_A>=6.5),('低A真值<5',frame.true_A<5)]
    rows = []
    for tag, mask in masks:
        part = frame[mask]
        r = dict(split=split, slice=tag, n=len(part), sentences=part.id.nunique())
        for method in ('before','after'):
            for dim in ('V','A'):
                r[f'{method}_{dim}_MAE'] = part[f'{method}_{dim}_abs_error'].mean()
        rows.append(r)
    return rows


def pair_order(frame, split):
    rows = []
    for dim in ('V','A'):
        for method in ('before','after'):
            correct = ties = total = 0
            for _, part in frame.groupby('id', sort=False):
                for (y1,p1),(y2,p2) in combinations(part[['true_'+dim,method+'_'+dim]].to_numpy(),2):
                    if y1==y2: continue
                    total += 1
                    correct += int((y1-y2)*(p1-p2)>0)
                    ties += int(p1==p2)
            rows.append(dict(split=split,dimension=dim,method=method,pairs=total,correct=correct,ties=ties,
                             accuracy=correct/total if total else None))
    return rows


def review_sample(frame):
    chosen = []
    used = set()
    labels = list(RULES)+['同句多方面','高A真值>=6.5','全部样本']
    for label in labels:
        if label=='高A真值>=6.5': subset=frame[frame.true_A>=6.5]
        elif label=='全部样本': subset=frame
        else: subset=frame[frame['规则标签'].map(lambda s:label in s.split('|'))]
        for direction, column, ascending in [('改善案例','平均误差减少_正数为改善',False),
                                              ('退步案例','平均误差减少_正数为改善',True),
                                              ('仍有较大误差','after_mean_abs_error',False)]:
            subset2=subset.sort_values(column,ascending=ascending,kind='stable')
            if direction=='改善案例': subset2=subset2[subset2[column]>0]
            if direction=='退步案例': subset2=subset2[subset2[column]<0]
            for idx,r in subset2.iterrows():
                key=(r.id,r.aspect)
                if key in used: continue
                used.add(key)
                item=r.to_dict()
                item['抽样原因']=label+' / '+direction
                for col in ['人工确认标签','人工解释','复核人','复核日期']: item[col]=''
                chosen.append(item)
                break
    return pd.DataFrame(chosen)


def check_rules():
    # Meaningful regression checks for common Chinese exclusion phrases and evidence spans.
    for text in ('不仅好吃，而且便宜','不但好吃','不得不说好吃','没想到这么好吃','无论如何都好吃'):
        tags,_=cues(text,'菜',False)
        assert '否定线索' not in tags, text
    tags,e=cues('汤不太好喝，但是服务非常好！','汤',True)
    assert {'否定线索','转折线索','程度强调线索','同句多方面'} <= set(tags)
    assert all('汤不太好喝，但是服务非常好！'[x['start']:x['end']]==x['text'] for x in e)
    assert '隐式方面NULL' in cues('还不错','NULL',False)[0]


def main():
    check_rules()
    summary, pairs = [], []
    for split in ('dev','test'):
        frame=annotate(load_comparison(split))
        frame.to_csv(OUT/f'{split}_diagnostic_cases.csv',index=False,encoding='utf-8-sig')
        summary += slices(frame,split)
        pairs += pair_order(frame,split)
        if split=='test':
            review=review_sample(frame)
            review.to_csv(OUT/'人工复核案例_待填写.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(OUT/'diagnostic_slice_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pairs).to_csv(OUT/'aspect_pair_order.csv',index=False)
    write_json(OUT/'taxonomy.json',dict(patterns=RULES,negation_exclusions=EXCLUSIONS,
        evidence_offsets='Python Unicode character index, start inclusive / end exclusive',
        sentence_level_cues=True,labels_are_human_annotations=False,
        implicit_evaluation='candidate only: eating/waiting/revisit words cannot establish sentiment by themselves',
        complex_relation='multi-aspect, repeated aspect and coreference cues only; no dependency parser or relation gold labels',
        diagnosis='rule membership is not a causal explanation of model error',
        overlapping_slices=True,human_review_rows=len(review),human_review_completed=False,
        rule_regression_checks='passed',comparison='matched full-training baseline vs CV-chosen candidate/probe'))
    print('Error diagnostics saved; manual review rows:',len(review))


if __name__=='__main__': main()
