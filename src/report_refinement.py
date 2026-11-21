"""Verify archived refinement predictions and produce fixed-slice diagnostics and a report."""
import json
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import inputs, truth
from bert_experiment import scored_predictions
from refine_experiment import OUT
from window_experiment import extract_scores

NAMES={'baseline_single':'原方案单模型','baseline_ensemble':'原方案三模型平均',
       'candidate_single':'方面标记单模型','candidate_ensemble':'方面标记三模型平均'}
METRICS=['Valance MAE','Valance PCC','Arousal MAE','Arousal PCC']


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(str(v) for v in row)+'|' for row in rows])


def main():
    metrics=pd.read_csv(OUT/'final_metrics.csv')
    cases=pd.read_csv(OUT/'test_cases.csv',keep_default_na=False)
    choice=json.loads((OUT/'final_selection.json').read_text(encoding='utf-8'))
    selected=choice['selected_system']
    # Recompute the official scores using saved prediction files only, with no training/model loading.
    for split,tag in [('dev','Validation'),('test','Testing')]:
        rows=inputs(tag);gold=truth(tag)
        for system in NAMES:
            pred=pd.read_csv(OUT/'systems'/system/f'{split}_predictions.txt',sep=' ',keep_default_na=False)
            values=extract_scores(pred,rows)
            recomputed=scored_predictions(rows,values,gold)[1]
            stored=metrics[(metrics.system==system)&(metrics.split==split)].iloc[0]
            assert all(np.isclose(recomputed[k],stored[k],atol=1e-9,rtol=0) for k in METRICS)
            if split=='test':
                assert cases[['id','text','aspect']].equals(rows[['id','text','aspect']])
                assert np.allclose(cases[[system+'_V',system+'_A']],values,atol=1e-9,rtol=0)
                assert np.allclose(cases[['true_V','true_A']],extract_scores(gold,rows),atol=1e-9,rtol=0)
    masks={
        '全部':np.ones(len(cases),dtype=bool),
        '同句多方面':cases.groupby('id').aspect.transform('nunique')>1,
        '极端V标签':(cases.true_V<=4)|(cases.true_V>=7.5),
        '高A标签':cases.true_A>=6.5,
        '含否定字词':cases.text.str.contains('不|没|无|難|难|非',regex=True),
    }
    slices=[]
    for label,mask in masks.items():
        g=cases.loc[mask]
        for system in NAMES:
            slices.append(dict(subset=label,n=len(g),system=system,
                               V_MAE=float((g[system+'_V']-g.true_V).abs().mean()),
                               A_MAE=float((g[system+'_A']-g.true_A).abs().mean())))
    pd.DataFrame(slices).to_csv(OUT/'slice_metrics.csv',index=False,encoding='utf-8-sig')
    pairs=[]
    for system in NAMES:
        for dim in ['V','A']:
            total=correct=tied=0
            for _,g in cases.groupby('id',sort=False):
                for a,b in combinations(g.index,2):
                    gap=cases.loc[a,'true_'+dim]-cases.loc[b,'true_'+dim]
                    if abs(gap)<1e-9:continue
                    delta=cases.loc[a,system+'_'+dim]-cases.loc[b,system+'_'+dim]
                    total+=1;correct+=gap*delta>0;tied+=abs(delta)<1e-9
            pairs.append(dict(system=system,dimension=dim,pairs=total,correct=int(correct),tied=int(tied),accuracy=correct/total))
    pd.DataFrame(pairs).to_csv(OUT/'aspect_order.csv',index=False,encoding='utf-8-sig')
    for system in NAMES:
        cases[system+'_mean_abs_error']=((cases[system+'_V']-cases.true_V).abs()+(cases[system+'_A']-cases.true_A).abs())/2
    cases['single_error_change']=cases.candidate_single_mean_abs_error-cases.baseline_single_mean_abs_error
    cases['selected_error_change']=cases[selected+'_mean_abs_error']-cases.baseline_single_mean_abs_error
    cases.to_csv(OUT/'case_changes.csv',index=False,encoding='utf-8-sig')
    seed=pd.read_csv(OUT/'per_seed_metrics.csv')
    summary=[]
    for system,g in seed[seed.split=='test'].groupby('system'):
        summary.append(dict(system=system,**{k+'_mean':g[k].mean() for k in METRICS},
                            **{k+'_sd':g[k].std(ddof=1) for k in METRICS}))
    pd.DataFrame(summary).to_csv(OUT/'seed_summary.csv',index=False,encoding='utf-8-sig')
    test=metrics[metrics.split=='test'].set_index('system').loc[list(NAMES)]
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':11})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    labels=['原方案\n单模型','原方案\n三模型平均','方面标记\n单模型','方面标记\n三模型平均']
    for ax,metric,title in zip(axes.flat,METRICS,['V-MAE 越低越好','V-PCC 越高越好','A-MAE 越低越好','A-PCC 越高越好']):
        bars=ax.bar(labels,test[metric],color=['#7E95AB','#A8B7C5','#267E73','#72AAA0'],width=.58)
        ax.bar_label(bars,fmt='%.4f',padding=3)
        ax.set(title=title,ylim=(0,1 if 'PCC' in metric else float(test[metric].max())*1.25))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('固定方案的测试结果对照  单模型为种子 42',fontsize=15)
    fig.savefig(OUT/'metrics_comparison.png',dpi=180);plt.close(fig)
    dev=metrics[metrics.split=='dev']
    report=['# 方面标记改进实验','',
      f'开发集预先选定的演示方案为 **{NAMES[selected]}**。所有测试结果均在锁定这一决定后计算。原演示入口仍使用旧模型，新入口为《改进演示.cmd》。','',
      '## 改了什么','',
      '在评论中的目标方面前后加入 [unused1] 和 [unused2]，明确标出当前评价对象。模型仍输入“方面词、完整评论”，沿用中文 BERT 和两个线性回归头。标记复用原词表并注册为特殊词元，训练与推断使用相同规则。多个命中全部标记，NULL 或未匹配的方面保留原文。','',
      '另外固定比较了输入长度 128 的普通句对，以及带标记的训练评分重加权方案。诊断发现原训练/开发最大长度为 62/47，标记后为 68/49，均未超过原上限 80；单纯加长输入完全复现原 seed42 的开发指标，没有信息增益。重加权按训练标签每维 1 分宽分箱，使用逆平方根频率，先裁剪至 0.5～3，再归一化为每维均值 1。它未在开发集胜过普通标记方案，未进入后续测试。','',
      '## 训练和选模','',
      '各新模型训练 3 轮，batch 32，AdamW 学习率 2e-5，权重衰减 0.01，预热 10%，梯度裁剪 1。按开发集 V/A 平均 MAE 选检查点。先在 seed42 筛选固定候选，再对方面标记补种子 17、2026。最后比较原方案与标记方案的单模型及三种子等权平均，只依据开发集选定演示方案。集成先使用各成员正式的两位小数分数求平均，再统一保留两位小数，双方口径一致。','',
      table(['系统','开发集平均MAE'],[[NAMES[r.system],f'{r.mean_MAE:.6f}'] for r in dev.itertuples()]),'',
      '## 测试结果','',table(['系统','V-MAE','V-PCC','A-MAE','A-PCC','两维平均MAE'],
        [[NAMES[name],*[f'{row[k]:.4f}' for k in METRICS],f'{row.mean_MAE:.4f}'] for name,row in test.iterrows()]),'',
      '标记单模型的 V-MAE 降低，A-MAE 变化很小，V-PCC 略有下降。新方案不是四项指标全面领先。标记三模型平均的测试误差更低，但其开发成绩不如标记单模型，因此保留为对照，没有按测试成绩更换演示方案。三模型平均需要运行三份模型，成本高于单模型。','',
      '## 三种子稳定性','',table(['方案','种子','V-MAE','A-MAE'],[[{'baseline':'原方案','candidate':'方面标记'}[r.system],r.seed,f'{seed.loc[i,"Valance MAE"]:.4f}',f'{seed.loc[i,"Arousal MAE"]:.4f}'] for i,r in seed[seed.split=='test'].iterrows()]),'',
      '各次结果和样本标准差见 per_seed_metrics.csv、seed_summary.csv。种子只有三个，不能根据一次最好结果声称稳定全面提升。','',
      '## 困难子集','',
      '规则在新测试前固定：极端 V 为 V≤4 或 V≥7.5，高 A 为 A≥6.5，多方面按同句方面数大于 1；含否定字词只检查“不、没、无、難、难、非”，并非人工标注的语义否定。下表比较原单模型与开发集选定方案。','']
    slice_rows=[]
    for label in masks:
        a=next(r for r in slices if r['subset']==label and r['system']=='baseline_single')
        b=next(r for r in slices if r['subset']==label and r['system']==selected)
        slice_rows.append([label,a['n'],f"{a['V_MAE']:.4f}",f"{b['V_MAE']:.4f}",f"{a['A_MAE']:.4f}",f"{b['A_MAE']:.4f}"])
    report.extend([table(['子集','方面数','原V误差','选定V误差','原A误差','选定A误差'],slice_rows),'',
        '这次改善主要体现在 V 误差与方面区分。高 A 子集的两维误差反而上升，因此尚未解决强烈情绪问题。不能只展示成功案例。','',
        '同句方面排序也作了补充诊断：人工分数不同的方面对纳入分母，预测打平不算正确。原单模型与标记单模型的 V 正确率分别为 383/472（81.14%）和 400/472（84.75%），A 为 363/469（77.40%）和 370/469（78.89%）。完整结果见 aspect_order.csv。','',
        '## 原有展示案例的变化',''])
    for sid,asp in [('R2103:S002','服务态度'),('R0060:S023','味道'),('R2120:S005','土鸡')]:
        r=cases[(cases.id==sid)&(cases.aspect==asp)].iloc[0]
        report.extend([f'**{sid} / {asp}**：{r.text}',
            f'人工 V/A {r.true_V:.2f}/{r.true_A:.2f}；原模型 {r.baseline_single_V:.2f}/{r.baseline_single_A:.2f}；选定方案 {r[selected+"_V"]:.2f}/{r[selected+"_A"]:.2f}。两维平均误差变化为 {r.selected_error_change:+.3f} 分（负数为改善）。',''])
    report.extend(['## 解释边界与复现','',
      '开发集仅 100 句、115 方面，且旧测试集已用于前期错误分析。这是固定候选的探索性改进，仍需独立的新测试集验证。方面标记帮助明确对象，但不能保证理解隐含意见或强烈情绪；各子集变化应逐项报告。测试子集没有直接给出“隐含评价”人工标签，因此不声称已经解决该类问题。','',
      '运行 python src/report_refinement.py 可用已保存预测复核四系统开发/测试指标并重建图表和文字，不重新训练。训练和评估源码分别为 refine_experiment.py、evaluate_refinement.py；实验计划、候选筛选和最终选模记录分别为 experiment_plan.json、candidate_selection.json、final_selection.json。',''])
    (OUT/'改进实验说明.md').write_text('\n'.join(report),encoding='utf-8')
    (OUT/'report_verification.json').write_text(json.dumps(dict(official_prediction_files_verified=8,case_table_rows_verified=len(cases),selected_system=selected),ensure_ascii=False,indent=2),encoding='utf-8')
    print(test.to_string());print(pd.DataFrame(slice_rows).to_string(index=False));print('Report and chart saved to',OUT)


if __name__=='__main__':main()
