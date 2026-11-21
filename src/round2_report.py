"""Generate the report from saved CV and official-set predictions, without selecting on test."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import O, P

OUT=O/'round2'
LABELS={'marked_l1':'同条件 L1 对照','arousal_mse':'A 改用 MSE','arousal_mse_aux':'A 的 MSE + 辅助分类'}


def md_table(frame):
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def main():
    cv=pd.read_csv(OUT/'cv_pooled_metrics.csv')
    folds=pd.read_csv(OUT/'cv_fold_metrics.csv')
    official=pd.read_csv(OUT/'official_metrics.csv')
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    deployment=json.loads((OUT/'deployment.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'fold_audit.json').read_text(encoding='utf-8'))
    taxonomy=json.loads((OUT/'taxonomy.json').read_text(encoding='utf-8'))
    candidate=deployment['candidate']
    stats=folds.groupby('method')[['V_MAE','A_MAE','mean_MAE','high_A_MAE']].agg(['mean','std'])
    stats.columns=['_'.join(c) for c in stats.columns]
    stats.reset_index().to_csv(OUT/'cv_fold_mean_std.csv',index=False)
    show=cv[['method','V_MAE','A_MAE','mean_MAE','high_A_MAE']].copy()
    show['method']=show.method.map(LABELS)
    show=show.rename(columns={'method':'方法','mean_MAE':'V/A平均MAE','high_A_MAE':'高A的MAE'})
    for c in show.columns[1:]: show[c]=show[c].map(lambda x:f'{x:.4f}')
    show2=official[['method','split','V_MAE','A_MAE','mean_MAE','high_A_MAE']].copy()
    show2['method']=show2.method.map(LABELS)
    for c in show2.columns[2:]: show2[c]=show2[c].map(lambda x:f'{x:.4f}')
    old=pd.read_csv(O/'refinement'/'final_metrics.csv')
    # Existing published result is supplementary; never used to change this round's selection.
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    x=np.arange(len(cv))
    for ax,col,title in zip(axes,['A_MAE','high_A_MAE'],['训练集内部：全部 A 误差','训练集内部：高 A 误差（真值≥6.5）']):
        ax.bar(x,cv[col],color=['#6984A3','#32909B','#CE845A'])
        ax.set_xticks(x,[LABELS[m].replace(' + ','\n+ ') for m in cv.method],fontsize=9)
        ax.set_ylabel('MAE（越低越好）'); ax.set_title(title)
        ax.set_ylim(0,cv[col].max()*1.22)
        for i,v in enumerate(cv[col]): ax.text(i,v+.005,f'{v:.4f}',ha='center',fontsize=10)
    fig.savefig(OUT/'cv_arousal_comparison.png',dpi=180);plt.close(fig)
    sliced=pd.read_csv(OUT/'diagnostic_slice_metrics.csv')
    sub=sliced[(sliced.split=='test')&(sliced.n>=20)].copy()
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    x=np.arange(len(sub))
    ax.bar(x-.18,sub.before_A_MAE,.36,label='同条件 L1 对照',color='#6984A3')
    ax.bar(x+.18,sub.after_A_MAE,.36,label=LABELS[candidate],color='#32909B')
    ax.set_xticks(x,[f'{r.slice}\n(n={r.n})' for r in sub.itertuples()],rotation=28,ha='right',fontsize=9)
    ax.set_ylabel('A MAE（越低越好）');ax.set_title('测试集诊断切片：规则可重叠，不是人工错误分类');ax.legend()
    fig.savefig(OUT/'diagnostic_arousal_slices.png',dpi=180);plt.close(fig)
    cvbase=cv.iloc[0]; cvnew=cv[cv.method==candidate].iloc[0]
    dev=official[official.split=='dev'].set_index('method')
    test=official[official.split=='test'].set_index('method')
    accepted=deployment['accepted']
    decision=('候选通过训练集交叉验证及开发集门槛，可作为本轮新模型使用。' if accepted else
              '候选未通过预先约定的全部门槛，保留已有方面标记模型作为常规演示。新候选保留作研究对照，不宣称全面改进。')
    report=f'''# 三项深化实验结果

{decision}

## 1. 高强度情绪预测

A 是唤醒度，表示情绪激活程度；V 是效价，表示偏负面或正面。高 A 不等于正面，也不等于句中一定有程度副词。本轮高 A 诊断阈值固定为真值 A≥6.5，误差单位为 1–9 分量表上的分数。

本轮对照固定同一编码器、方面标记、种子 42、学习率 2e-5、批量 32、训练 3 轮。每一折使用第 3 轮预测，不用该折挑选最佳轮次。改动顺序为 A 的 L1→MSE→再加 0.1 权重的三档辅助分类。

{md_table(show)}

候选 `{candidate}` 相对本轮同条件对照：训练集折外 A MAE {cvbase.A_MAE:.4f}→{cvnew.A_MAE:.4f}，高 A MAE {cvbase.high_A_MAE:.4f}→{cvnew.high_A_MAE:.4f}。这反映当前一次分组划分和种子下的实测结果。

## 2. 更可靠的验证

合并同评论 ID 及完全相同正文后，{audit['n']} 条方面样本形成 {audit['groups']} 个连通组。三折的训练/留出评论重叠数和正文重叠数均为 0。每条训练样本获得一次折外预测。`fold_assignments.csv` 保存逐条划分，`fold_audit.json` 保存交叉检查。

`cv_fold_metrics.csv` 是各折指标；`cv_fold_mean_std.csv` 是三折未加权均值与样本标准差；上表则把全部折外预测合并评分，两者因样本数不同会有少量差异。三折标准差不是统计显著性，也不能替代多个随机种子。未进行近重复正文检测。

预先固定的采用门槛：A 总体 MAE 和高 A MAE 都降低，且 V/A 平均 MAE 增幅不超过 0.003；多个候选满足时，选 A MAE 最小的。完成交叉验证选择后再进行全训练集训练和官方开发集确认。开发集应用同一门槛。

## 3. 官方数据上的补充对照

{md_table(show2)}

dev 是开发集，test 是测试集。候选开发集高 A MAE 为 {dev.loc[candidate,'high_A_MAE']:.4f}，同条件对照为 {dev.loc['marked_l1','high_A_MAE']:.4f}；测试集分别为 {test.loc[candidate,'high_A_MAE']:.4f} 和 {test.loc['marked_l1','high_A_MAE']:.4f}。这些数值未用于事后修改方法或阈值。

注意：本轮“before”是重新训练的同条件 3 轮 L1 对照；之前日常演示使用的是开发集选出的第 2 轮模型。两者不是同一个模型。之前已查看过官方测试集，因此本轮测试比较属于探索性结果，不称为新的独立盲测。旧成果原样保存在 `results/refinement`，不覆盖、不混用数字。

## 4. 错误分类和人工复核

增加否定、转折、程度强调、隐含评价候选、指代、同句多方面、方面重复出现等诊断线索，保存每处命中文字及原文位置。排除“不但、不得不、没想到”等常见假否定触发。`test_diagnostic_cases.csv` 包含全部预测、真值、误差和标签；`diagnostic_slice_metrics.csv` 汇总各类指标；`aspect_pair_order.csv` 检查同句不同方面的相对强度顺序。

`人工复核案例_待填写.csv` 提供 {taxonomy['human_review_rows']} 条案例，兼顾改善、退步与仍有较大误差的样本。人工标签、解释、复核人和日期留空，由组员阅读原文填写。规则类别可以重叠，不能把各类样本数相加当作总数；关键词只提供线索，不能自动认定隐含情绪、反讽或错误原因。尚未完成真实人工复核。

## 使用与复现

- 查看本说明和两张 PNG 图即可了解本轮结论，无需重新训练。
- `三项改进对比演示.cmd` 可输入自己的评论及方面词，对比已有演示模型与本轮候选。候选是否通过采用门槛会显示在终端。
- `复核三项改进.cmd` 根据已保存预测重新核对指标、数据划分和规则证据，不进行训练。
- 若要重跑训练，先备份本轮 `results/round2`，并确保本地 `模型缓存/chinese-bert-wwm-ext` 存在。依次运行 `python src/round2_experiment.py cv`、`python src/round2_experiment.py full`、`python src/round2_errors.py`、`python src/round2_report.py`。CV 命令会复用已完成折的预测；更改方案应使用新的输出目录，避免混入旧结果。
- 训练使用原有 test1 环境和 NVIDIA GPU；演示与复核不需要联网，已安装原项目依赖的电脑无需新增安装。

来源和方法区别见项目根目录《本轮方法来源.md》。本轮没有进行领域继续预训练、近重复去重或真实人工复核，报告中不将这些列为已完成工作。
'''
    (OUT/'三项深化实验报告.md').write_text(report,encoding='utf-8')
    guide=f'''# 三项改进版使用说明

{decision}

1. **先看结果**：打开 `results/round2/三项深化实验报告.md`，两张 PNG 图可用于汇报材料。
2. **现场演示**：双击 `三项改进对比演示.cmd`，输入一条餐饮评论，再输入要评价的方面词；多个方面用英文逗号分隔。输入空评论退出。输出 V 越高越正面，A 越高情绪越激烈，二者均为 1–9 分。
3. **改评论的位置**：直接在演示窗口的“评论”提示后输入，无需修改 Python 文件。例如“汤非常难喝，但是服务很好。”，方面填写“汤,服务”。这些是演示句，没有人工真值，不能用它们证明模型准确率。
4. **核对结果**：双击 `复核三项改进.cmd`，看到全部检查通过后截图；它不训练模型。
5. **分给组员的任务**：人工阅读 `results/round2/人工复核案例_待填写.csv`，填写标签、原因、复核人、日期。没有依据时写“无法判断”，不要把自动线索直接当成人工结论。

本轮候选：`{candidate}`。内部验证采用：`{selection['selected']}`；最终门槛通过：{'是' if accepted else '否'}。现有 `改进演示.cmd` 保持原用途；新的对比演示会明确标注实验状态。

本轮对照重新训练了 3 轮，与旧演示第 2 轮模型不同。旧成绩在 `results/refinement`；新成绩在 `results/round2`。测试数据已在之前研究中查看，因此新测试指标是探索性对比，不能称为全新盲测。

启动器先寻找当前目录的 `.venv`，再寻找本机 `anaconda3/envs/test1`，最后尝试 PATH 中的 Python。组员电脑需要安装原有依赖，并解压整个压缩包后运行，不要直接在压缩软件内点启动器。
'''
    (P/'三项改进版使用说明.md').write_text(guide,encoding='utf-8')
    print('Report and figures generated')


if __name__=='__main__':main()
