"""Build a local, evidence-linked study report; no training or official holdout access."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backbone_study_v1 import ROOT,OUT,PROJECT,manifest,metrics,cluster_ci
from evaluate_target_balance_v1 import load_predictions,sha

def load_existing(folder,seed,m,h):
    p,c,f,w=load_predictions(folder,m,h,sha(ROOT/'validation/proposed_3fold_manifest.csv'),seed)
    return np.abs(p[['pred_V','pred_A']].to_numpy(float)-m[['V','A']].to_numpy(float)),w

def main():
    m,h=manifest(); records=[]; warnings=[]; errors={}
    for name,path in [('baseline',ROOT/'training_v2/marked_l1'),('EmoBank_E2',ROOT/'emobank_transfer_v1/e2_sighan_cv/training_v2/marked_l1')]:
        for seed in [42,43,44]:
            error,w=load_existing(path,seed,m,h);warnings+=w;errors[name,seed]=error
            values=metrics(error,m)
            records.append({'model':name,'seed':seed,**{f'{group}_{metric}':score for group,v in values.items() for metric,score in v.items() if metric!='n'}})
    b=np.mean([errors['baseline',s] for s in [42,43,44]],0)
    e=np.mean([errors['EmoBank_E2',s] for s in [42,43,44]],0)
    d=e-b
    e2={'baseline':metrics(b,m),'candidate':metrics(e,m),'overall_group_bootstrap_95ci':cluster_ci(np.column_stack([d,d.mean(1)]),m.group.to_numpy(),2000,20261120),
        'overall_delta':float(d.mean()),'source_hash_warnings':warnings,'official_dev_test_read':False,'seeds':[42,43,44],
        'scope':'same auxiliary pretrained encoder, only downstream fine-tuning seeds varied; grouped OOF CI conditional on models'}
    (OUT/'emobank_e2_independent_audit.json').write_text(json.dumps(e2,ensure_ascii=False,indent=2),encoding='utf-8')
    status_path=OUT/'study_status.json'
    status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    lines=['本轮优化结果：预训练模型对比','',
           '沿用原始训练集7476个方面实例、同一冻结三折分组、3轮训练、batch size=8、L1损失。',
           '训练、候选初筛和本报告不读取官方Dev/Test标签。指标为折外预测MAE，越低越好。',
           '所有新候选默认不覆盖当前网页模型。','',
           '一、已完成方案复核',
           f"原始BERT三种子平均整体MAE：{b.mean():.6f}",
           f"EmoBank E2三种子平均整体MAE：{e.mean():.6f}；变化：{d.mean():+.6f}",
           'E2固定同一个辅助预训练编码器，只改变下游微调种子；不代表整个迁移流程的多种子稳定性。',
           '分层加权seed42整体MAE：0.305858→0.308340；极高A的A_MAE：0.820192→0.702500。',
           '分层加权有局部收益，但整体及多方面退步；暂不作为正式模型。','',
           '二、修复与验证',
           'MacBERT与RoBERTa的方面标记已注册为完整词元，不增加词表或修改下载文件。',
           '全部7476条训练实例的input_ids/token_type_ids/attention_mask均与原版一致。',
           '新训练入口使用真实项目路径，记录数据、模型、词表校验值，拒绝旧的错误目录。',
           '两份重复训练已停止，未改写原结果；完整日志和配置保留在本目录。','',
           '三、本轮实测结果']
    for r in status.get('screen',[]):
        name=r['model'];a=r['baseline']['OVERALL']['mean_MAE'];c=r['candidate']['OVERALL']['mean_MAE']
        lines.append(f"{name} seed42：整体MAE {a:.6f} → {c:.6f}；初筛{'通过' if r['screen_pass'] else '未通过'}。")
        lines.append(f"  多方面MAE {r['baseline']['MULTI']['mean_MAE']:.6f} → {r['candidate']['MULTI']['mean_MAE']:.6f}；高A的A_MAE {r['baseline']['HIGH_A']['A_MAE']:.6f} → {r['candidate']['HIGH_A']['A_MAE']:.6f}。")
    if 'confirmation' in status:
        r=status['confirmation'];a=r['baseline']['OVERALL']['mean_MAE'];c=r['candidate']['OVERALL']['mean_MAE']
        lines.append(f"{r['model']}三种子平均：整体MAE {a:.6f} → {c:.6f}；分组Bootstrap区间（候选减基线）{r['overall_group_bootstrap_95ci']['mean_MAE']}。")
        lines.append('训练集交叉验证确认结果：'+('通过，尚需独立开发验证与部署决定。' if r['cv_confirmation_pass'] else '未通过，不替换正式模型。'))
    elif status.get('screen'):
        lines.append('目前没有候选完成三种子确认；不得声称已稳定提高模型准确率。')
    else:lines.append('候选模型仍在训练，尚无完整比较结果。')
    lines += ['', '四、解释限制',
              '同一训练集已探索多个方案，本轮结果属于探索性比较。Bootstrap只反映固定模型下评论分组抽样变化，不包含重新训练或反复选型的全部不确定性。',
              '开发集和测试集此前已被查看，后续开发集确认也不能描述为全新盲测。',
              'E2及旧加权实验部分config没有源文件哈希；现已核对逐行来源与冻结分折，不能补造训练当时的哈希记录。',
              '', '证据：frozen_plan.json；各模型tokenizer_audit.json；study_status.json；*_comparison.json；各训练目录config/history/heldout_predictions；emobank_e2_independent_audit.json。']
    (OUT/'本轮优化结果.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(OUT/'已有模型逐种子复核.csv',index=False,encoding='utf-8-sig')
    print('\n'.join(lines))

if __name__=='__main__':main()
