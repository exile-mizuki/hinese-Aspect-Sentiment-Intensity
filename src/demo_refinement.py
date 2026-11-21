"""Interactive original-versus-refined demo using the system frozen on development data."""
import argparse
import json
import os
import re
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from bert_experiment import IntensityModel, infer_rows
from main import save_pred
from refine_experiment import OUT, ROOT, infer, load_model, tokenizer_for
from window_experiment import extract_scores


def original(seed):
    path=ROOT/'checkpoints'/'bert_aspect'/'best.pt' if seed==42 else ROOT/'checkpoints'/'repeats'/f'seed_{seed}'/'bert_aspect'/'best.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=IntensityModel(AutoModel.from_config(AutoConfig.from_pretrained(ROOT/'demo_tokenizer',local_files_only=True)))
    model.load_state_dict(state['model'])
    return model


def predict(rows, model, variant, tokenizer):
    values=infer_rows(model,rows,'aspect',tokenizer,80,32,torch.device('cpu')) if variant=='baseline' else infer(model,rows,variant,tokenizer,torch.device('cpu'))
    return extract_scores(save_pred(rows,values,None),rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text')
    parser.add_argument('--aspects',nargs='+')
    parser.add_argument('--output',type=Path,help='save noninteractive demonstration predictions as JSON')
    args=parser.parse_args()
    if bool(args.text)!=bool(args.aspects):parser.error('--text 和 --aspects 必须一起提供')
    torch.set_num_threads(3)
    selected=json.loads((OUT/'final_selection.json').read_text(encoding='utf-8'))
    system=selected['selected_system']; variant=selected['selected_variant']
    print('正在加载本地模型，首次启动请稍等。',flush=True)
    base_tokenizer=AutoTokenizer.from_pretrained(ROOT/'demo_tokenizer',local_files_only=True)
    base=original(42)
    selected_seeds=selected['seeds'] if 'ensemble' in system else [42]
    if system.startswith('candidate'):
        selected_models=[load_model(variant,s,torch.device('cpu')) for s in selected_seeds]
        selected_tokenizer=tokenizer_for(variant)
        selected_variant=variant
    else:
        selected_models=[base if s==42 else original(s) for s in selected_seeds]
        selected_tokenizer=base_tokenizer; selected_variant='baseline'
    label=('方面标记' if selected_variant!='baseline' else '原方案')+('三模型平均' if len(selected_models)>1 else '单模型')
    print(f'当前开发集选定方案：{label}。V 为愉悦度，A 为唤醒度，范围均为 1～9。',flush=True)
    while True:
        if args.text:
            text,aspects=args.text,args.aspects
        else:
            try:
                text=input('\n输入评论（直接回车退出）：').strip()
                if not text:break
                aspects=[a.strip() for a in re.split('[,，]',input('输入方面词，多个方面用逗号隔开：')) if a.strip()]
            except EOFError:break
            if not aspects:
                print('请至少填写一个方面。');continue
        aspects=list(dict.fromkeys(aspects))
        rows=pd.DataFrame(dict(id=['demo']*len(aspects),text=text,aspect=aspects))
        old=predict(rows,base,'baseline',base_tokenizer)
        members=[predict(rows,m,selected_variant,selected_tokenizer) for m in selected_models]
        new=np.round(np.mean(members,axis=0),2)
        print('\n评论：'+text)
        print('方面\t原模型 V/A\t选定方案 V/A')
        records=[]
        for aspect,a,b in zip(aspects,old,new):
            print(f'{aspect}\t{a[0]:.2f}/{a[1]:.2f}\t{b[0]:.2f}/{b[1]:.2f}')
            records.append(dict(aspect=aspect,original_V=float(a[0]),original_A=float(a[1]),selected_V=float(b[0]),selected_A=float(b[1])))
        print('以上为模型预测；自拟评论没有人工真值，不能据此计算准确率。')
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(dict(text=text,selected_system=system,predictions=records),ensure_ascii=False,indent=2),encoding='utf-8')
        if args.text:break


if __name__=='__main__':main()
