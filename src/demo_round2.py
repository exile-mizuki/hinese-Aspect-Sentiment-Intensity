"""Interactive comparison: previously selected marked model vs round2 candidate."""
import argparse
import json
import re
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel
from refine_experiment import ROOT, infer, tokenizer_for, load_model
from round2_experiment import ArousalModel, OUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text')
    parser.add_argument('--aspects',nargs='+')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    if args.text and not args.aspects: parser.error('--text requires --aspects')
    torch.set_num_threads(3)
    device=torch.device('cpu')
    deployment=json.loads((OUT/'deployment.json').read_text(encoding='utf-8'))
    if not args.json:
        print('正在加载两个本地模型，请稍候……',flush=True)
    old=load_model('marked',42,device)
    state=torch.load(ROOT/'checkpoints'/'round2'/'candidate.pt',map_location='cpu',weights_only=True)
    candidate=ArousalModel(AutoModel.from_config(AutoConfig.from_pretrained(ROOT/'demo_tokenizer',local_files_only=True)))
    candidate.load_state_dict(state['model'])
    assert state['method']==deployment['candidate']
    tok=tokenizer_for('marked')
    if not args.json:
        print('V：越高越正面；A：越高情绪越激烈；量表均为 1–9 分。')
        print('旧模型：上轮方面标记模型。候选：'+deployment['candidate'])
        print('候选采用门槛：'+('通过' if deployment['accepted'] else '未全部通过，作为研究对照'))
        print('请输入餐饮评论及方面词。方面词应出现在评论中；多个方面用逗号分隔。空评论退出。')
    while True:
        try:
            text=args.text if args.text is not None else input('\n评论：').strip()
            if not text: break
            aspects=args.aspects if args.text is not None else [s.strip() for s in re.split('[,，;；]',input('方面：')) if s.strip()]
        except (EOFError,KeyboardInterrupt): break
        if not aspects:
            print('请至少填写一个方面词。');continue
        missing=[a for a in aspects if a!='NULL' and a not in text]
        if missing:
            if args.text: raise ValueError('方面词不在评论中：'+','.join(missing))
            print('方面词不在评论中：'+','.join(missing)+'。请重新输入。');continue
        rows=pd.DataFrame([dict(id=f'demo:{i}',text=text,aspect=a) for i,a in enumerate(aspects)])
        before=infer(old,rows,'marked',tok,device).clip(1,9).round(2)
        after=infer(candidate,rows,'marked',tok,device).clip(1,9).round(2)
        records=[]
        for a,b,c in zip(aspects,before,after):
            records.append(dict(aspect=a,old_V=float(b[0]),old_A=float(b[1]),candidate_V=float(c[0]),candidate_A=float(c[1])))
            if not args.json: print(f'{a}  旧模型 V={b[0]:.2f} A={b[1]:.2f}  |  本轮候选 V={c[0]:.2f} A={c[1]:.2f}')
        if args.json: print(json.dumps(dict(text=text,accepted=deployment['accepted'],method=state['method'],predictions=records),ensure_ascii=False))
        if args.text is not None:break


if __name__=='__main__':main()
