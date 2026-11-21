#!/usr/bin/env python3
"""Aggregate only COMPLETED human-reviewed OOF cases, never machine tags.
Usage: python scripts/summarize_manual_review.py --input <human-edited CSV> --output <summary CSV>
"""
import argparse, csv
from pathlib import Path
from collections import defaultdict

def f(v):return float(v)
def main(input_path,output_path):
    with input_path.open(encoding='utf-8-sig',newline='') as h:records=list(csv.DictReader(h))
    groups=defaultdict(list)
    completed=0
    for row in records:
        manual=(row.get('人工确认标签') or '').strip()
        explain=(row.get('人工解释') or '').strip()
        if not (manual and explain):continue
        completed+=1
        gain=float(row['gain_positive_better']) if 'gain_positive_better' in row else float(row['平均误差减少_正数为改善'])
        for category in set(x.strip() for x in manual.replace('|','；').replace(';','；').split('；') if x.strip()):
            groups[category].append(gain)
    rows=[]
    for k,gains in sorted(groups.items()):
        rows.append({'人工确认类别':k,'已复核实例数':len(gains),'误差改善实例数':sum(v>1e-9 for v in gains),
                     '退步实例数':sum(v<-1e-9 for v in gains),'持平实例数':sum(abs(v)<=1e-9 for v in gains),
                     '平均误差改善值_正数更好':sum(gains)/len(gains),'备注':'类别可重叠，样本计数不可相加'})
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=['人工确认类别','已复核实例数','误差改善实例数','退步实例数','持平实例数','平均误差改善值_正数更好','备注']);w.writeheader();w.writerows(rows)
    print(f'总案例 {len(records)}，人工解释完整案例 {completed}，类别 {len(groups)}；输出 {output_path}')
    if completed==0:print('当前无人工作业填写：不能生成真实的人工错误分类结论。')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();main(a.input,a.output)
