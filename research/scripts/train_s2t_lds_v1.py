#!/usr/bin/env python3
# coding: utf-8
"""Training-only, 3-fold grouped-CV comparisons for SIGHAN dimABSA.

Two clean one-variable comparisons:
  - repeat_control vs s2t_augment: duplicated sample exposure vs S->T views.
  - lds_control vs lds_gaussian: identical training, with and without LDS-inspired continuous weighting.

Uses the VERIFIED manifest/training code, NEVER reads Dev/Test.
Place beside optimization_v2/scripts/train_ablation_v2.py.
No pretrained weights or private datasets are bundled with this script.
"""

from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_ablation_v2 as original
from training_utils_v2 import fold_weights, training_loss

EXPERIMENT_VERSION = 's2t-lds-v1.1-20261121-anchored-fix'
METHODS = ['repeat_control','s2t_augment','lds_control','lds_gaussian_s03']
LDS_BW = 0.30                # Label unit on [1,9]
LDS_BIN = 0.10
LDS_CAP = (0.67,1.50)       # Fixed before CV; not chosen on Dev


def continuous_lds_weights(values: np.ndarray, *, bandwidth=LDS_BW, grid_step=LDS_BIN):
    """LDS-inspired, fold-train-only smoothed histogram, weight ~ 1/sqrt(density).
    This is an adaptation, not a verbatim reproduction of the ICML code.
    Each V/A dimension is calculated and normalized independently.
    """
    y = np.asarray(values, dtype=np.float64)
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
    assert (y >= 1).all() and (y <= 9).all()
    centers = np.linspace(1.0, 9.0, 81)
    kernel_radius = int(math.ceil(4 * bandwidth / grid_step))
    offsets = np.arange(-kernel_radius, kernel_radius + 1)
    kernel = np.exp(-0.5 * ((offsets * grid_step) / bandwidth)**2)
    kernel /= kernel.sum()
    result = np.empty_like(y)
    histogram_info = []
    for dim in range(2):
        hist, _ = np.histogram(y[:,dim], bins=np.linspace(0.95,9.05,82))
        # Zero pad, *not* reflect: no artificial extra observations at the boundaries.
        smooth = np.convolve(hist.astype(np.float64), kernel, mode='same')
        per_item_density = np.interp(y[:,dim], centers, smooth)
        raw = 1.0 / np.sqrt(np.maximum(per_item_density,1e-6))
        raw /= raw.mean()
        raw = np.clip(raw, LDS_CAP[0], LDS_CAP[1])
        raw /= raw.mean()
        result[:,dim] = raw
        histogram_info.append({
            'dimension': ('V','A')[dim],
            'min_train_label': float(y[:,dim].min()),
            'max_train_label': float(y[:,dim].max()),
            'min_weight':float(raw.min()), 'max_weight':float(raw.max()),
            'mean_weight':float(raw.mean()),
        })
    return torch.tensor(result.astype(np.float32)), histogram_info


def build_training_views(train_rows, method):
    """ONLY training fold; never augment validation fold.
    s2t_augment reuses each row's *own* label; no synthetic new ratings.
    """
    if method not in METHODS:raise ValueError(method)
    orig = train_rows.reset_index(drop=True).copy()
    if method == 'repeat_control':
        return pd.concat([orig,orig],ignore_index=True), {'augmentation':'repeat same simplified rows','changed_views':0,'original_rows':len(orig)}
    if method != 's2t_augment':
        return orig, {'augmentation':'none','changed_views':0,'original_rows':len(orig)}
    from opencc import OpenCC
    converter = OpenCC('s2t')
    converted = orig.copy()
    converted['text'] = converted['text'].astype(str).map(converter.convert)
    converted['aspect'] = converted['aspect'].astype(str).map(converter.convert)
    has_aspect = orig.apply(lambda r:r.aspect not in ('','NULL') and r.aspect in r.text, axis=1).to_numpy()
    converted_ok = converted.apply(lambda r:r.aspect in r.text, axis=1).to_numpy()
    # OpenCC can use a phrase-dependent mapping when converting a full
    # sentence, so converting the aspect independently may produce a string
    # that is not a substring of the converted sentence. Repair only those
    # affected rows by converting [left context] + [target aspect] + [right
    # context] separately; this makes the aspect span unambiguous to encode().
    repaired_indices = np.flatnonzero(has_aspect & ~converted_ok)
    for i in repaired_indices:
        text_before = str(orig.at[i, 'text'])
        aspect_before = str(orig.at[i, 'aspect'])
        start = text_before.find(aspect_before)
        if start < 0:
            raise ValueError(f'Original aspect not present at index {i}')
        aspect_after = converter.convert(aspect_before)
        text_after = (
            converter.convert(text_before[:start])
            + aspect_after
            + converter.convert(text_before[start + len(aspect_before):])
        )
        if not aspect_after or aspect_after not in text_after:
            raise ValueError(f'Anchored conversion failed at index {i}')
        converted.at[i, 'text'] = text_after
        converted.at[i, 'aspect'] = aspect_after

    converted_ok = converted.apply(lambda r: r.aspect in r.text, axis=1).to_numpy()
    if not bool(np.all(converted_ok[has_aspect])):
        bad = np.flatnonzero(has_aspect & ~converted_ok)[:5].tolist()
        raise ValueError(f'Aspect still missing after anchored repair: {bad}')
    if len(repaired_indices):
        print('S2T ASPECT REPAIR', {'indices': repaired_indices.tolist(),
                                   'count': int(len(repaired_indices))}, flush=True)
    changed = int(((converted.text!=orig.text) | (converted.aspect!=orig.aspect)).sum())
    # Labels must be unchanged, and each original row contributes exactly one extra view.
    assert np.array_equal(converted[['V','A']].to_numpy(),orig[['V','A']].to_numpy())
    return pd.concat([orig,converted],ignore_index=True), {
        'augmentation':'Simplified plus Traditional views', 'changed_views':changed,
        'unchanged_views':len(orig)-changed, 'original_rows':len(orig),
        'matched_original_aspects':int(has_aspect.sum()),
        'anchored_repair_count':int(len(repaired_indices)),
    }


def train_one(train_rows,eval_rows,*,method,seed,model_path,project,epochs,batch_size,device):
    from transformers import AutoModel,set_seed,get_linear_schedule_with_warmup
    from refine_experiment import tokenizer_for,encode,infer
    from round2_experiment import ArousalModel
    from bert_experiment import token_batch
    if method not in METHODS:raise ValueError(method)
    set_seed(seed)
    train_view, info = build_training_views(train_rows,method)
    # Enforce the same original training rows; label histogram for LDS estimated only on fold Train.
    if method == 'lds_gaussian_s03':
        sample_weights,lds_info=continuous_lds_weights(train_rows[['V','A']].to_numpy(dtype=np.float32))
    else:
        sample_weights=torch.ones((len(train_view),2),dtype=torch.float32)
        lds_info=[]
    assert sample_weights.shape == (len(train_view),2)
    info.update({'lds':lds_info,'used_rows':len(train_view)})
    print('FOLD TRAIN AUDIT',method,seed,info,flush=True)
    tok=tokenizer_for('marked')
    tokens=encode(train_view,'marked',tok)
    y=torch.tensor(train_view[['V','A']].to_numpy(dtype=np.float32))
    labels,ce_weights,_,counts=fold_weights(y,False)
    encoder=AutoModel.from_pretrained(str(model_path),local_files_only=True)
    model=ArousalModel(encoder,means=y.mean(0)).to(device)
    steps=epochs*math.ceil(len(y)/batch_size)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(steps*.1),steps)
    amp=device.type=='cuda'
    dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    generator=torch.Generator().manual_seed(seed)
    history=[]
    for epoch in range(1,epochs+1):
        model.train(); start=time.perf_counter(); total=0.0
        for idx in torch.randperm(len(y),generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                values,aux=model(return_aux=True,**token_batch(tokens,idx,device))
                true_batch=y[idx].to(device)
                if method == 'lds_gaussian_s03':
                    delta=(values.float()-true_batch.float()).abs()
                    weights=sample_weights[idx].to(device)
                    loss=(delta*weights).mean()
                else:
                    loss=training_loss(values,true_batch,aux_logits=aux,aux_weight=0.,kind='l1')
            if not bool(torch.isfinite(loss).item()):raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            old=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=old:scheduler.step()
            total+=float(loss.detach().item())*len(idx)
        line={'epoch':epoch,'train_loss':total/len(y),'seconds':round(time.perf_counter()-start,2)}
        print('TRAIN',method,seed,line,flush=True);history.append(line)
    preds=infer(model,eval_rows,'marked',tok,device,batch_size=batch_size)
    if preds.shape!=(len(eval_rows),2):raise ValueError(f'Predictions wrong shape: {preds.shape}')
    del model,optimizer,scheduler,tokens,tok
    gc.collect()
    if amp:torch.cuda.empty_cache()
    return preds,history,counts.tolist(),ce_weights.tolist()


def signature(args,rows,manifest):
    s=_original_signature(args,rows,manifest)
    s.update({
        'independent_experiment_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'experiment':EXPERIMENT_VERSION,
        'lds_bandwidth':LDS_BW,'lds_bin_size':LDS_BIN,'lds_weight_clip':list(LDS_CAP),
        'augmentation_policy':'training folds only; paired 2x exposure control; no Dev/Test',
    })
    return s

_original_signature = original.signature

def self_test():
    from opencc import OpenCC
    assert OpenCC('s2t').convert('这条评论的味道很好') == '這條評論的味道很好'
    df=pd.DataFrame({'id':['r1','r2'],'text':['味道很好。','面条太咸。'],'aspect':['味道','面条'], 'V':[7.0,2.5], 'A':[6.5,7.8]})
    x,au=build_training_views(df,'s2t_augment')
    dup,dupinfo=build_training_views(df,'repeat_control')
    assert len(x)==len(dup)==4 and au['changed_views']>=1
    assert np.array_equal(x[['V','A']].to_numpy(),dup[['V','A']].to_numpy())
    ys=np.stack([np.linspace(1,9,40),np.linspace(9,1,40)],axis=1)
    weights,info=continuous_lds_weights(ys)
    assert weights.shape==(40,2) and torch.isfinite(weights).all()
    assert torch.allclose(weights.mean(0),torch.ones(2),atol=1e-5)
    p=torch.tensor([[4.2,5.1],[2.3,7.4]])
    t=torch.tensor([[5.,5.],[3.,7.]])
    ref=training_loss(p,t,kind='l1')
    same=((p-t).abs()*torch.ones_like(p)).mean()
    assert torch.allclose(ref,same,atol=1e-7)
    print('S2T LDS SELF TEST PASSED',{'s2t_changed':au['changed_views'],'lds_info':info},flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--action',choices=['self-test','check','run','aggregate'],default='check')
    ap.add_argument('--project',type=Path,default=Path('技术成果核对表'))
    ap.add_argument('--out',type=Path,default=Path('optimization_v2/s2t_lds_v1'))
    ap.add_argument('--manifest',type=Path,default=Path('optimization_v2/validation/proposed_3fold_manifest.csv'))
    ap.add_argument('--model-path',type=Path,default=Path('模型缓存/chinese-bert-wwm-ext'))
    ap.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    ap.add_argument('--seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=ap.parse_args()
    if args.action=='self-test':self_test();return
    if args.epochs <1 or args.batch_size<1:ap.error('Epochs and batch size must be positive')
    args.project=args.project.resolve();args.out=args.out.resolve()
    args.manifest=args.manifest.resolve();args.model_path=args.model_path.resolve()
    rows,m=original.verify_manifest(args.project,args.manifest)
    if args.action=='check':
        print('TRAIN ONLY CHECK PASSED',{'train_rows':len(rows),'folds':sorted(map(int,m.fold.unique())),
              'methods':args.methods,'seeds':args.seeds,'extra_train_data':'none','official_Dev_Test':'not accessed'},flush=True)
        return
    if args.action=='run':
        args.out.mkdir(parents=True,exist_ok=True)
        original.run(args,rows,m)
    else:
        original.aggregate(args,rows,m)


# All original integrity checks, output persistence, exact fold alignment and aggregation reused.
original.METHODS.update({k:('l1',0.,False) for k in METHODS})
original.train_one=train_one
original.signature=signature

if __name__=='__main__':main()
