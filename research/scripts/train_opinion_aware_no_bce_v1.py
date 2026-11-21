#!/usr/bin/env python3
"""Reproducible training-only grouped-CV ablations. NEVER reads Dev/Test.

Typical workflow: --action check -> --action run [--methods ...] -> --action aggregate.
Stage 1: seed 42 and small set of weights on the SAME grouped CV folds.
Stage 2: after stage1 analysis, expand to seeds 42, 52, 62 (must retrain matched baseline).
Full/deployment testing is intentionally NOT included, to avoid accidentally using
an already-consulted official Test set during model selection.
"""
import argparse, gc, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from training_utils_v2 import bin_labels, fold_weights, scored_metrics, training_loss

METHODS={
    'marked_l1':('l1',0.,False),
    'a_mse':('mse',0.,False),
    'aux_0.02':('mse',.02,False),
    'aux_0.05':('mse',.05,False),
    'aux_0.10':('mse',.10,False),
    'aux_0.20':('mse',.20,False),
    'a_huber':('huber',0.,False),
    'both_huber':('both_huber',0.,False),
    'a_reweight_l1':('l1',0.,True),
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_rows(project):
    sys.path.insert(0,str(project/'src'))
    from main import train
    rows,_=train()
    return rows

def verify_manifest(project,manifest):
    if not manifest.is_file():raise FileNotFoundError(f'Missing frozen CV manifest: {manifest}')
    m=pd.read_csv(manifest,encoding='utf-8-sig',keep_default_na=False)
    rows=source_rows(project)
    need=['row_index','id','text','aspect','V','A','group','fold']
    if not set(need)<=set(m):raise ValueError('Manifest lacks required columns')
    if len(m)!=len(rows) or not np.array_equal(m.row_index.to_numpy(),np.arange(len(rows))):
        raise ValueError('Manifest row count/order differs from source training data')
    for k in ('id','text','aspect'):
        if not np.array_equal(m[k].to_numpy(),rows[k].to_numpy()):raise ValueError(f'Manifest data mismatch: {k}')
    if not np.allclose(m[['V','A']].to_numpy(),rows[['V','A']].to_numpy(),atol=1e-6,rtol=1e-6):
        raise ValueError('Manifest labels differ from source training labels')
    if m.fold.isna().any() or m.group.isna().any():raise ValueError('Missing fold/group assignments')
    for group,g in m.groupby('group'):
        if g.fold.nunique()!=1:raise ValueError('A connected group spans multiple folds')
    if m.fold.nunique()<3:raise ValueError('Require at least 3 folds')
    review=m.id.str.split(':').str[0]
    for fold in sorted(m.fold.unique()):
        tr=m.fold!=fold;va=~tr
        if set(review[tr])&set(review[va]):raise ValueError('Cross-fold review leak')
        if set(m.text[tr])&set(m.text[va]):raise ValueError('Cross-fold text leak')
    return rows,m

def train_one(train_rows, eval_rows, *, method, seed, model_path, project, epochs, batch_size, device):
    import torch.nn.functional as F
    from transformers import AutoModel, set_seed, get_linear_schedule_with_warmup
    from refine_experiment import tokenizer_for, encode, infer
    from opinion_aware_model_v2 import OpinionAwareModel
    from opinion_supervision_v2 import build_opinion_labels
    from bert_experiment import token_batch

    OPINION_WEIGHT = 0.0

    kind, aux_weight, use_weight = METHODS[method]
    set_seed(seed)

    tok = tokenizer_for('marked')
    tokens = encode(train_rows, 'marked', tok)
    opinion_labels, opinion_stats = build_opinion_labels(
        train_rows, tok, tokens
    )

    if opinion_stats.get('matched_raw_group', 0) != len(train_rows):
        raise RuntimeError("Raw opinion matching incomplete")

    if opinion_stats.get('aggregate_va_verified', 0) != len(train_rows):
        raise RuntimeError("Training V/A aggregation mismatch")

    print(
        "OPINION_TRAIN_COVERAGE",
        opinion_stats['supervised_examples'],
        "/",
        len(train_rows),
        flush=True
    )

    y = torch.tensor(
        train_rows[['V', 'A']].to_numpy(dtype=np.float32)
    )

    labels, ce_weights, a_weights, counts = fold_weights(
        y, use_weight
    )

    encoder = AutoModel.from_pretrained(
        str(model_path), local_files_only=True
    )

    model = OpinionAwareModel(
        encoder,
        means=y.mean(0),
        start_id=tok.convert_tokens_to_ids('[unused1]'),
        end_id=tok.convert_tokens_to_ids('[unused2]'),
        sep_id=tok.sep_token_id
    ).to(device)

    steps = epochs * math.ceil(len(y) / batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=.01
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps * .1), steps
    )

    amp = device.type == 'cuda'
    dtype = (
        torch.bfloat16
        if amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    scaler = torch.amp.GradScaler(
        'cuda', enabled=amp and dtype == torch.float16
    )

    generator = torch.Generator().manual_seed(seed)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        total_opinion_loss = 0.0

        for idx in torch.randperm(
            len(y), generator=generator
        ).split(batch_size):

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=amp
            ):
                values, aux, opinion_logits = model(
                    return_opinion=True,
                    **token_batch(tokens, idx, device)
                )

                va_loss = training_loss(
                    values,
                    y[idx].to(device),
                    aux_logits=aux,
                    aux_weight=aux_weight,
                    kind=kind,
                    sample_weights=(
                        a_weights[idx].to(device)
                        if use_weight else None
                    ),
                    aux_class_weights=(
                        ce_weights.to(device)
                        if aux_weight else None
                    )
                )

                full_gold = opinion_labels[idx]
                seq_len = opinion_logits.shape[1]

                if full_gold[:, seq_len:].ne(-100).any().item():
                    raise RuntimeError(
                        "Active opinion labels would be truncated"
                    )

                gold = full_gold[:, :seq_len].to(device)
                valid = gold.ne(-100)

                if opinion_logits.shape != gold.shape:
                    raise RuntimeError(
                        "Opinion logits/labels shape mismatch"
                    )

                token_loss = F.binary_cross_entropy_with_logits(
                    opinion_logits.float(),
                    gold.clamp_min(0).float(),
                    reduction='none'
                )

                token_counts = valid.sum(dim=1)

                sample_loss = (
                    (token_loss * valid.float()).sum(dim=1)
                    / token_counts.clamp_min(1)
                )

                supervised = token_counts.gt(0)

                opinion_loss = (
                    sample_loss.sum()
                    / supervised.sum().clamp_min(1)
                )

                loss = va_loss + OPINION_WEIGHT * opinion_loss

            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.
            )

            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            if scaler.get_scale() >= old_scale:
                scheduler.step()

            total_loss += float(loss.detach()) * len(idx)
            total_opinion_loss += (
                float(opinion_loss.detach()) * len(idx)
            )

        record = {
            'epoch': epoch,
            'train_loss': total_loss / len(y),
            'train_opinion_loss': total_opinion_loss / len(y),
            'seconds': round(
                time.perf_counter() - start, 2
            )
        }

        print('TRAIN', method, seed, record, flush=True)
        history.append(record)

    # 与旧模型采用相同的 V/A 推理与评估流程
    preds = infer(
        model,
        eval_rows,
        'marked',
        tok,
        device,
        batch_size=batch_size
    )

    if preds.shape != (len(eval_rows), 2):
        raise ValueError("Output has wrong shape")

    # 以下只评估留出折 Opinion F1，不参与梯度更新
    held_tokens = encode(eval_rows, 'marked', tok)
    held_gold, held_stats = build_opinion_labels(
        eval_rows, tok, held_tokens
    )

    tp = fp = fn = 0
    model.eval()

    with torch.inference_mode():
        for idx in torch.arange(
            len(eval_rows)
        ).split(batch_size):

            _, _, logits = model(
                return_opinion=True,
                **token_batch(held_tokens, idx, device)
            )

            predicted = logits.float().cpu().gt(0)

            full_gold = held_gold[idx]
            seq_len = logits.shape[1]

            if full_gold[:, seq_len:].ne(-100).any().item():
                raise RuntimeError(
                    "Active heldout labels would be truncated"
                )

            gold = full_gold[:, :seq_len]

            valid = gold.ne(-100)
            positive = gold.eq(1)

            tp += int(
                (predicted & positive & valid).sum()
            )
            fp += int(
                (predicted & ~positive & valid).sum()
            )
            fn += int(
                (~predicted & positive & valid).sum()
            )

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)

    opinion_metric = {
        'metric': 'heldout_opinion_token_f1',
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'f1': round(f1, 6),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'supervised_rows': held_stats['supervised_examples']
    }

    history.append(opinion_metric)
    print('OPINION_HELDOUT_F1', opinion_metric, flush=True)

    del model, optimizer, scheduler, tokens, tok
    del opinion_labels, held_tokens, held_gold
    gc.collect()

    if amp:
        torch.cuda.empty_cache()

    return preds, history, counts.tolist(), ce_weights.tolist()

def signature(args,rows,manifest):
    return {'architecture':'opinion_aware_fusion_no_bce_v1','training_source_file_sha256':{p.name:sha(p) for p in sorted((args.project/'data').glob('*Training*.json'))},
            'manifest_sha256':sha(args.manifest),'rows':len(rows),'fold_count':int(manifest.fold.nunique()),
            'model_path':str(args.model_path.resolve()),'epochs':args.epochs,'batch_size':args.batch_size,
            'learning_rate':2e-5,'max_length':128,'input_variant':'marked',
            'aux_bins':[5.0,6.5], 'prediction_rounding':'clip 1..9 then round 2',
            'dev_test_access':False}

def run(args,rows,m):
    import importlib.util
    if importlib.util.find_spec('transformers') is None:
        raise RuntimeError('Transformers is missing. Install in your own NLP training environment first.')
    if not args.model_path.is_dir() or not (args.model_path/'config.json').exists():
        raise FileNotFoundError('Missing local pretrained BERT base (config.json and weights); supply --model-path')
    from refine_experiment import ROOT
    if not (ROOT/'demo_tokenizer'/'tokenizer.json').is_file():raise FileNotFoundError('Original tokenizer missing')
    dev=torch.device(args.device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('No CUDA device. Use a GPU PC, or --device cpu (slow).')
    params=signature(args,rows,m)
    base=args.out/'training_v2';base.mkdir(parents=True,exist_ok=True)
    for method in args.methods:
        if method not in METHODS:raise ValueError(f'Unknown method: {method}')
        for seed in args.seeds:
            for fold in sorted(m.fold.unique()):
                fold=int(fold)
                folder=base/method/f'seed_{seed}'/f'fold_{fold}'
                folder.mkdir(parents=True,exist_ok=True)
                config={**params,'method':method,'method_config':METHODS[method], 'seed':seed,'fold':fold}
                confpath=folder/'config.json'
                predpath=folder/'heldout_predictions.csv'
                if confpath.exists() or predpath.exists():
                    if not (confpath.exists() and predpath.exists() and json.loads(confpath.read_text(encoding='utf-8'))==config):
                        raise RuntimeError(f'Resume config mismatch or partial run in {folder}; inspect before proceeding')
                    print('REUSE verified existing result:',folder,flush=True);continue
                tr=rows.loc[m.fold!=fold].reset_index(drop=True)
                held=rows.loc[m.fold==fold].reset_index(drop=True)
                preds,hist,counts,ceweights=train_one(tr,held,method=method,seed=seed,model_path=args.model_path,
                                                       project=args.project,epochs=args.epochs,batch_size=args.batch_size,
                                                       device=dev)
                out=held[['id','text','aspect','V','A']].copy()
                out[['pred_V','pred_A']]=np.round(np.clip(preds,1,9),2)
                out['fold']=fold
                out['group']=m.loc[m.fold==fold,'group'].to_numpy()
                out.to_csv(predpath,index=False,encoding='utf-8-sig')
                (folder/'history.json').write_text(json.dumps({'history':hist,'train_only_bin_counts':counts,
                    'train_only_aux_class_weights':ceweights,'heldout_metrics':scored_metrics(held[['V','A']].to_numpy(),preds)},
                    ensure_ascii=False,indent=2),encoding='utf-8')
                confpath.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
                print('SAVED',folder,flush=True)

def aggregate(args,rows,m):
    # Only summarize complete method×seed grids, never silently ignore missing folds.
    output=[];folds=[]
    target=args.out/'training_v2'
    for method in args.methods:
        for seed in args.seeds:
            buf=[]
            for fold in sorted(m.fold.unique()):
                path=target/method/f'seed_{seed}'/f'fold_{int(fold)}'/'heldout_predictions.csv'
                if not path.exists():raise FileNotFoundError(f'Incomplete training grid: {path}')
                d=pd.read_csv(path,encoding='utf-8-sig',keep_default_na=False)
                check=rows.loc[m.fold==fold].reset_index(drop=True)
                if not d[['id','text','aspect']].equals(check[['id','text','aspect']]):
                    raise ValueError(f'Results do not match held-out source rows: {path}')
                score=scored_metrics(d[['V','A']].to_numpy(),d[['pred_V','pred_A']].to_numpy())
                folds.append({'method':method,'seed':seed,'fold':int(fold),**score})
                buf.append(d)
            joined=pd.concat(buf,ignore_index=True)
            if len(joined)!=len(rows):raise ValueError('OOF size mismatch')
            output.append({'method':method,'seed':seed,**scored_metrics(joined[['V','A']].to_numpy(),joined[['pred_V','pred_A']].to_numpy())})
    summary=pd.DataFrame(output);target.mkdir(exist_ok=True)
    summary.to_csv(target/'oof_seed_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(folds).to_csv(target/'oof_fold_metrics.csv',index=False,encoding='utf-8-sig')
    metric_cols=['V_MAE','A_MAE','mean_MAE','high_A_MAE']
    table=summary.groupby('method')[metric_cols].agg(['mean','std'])
    table.to_csv(target/'seed_mean_std.csv',encoding='utf-8-sig')
    print(summary.to_string(index=False));print(table)
    (target/'aggregate_status.json').write_text(json.dumps({'training_source':'only local Training JSON',
       'official_dev_test_used_for_new_training':False,'status':'training runs aggregated; still need human error audit and independent development gate',
       'methods':args.methods,'seeds':args.seeds,'folds':int(m.fold.nunique()),
       'warning':'Never compare against old CV if fold assignments and training seeds differ; rerun matched baseline.'},
       ensure_ascii=False,indent=2),encoding='utf-8')

def self_test():
    torch.manual_seed(7)
    p=torch.nn.Parameter(torch.randn(12,2)+5)
    y=torch.ones(12,2)*5; y[:3,1]=7.7; y[3:8,1]=5.8
    logits=torch.nn.Parameter(torch.randn(12,3))
    _,w,sw,counts=fold_weights(y,True)
    for kind in ['l1','mse','huber','both_huber']:
        for aux in [0.,.02,.1]:
            a=training_loss(p,y,logits,aux,kind,sw,w)
            assert torch.isfinite(a)
            a.backward();assert torch.isfinite(p.grad).all()
            p.grad=None;logits.grad=None
    print(json.dumps({'self_test':'pass','training_fold_bin_counts':counts.tolist(),
                      'losses':['l1','mse','huber','both_huber'],'aux_weights_tested':[0.,.02,.1]},ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--action',choices=['check','run','aggregate','self-test'],default='check')
    ap.add_argument('--project',type=Path,default=Path('技术成果核对表'))
    ap.add_argument('--out',type=Path,default=Path('optimization_v2'))
    ap.add_argument('--manifest',type=Path,default=None)
    ap.add_argument('--model-path',type=Path,default=None)
    ap.add_argument('--methods',nargs='+',choices=list(METHODS),default=['marked_l1','aux_0.02','aux_0.05','aux_0.10','aux_0.20'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=ap.parse_args()
    if args.action=='self-test':self_test();return
    args.project=args.project.resolve();args.out=args.out.resolve()
    if args.manifest is None:args.manifest=args.out/'validation'/'proposed_5fold_manifest.csv'
    if args.model_path is None:args.model_path=args.project.parent/'模型缓存'/'chinese-bert-wwm-ext'
    if args.epochs<1 or args.batch_size<1:ap.error('epochs and batch_size must be >=1')
    rows,m=verify_manifest(args.project,args.manifest)
    summary={'verified':'TRAIN only','project':str(args.project),'manifest':str(args.manifest),
             'model_path':str(args.model_path),'folds':int(m.fold.nunique()),'n':len(rows),
             'methods':args.methods,'seeds':args.seeds,
             'required_full_trainings':len(args.methods)*len(args.seeds)*int(m.fold.nunique()),
             'pretrained_base_present':args.model_path.is_dir() and (args.model_path/'config.json').is_file(),
             'training_mode':'fixed epochs, no heldout early stopping, no official dev/test'}
    if args.action=='check':
        print(json.dumps(summary,ensure_ascii=False,indent=2));return
    if args.action=='run':run(args,rows,m)
    else:aggregate(args,rows,m)

if __name__=='__main__':main()

