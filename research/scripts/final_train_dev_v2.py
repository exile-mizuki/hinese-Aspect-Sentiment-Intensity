#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train fixed full-TRAIN candidates, save loadable checkpoints, then evaluate Validation.

Compatible with the original SIGHAN dimABSA project and optimization_v2 scripts.
No Testing inputs or truth are opened. Evaluation does NOT automatically deploy a model.
Usage: python final_train_dev_v2.py --action check ... / --action train ...
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

METHODS = {
    'marked_l1': ('l1', 0.0, False),
    'a_mse': ('mse', 0.0, False),
    'aux_0.05': ('mse', 0.05, False),
}


def sha256(file):
    h = hashlib.sha256()
    with open(file, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_project(project, scripts):
    # Take the user's existing implementations instead of silently introducing new preprocessing.
    sys.path.insert(0, str(project / 'src'))
    sys.path.insert(0, str(scripts))
    from main import train, inputs, truth
    from refine_experiment import tokenizer_for, encode, infer
    from round2_experiment import ArousalModel
    from bert_experiment import token_batch, scored_predictions
    from window_experiment import extract_scores
    from training_utils_v2 import fold_weights, training_loss, scored_metrics
    return train, inputs, truth, tokenizer_for, encode, infer, ArousalModel, token_batch, scored_predictions, extract_scores, fold_weights, training_loss, scored_metrics


def train_and_eval(method, args, rows, helpers, source_hashes, device):
    from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup, set_seed
    (train, inputs, truth, tokenizer_for, encode, infer, ArousalModel,
     token_batch, scored_predictions, extract_scores, fold_weights, training_loss, scored_metrics) = helpers
    folder = args.out / 'final_model' / method / f'seed_{args.seed}'
    checkpoint = folder / 'model.pt'
    metrics_path = folder / 'dev_metrics.json'
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Output folder is not empty: {folder}. Inspect existing results; do not overwrite.')
    folder.mkdir(parents=True, exist_ok=True)
    kind, aux_weight, use_weight = METHODS[method]
    set_seed(args.seed)
    tok = tokenizer_for('marked')
    train_tokens = encode(rows, 'marked', tok)
    y = torch.tensor(rows[['V', 'A']].to_numpy(dtype=np.float32))
    _, class_weights, sample_weights, counts = fold_weights(y, use_weight)
    encoder = AutoModel.from_pretrained(str(args.model_path), local_files_only=True)
    model = ArousalModel(encoder, means=y.mean(0)).to(device)
    steps = args.epochs * math.ceil(len(y) / args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * .1), steps)
    use_amp = device.type == 'cuda'
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and dtype == torch.float16)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    print(f'BEGIN FULL TRAIN {method} seed={args.seed} train={len(rows)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        begin = time.perf_counter()
        for idx in torch.randperm(len(y), generator=generator).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                values, logits = model(return_aux=True, **token_batch(train_tokens, idx, device))
                loss = training_loss(values, y[idx].to(device), aux_logits=logits,
                                     aux_weight=aux_weight, kind=kind,
                                     sample_weights=sample_weights[idx].to(device) if use_weight else None,
                                     aux_class_weights=class_weights.to(device) if aux_weight else None)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {method}, epoch={epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            total += float(loss.detach()) * len(idx)
        record = {'epoch': epoch, 'train_loss': total / len(y), 'seconds': round(time.perf_counter()-begin,2)}
        history.append(record)
        print(f'TRAIN {method} {args.seed} {record}', flush=True)

    # Save BEFORE consulting Validation labels; fixed-epoch checkpoint, never Dev early stopping.
    tok.save_pretrained(folder / 'tokenizer')
    model.encoder.config.save_pretrained(folder / 'encoder_config')
    payload = {'model': model.state_dict(), 'method': method, 'seed': args.seed,
               'epoch': args.epochs, 'input_variant': 'marked', 'max_length': 128,
               'train_rows': len(rows), 'training_sources_sha256': source_hashes}
    temporary = folder / 'model.pt.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    json_dump(folder / 'train_history.json', {'method': method, 'seed': args.seed, 'epochs': args.epochs,
              'batch_size': args.batch_size, 'lr': 2e-5, 'max_length': 128,
              'history': history, 'train_bin_counts': counts.tolist(),
              'trained_only_with': 'Training JSON', 'model_path': str(args.model_path),
              'training_sources_sha256': source_hashes})
    print(f'CHECKPOINT SAVED {checkpoint}', flush=True)

    # Free the trained model and explicitly reload a clean copy of the saved weights.
    del model, optimizer, scheduler, train_tokens
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['method'] != method or state['seed'] != args.seed or state['train_rows'] != len(rows):
        raise ValueError('Checkpoint metadata mismatch')
    fresh_encoder = AutoModel.from_config(AutoConfig.from_pretrained(folder / 'encoder_config', local_files_only=True))
    restored = ArousalModel(fresh_encoder)
    restored.load_state_dict(state['model'], strict=True)
    restored = restored.to(device)
    del state
    print('CHECKPOINT RELOAD PASSED', flush=True)

    # After checkpoint is frozen, evaluate official Validation exactly once for this method.
    dev = inputs('Validation')
    dev_gold = truth('Validation')
    preds = infer(restored, dev, 'marked', tok, device, batch_size=args.batch_size)
    prediction, official = scored_predictions(dev, preds, dev_gold)
    gold_array = extract_scores(dev_gold, dev)
    scores = scored_metrics(gold_array, preds)
    prediction.to_csv(folder / 'dev_official_predictions.txt', sep=' ', index=False, encoding='utf-8')
    cases = dev[['id','text','aspect']].copy()
    cases[['true_V','true_A']] = gold_array
    cases[['pred_V','pred_A']] = np.round(np.clip(preds,1,9),2)
    cases.to_csv(folder / 'dev_cases.csv', index=False, encoding='utf-8-sig')
    json_dump(metrics_path, {'method': method, 'seed': args.seed,
             'split': 'Validation', 'training_source': 'Train only',
             'checkpoint_loaded_from_disk': True, 'deployment_approved': False,
             'note': 'Development already used in earlier project iterations; this is exploratory, not an untouched test.',
             'metrics': scores, 'official_task1_metrics': official,
             'checkpoint_sha256': sha256(checkpoint)})
    print(f'DEV {method} seed={args.seed}: {json.dumps(scores, ensure_ascii=False)}', flush=True)
    print('NOT DEPLOYED; existing web default unchanged.', flush=True)
    del restored, tok
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    return {'method':method, 'seed':args.seed, **scores}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--action', choices=['check','train','summary'], default='check')
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--model-path', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('./optimization_v2'))
    ap.add_argument('--methods', nargs='+', choices=sorted(METHODS), default=['marked_l1','a_mse','aux_0.05'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = ap.parse_args()
    for attr in ('project','model_path','out'):
        setattr(args, attr, getattr(args, attr).resolve())
    if args.batch_size < 1 or args.epochs < 1:
        ap.error('epochs and batch-size must be >= 1')
    if not (args.model_path/'config.json').is_file():
        ap.error('Base model config.json missing')
    if not (args.model_path/'pytorch_model.bin').is_file() and not (args.model_path/'model.safetensors').is_file():
        ap.error('Base model weights missing')
    scripts = args.out / 'scripts'
    if not (scripts/'training_utils_v2.py').is_file():
        ap.error('optimization_v2/scripts/training_utils_v2.py missing')
    if not (args.project/'demo_tokenizer'/'tokenizer.json').is_file():
        ap.error('Project demo_tokenizer/tokenizer.json missing')
    helpers = load_project(args.project, scripts)
    rows, raw_count = helpers[0]()
    if len(rows) != 7476:
        raise RuntimeError(f'Expected 7476 training instances; got {len(rows)}. Review data changes first.')
    train_paths = sorted((args.project/'data').glob('*Training*.json'))
    if len(train_paths) != 2:
        raise RuntimeError('Expected exactly two Training JSON files')
    source_hashes = {p.name: sha256(p) for p in train_paths}
    print(json.dumps({'action': args.action, 'train_rows': len(rows), 'raw_train_tuples': raw_count,
                      'methods': args.methods, 'seed': args.seed, 'epochs': args.epochs,
                      'batch_size': args.batch_size, 'device': args.device,
                      'model_path': str(args.model_path),
                      'output': str(args.out/'final_model'),
                      'training_source_files_sha256': source_hashes,
                      'test_data_access': False}, ensure_ascii=False, indent=2))
    if args.action == 'check':
        return
    if args.action == 'summary':
        records = []
        for method in args.methods:
            path = args.out/'final_model'/method/f'seed_{args.seed}'/'dev_metrics.json'
            if not path.is_file():
                raise FileNotFoundError(f'Missing official Dev results: {path}')
            d = json.loads(path.read_text(encoding='utf-8'))
            records.append({'method':method, 'seed':args.seed, **d['metrics']})
        table = pd.DataFrame(records)
        args.out.joinpath('final_model').mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out/'final_model'/'dev_comparison.csv',index=False,encoding='utf-8-sig')
        print(table.to_string(index=False))
        print('This is exploratory Dev comparison, NOT automatic model selection.')
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    results = []
    for method in args.methods:
        results.append(train_and_eval(method,args,rows,helpers,source_hashes,device))
    table = pd.DataFrame(results)
    table.to_csv(args.out/'final_model'/'dev_comparison.csv',index=False,encoding='utf-8-sig')
    print('\nDEV COMPARISON (exploratory; no deployment):')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
