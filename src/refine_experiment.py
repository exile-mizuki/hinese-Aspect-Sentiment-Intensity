"""Predeclared aspect marking and label-frequency weighting experiments; training reads dev only."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from bert_experiment import IntensityModel, scored_predictions, token_batch, write_json
from main import D, O, inputs, train, truth

ROOT = O.parent
OUT = O / 'refinement'
BASE = ROOT.parent / '模型缓存' / 'chinese-bert-wwm-ext'
VARIANTS = ('control_long', 'marked', 'marked_balanced')


def marked_text(text, aspect):
    if not aspect or aspect == 'NULL':
        return text
    return text.replace(aspect, '[unused1]' + aspect + '[unused2]')


def tokenizer_for(variant):
    tok = AutoTokenizer.from_pretrained(ROOT / 'demo_tokenizer', local_files_only=True)
    if variant != 'control_long':
        before = len(tok)
        tok.add_special_tokens({'additional_special_tokens': ['[unused1]', '[unused2]']})
        assert len(tok) == before, 'Markers must reuse existing BERT vocabulary entries.'
        assert tok.encode('[unused1]', add_special_tokens=False) == [tok.convert_tokens_to_ids('[unused1]')]
    return tok


def encode(rows, variant, tokenizer):
    texts = rows.text.tolist() if variant == 'control_long' else [marked_text(r.text, r.aspect) for r in rows.itertuples()]
    return tokenizer(rows.aspect.tolist(), texts, truncation=True, max_length=128,
                     padding=True, pad_to_multiple_of=8, return_tensors='pt')


def infer(model, rows, variant, tokenizer, device, batch_size=32):
    unique = rows[['text', 'aspect']].drop_duplicates(ignore_index=True)
    tokens = encode(unique, variant, tokenizer)
    pieces = []
    model.eval()
    with torch.inference_mode():
        for idx in torch.arange(len(unique)).split(batch_size):
            pieces.append(model(**token_batch(tokens, idx, device)).float().cpu().numpy())
    unique[['V', 'A']] = np.concatenate(pieces)
    return rows[['text', 'aspect']].merge(unique, on=['text', 'aspect'], how='left', sort=False,
                                        validate='many_to_one')[['V', 'A']].to_numpy()


def label_weights(y):
    # Each dimension has independent bins [1,2), ... [8,9]. Derived only from training labels.
    bins = np.clip(np.floor(y).astype(int) - 1, 0, 7)
    weights = np.ones_like(y)
    counts = []
    for j in range(2):
        n = np.bincount(bins[:, j], minlength=8)
        raw = np.sqrt(len(y) / (8 * np.maximum(n, 1)))
        weights[:, j] = np.clip(raw[bins[:, j]], .5, 3)
        weights[:, j] /= weights[:, j].mean()
        counts.append(n.tolist())
    return weights, counts


def checkpoint(variant, seed):
    return ROOT / 'checkpoints' / 'refinement' / variant / f'seed_{seed}' / 'best.pt'


def load_model(variant, seed, device):
    state = torch.load(checkpoint(variant, seed), map_location='cpu', weights_only=True)
    assert state['variant'] == variant and state['seed'] == seed
    model = IntensityModel(AutoModel.from_config(AutoConfig.from_pretrained(ROOT / 'demo_tokenizer', local_files_only=True)))
    model.load_state_dict(state['model'])
    return model.to(device)


def run(variant, seed, device):
    folder = OUT / variant / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint(variant, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    started = time.perf_counter()
    rows, _ = train()
    dev = inputs('Validation')
    gold = truth('Validation')
    tok = tokenizer_for(variant)
    tokens = encode(rows, variant, tok)
    y_np = rows[['V', 'A']].to_numpy(dtype=np.float32)
    y = torch.tensor(y_np)
    weights, counts = label_weights(y_np)
    w = torch.tensor(weights) if variant == 'marked_balanced' else torch.ones_like(y)
    model = IntensityModel(AutoModel.from_pretrained(BASE, local_files_only=True), y.mean(0)).to(device)
    epochs, batch_size = 3, 32
    steps = epochs * math.ceil(len(y) / batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*.1), steps)
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    generator = torch.Generator().manual_seed(seed)
    history, best = [], float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.
        tick = time.perf_counter()
        for idx in torch.randperm(len(y), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(**token_batch(tokens, idx, device))
                loss = ((pred.float()-y[idx].to(device)).abs()*w[idx].to(device)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            total += loss.detach().item()*len(idx)
        values = infer(model, dev, variant, tok, device)
        packed, scores = scored_predictions(dev, values, gold)
        mae = (scores['Valance MAE']+scores['Arousal MAE'])/2
        entry = dict(epoch=epoch, train_loss=total/len(y), dev_mean_MAE=mae,
                     seconds=round(time.perf_counter()-tick,2), **scores)
        history.append(entry)
        print(f'{variant} seed={seed} epoch={epoch} dev_mean_MAE={mae:.6f} time={entry["seconds"]}', flush=True)
        if mae < best:
            best = mae
            temp = ckpt.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), variant=variant, seed=seed, epoch=epoch, max_length=128), temp)
            os.replace(temp, ckpt)
            packed.to_csv(folder/'dev_predictions.txt', sep=' ', index=False)
            np.save(folder/'dev_raw.npy', values)
            write_json(folder/'dev_scores.json', scores)
    pd.DataFrame(history).to_csv(folder/'training_history.csv', index=False)
    write_json(folder/'run_log.json', dict(variant=variant, seed=seed, max_length=128, epochs=3, batch_size=32,
               learning_rate=2e-5, loss='training-frequency-weighted L1' if variant=='marked_balanced' else 'L1',
               label_bin_counts=counts, weight_formula='clip(sqrt(N/(8*bin_count)),0.5,3); normalize each dimension to mean 1',
               selected_epoch=min(history,key=lambda x:x['dev_mean_MAE'])['epoch'], selection='minimum dev mean MAE',
               history=history, test_read=False, pretrained=str(BASE), train_rows=len(rows), dev_rows=len(dev),
               seconds=round(time.perf_counter()-started,2), checkpoint=str(ckpt),
               torch=str(torch.__version__), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               data_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in D.iterdir() if 'Training' in f.name or 'Validation' in f.name}))
    del model, optimizer, scheduler
    if device.type == 'cuda': torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=[*VARIANTS,'all'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    for variant in VARIANTS if args.variant=='all' else [args.variant]:
        for seed in args.seeds: run(variant, seed, torch.device(args.device))


if __name__=='__main__': main()
