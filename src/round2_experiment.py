"""Training-only grouped CV for arousal loss / auxiliary-bin ablations.

All methods use a fixed three epochs. No held-out fold is used for early stopping.
Official development/test labels are not read by the cv command.
"""
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, get_linear_schedule_with_warmup, set_seed

from main import D, O, train
from bert_experiment import IntensityModel, token_batch, write_json
from refine_experiment import BASE, ROOT, encode, infer, tokenizer_for

OUT = O / 'round2'
METHODS = ('marked_l1', 'arousal_mse', 'arousal_mse_aux')
PLAN = dict(methods=list(METHODS), epochs=3, batch_size=32, seed=42,
            folds=3, fold_seed=20261119, learning_rate=2e-5,
            max_length=128, aux_weight=0.1, arousal_bins=[5.0, 6.5],
            grouping='union of review ID prefix before colon and exact sentence duplicates',
            epoch_selection='fixed epoch 3; no early stopping on held-out folds',
            selection='eligible if pooled OOF A_MAE and high_A_MAE both decrease, mean_MAE <= baseline + 0.003; choose smallest A_MAE',
            tie_break='method name ascending', score_rounding='clip [1,9], round 2 decimals before metrics',
            deployment_gate='CV must pass; full-training dev A_MAE and high_A_MAE must decrease vs matched baseline, mean_MAE <= baseline + 0.003',
            inference='regression heads only; auxiliary head used in training only',
            official_test_status='already examined in previous work; subsequent comparisons are exploratory, not an untouched holdout')


class ArousalModel(IntensityModel):
    def __init__(self, encoder, means=None):
        super().__init__(encoder, means)
        self.arousal_class = nn.Linear(encoder.config.hidden_size, 3)

    def forward(self, return_aux=False, **tokens):
        cls = self.encoder(**tokens).last_hidden_state[:, 0, :]
        values = torch.cat((self.valence(cls), self.arousal(cls)), dim=-1)
        return (values, self.arousal_class(cls)) if return_aux else values


def metrics(y, pred):
    p = np.round(np.clip(pred, 1, 9), 2)
    err = np.abs(p-y)
    high = y[:, 1] >= 6.5
    result = dict(n=len(y), V_MAE=float(err[:, 0].mean()), A_MAE=float(err[:, 1].mean()),
                  mean_MAE=float(err.mean()), high_A_n=int(high.sum()),
                  high_A_MAE=float(err[high, 1].mean()) if high.any() else None)
    for j, dim in enumerate(('V', 'A')):
        result[dim+'_PCC'] = float(np.corrcoef(y[:, j], p[:, j])[0, 1]) if np.std(p[:, j]) > 0 else None
    return result


def prepare_folds(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    reviews = rows.id.str.split(':').str[0]
    for values in (reviews, rows.text):
        first = {}
        for i, value in enumerate(values):
            if value in first: union(i, first[value])
            else: first[value] = i
    groups = np.array([find(i) for i in range(n)])
    folds = np.full(n, -1)
    audit = []
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=20261119)
    for fold, (a, b) in enumerate(splitter.split(rows, groups=groups)):
        folds[b] = fold
        review_overlap = set(reviews.iloc[a]) & set(reviews.iloc[b])
        text_overlap = set(rows.text.iloc[a]) & set(rows.text.iloc[b])
        assert not review_overlap and not text_overlap
        audit.append(dict(fold=fold, train_rows=len(a), heldout_rows=len(b),
                          train_groups=len(set(groups[a])), heldout_groups=len(set(groups[b])),
                          review_overlap=len(review_overlap), exact_text_overlap=len(text_overlap),
                          heldout_high_A=int((rows.A.iloc[b] >= 6.5).sum())))
    assert (folds >= 0).all()
    manifest = rows.copy()
    manifest.insert(0, 'row_index', np.arange(n))
    manifest['review_id'] = reviews
    manifest['group'] = groups
    manifest['fold'] = folds
    manifest.to_csv(OUT/'fold_assignments.csv', index=False, encoding='utf-8-sig')
    write_json(OUT/'fold_audit.json', dict(n=n, groups=len(set(groups)), folds=audit,
               official_dev_test_read=False, near_duplicate_detection=False))
    return folds


def fit(rows, method, device, folder):
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    tok = tokenizer_for('marked')
    tokens = encode(rows, 'marked', tok)
    y = torch.tensor(rows[['V','A']].to_numpy(dtype=np.float32))
    bins = torch.bucketize(y[:, 1].contiguous(), torch.tensor([5., 6.5]), right=True)
    counts = torch.bincount(bins, minlength=3).float()
    weights = (len(y)/(3*counts.clamp_min(1))).sqrt().clamp(.5, 3).to(device)
    model = ArousalModel(AutoModel.from_pretrained(BASE, local_files_only=True), y.mean(0)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    steps = 3*math.ceil(len(y)/32)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*.1), steps)
    use_amp = device.type == 'cuda'
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and dtype == torch.float16)
    generator = torch.Generator().manual_seed(42)
    history = []
    for epoch in range(1, 4):
        model.train()
        started = time.perf_counter()
        total = 0.
        for idx in torch.randperm(len(y), generator=generator).split(32):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                p, logits = model(return_aux=True, **token_batch(tokens, idx, device))
                delta = p.float()-y[idx].to(device)
                loss = (delta[:, 0].abs().mean() + (delta[:, 1].abs().mean() if method == 'marked_l1' else delta[:, 1].square().mean()))/2
                if method == 'arousal_mse_aux':
                    loss = loss + .1*nn.functional.cross_entropy(logits.float(), bins[idx].to(device), weight=weights)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old: scheduler.step()
            total += loss.item()*len(idx)
        record = dict(epoch=epoch, loss=total/len(y), seconds=round(time.perf_counter()-started,2))
        history.append(record)
        print(folder.name, method, record, flush=True)
    write_json(folder/'training.json', dict(method=method, rows=len(rows), history=history,
               bin_counts=counts.tolist(), bin_weights=weights.cpu().tolist(), seed=42, epochs=3,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    del optimizer, scheduler, tokens
    return model, tok


def cv(device):
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(PLAN, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                training_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*Training*.json')})
    path = OUT/'experiment_plan.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == plan, 'Plan changed; use a new output directory.'
    else: write_json(path, plan)
    rows, _ = train()
    folds = prepare_folds(rows)
    summary, fold_scores = [], []
    for method in METHODS:
        predictions = np.full((len(rows), 2), np.nan, dtype=np.float32)
        for fold in range(3):
            folder = OUT/method/f'fold_{fold}'
            heldout = rows.loc[folds == fold].reset_index(drop=True)
            if (folder/'predictions.npy').exists():
                values = np.load(folder/'predictions.npy')
            else:
                model, tok = fit(rows.loc[folds != fold].reset_index(drop=True), method, device, folder)
                values = infer(model, heldout, 'marked', tok, device)
                np.save(folder/'predictions.npy', values)
                del model, tok
                gc.collect()
                if device.type == 'cuda': torch.cuda.empty_cache()
            assert values.shape == (len(heldout), 2) and np.isfinite(values).all()
            predictions[folds == fold] = values
            score = dict(method=method, fold=fold, **metrics(heldout[['V','A']].to_numpy(), values))
            fold_scores.append(score)
            print('FOLD', score, flush=True)
        assert np.isfinite(predictions).all()
        out = rows.copy()
        out['fold'] = folds
        out[['pred_V','pred_A']] = np.round(np.clip(predictions,1,9),2)
        out.to_csv(OUT/method/'oof_predictions.csv', index=False, encoding='utf-8-sig')
        summary.append(dict(method=method, **metrics(rows[['V','A']].to_numpy(), predictions)))
        pd.DataFrame(fold_scores).to_csv(OUT/'cv_fold_metrics.csv', index=False)
        pd.DataFrame(summary).to_csv(OUT/'cv_pooled_metrics.csv', index=False)
    base = summary[0]
    eligible = [r for r in summary[1:] if r['A_MAE'] < base['A_MAE'] and r['high_A_MAE'] < base['high_A_MAE'] and r['mean_MAE'] <= base['mean_MAE']+.003]
    selected = min(eligible, key=lambda r:(r['A_MAE'],r['method'])) if eligible else base
    probe = min(summary[1:], key=lambda r:(r['A_MAE'],r['method']))
    write_json(OUT/'selection.json', dict(selected=selected['method'], experimental_probe=probe['method'],
               eligible=[r['method'] for r in eligible], rule=PLAN['selection'],
               official_dev_test_read=False, scores=summary, selected_before_official_evaluation=True))
    print('SELECTED', selected, flush=True)


def full(device):
    # Only called after the complete training-only selection file exists.
    from main import inputs, truth
    from bert_experiment import scored_predictions
    from window_experiment import extract_scores
    choice = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    candidate = choice['selected'] if choice['selected'] != 'marked_l1' else choice['experimental_probe']
    rows, _ = train()
    records = []
    for method in ('marked_l1', candidate):
        folder = OUT/method/'full'
        model, tok = fit(rows, method, device, folder)
        if method == candidate:
            ckpt = ROOT/'checkpoints'/'round2'/'candidate.pt'
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), method=method, epoch=3, seed=42), ckpt)
        for split, name in [('Validation','dev'), ('Testing','test')]:
            r = inputs(split)
            values = infer(model, r, 'marked', tok, device)
            gold = truth(split)
            packed, official = scored_predictions(r, values, gold)
            packed.to_csv(folder/f'{name}_predictions.txt', sep=' ', index=False)
            np.save(folder/f'{name}_raw.npy', values)
            true = extract_scores(gold, r)
            score = dict(method=method, split=name, **metrics(true, values))
            records.append(score)
            write_json(folder/f'{name}_scores.json', dict(metrics=score, official=official))
            cases = r.copy()
            cases[['true_V','true_A']] = true
            cases[['pred_V','pred_A']] = np.round(np.clip(values,1,9),2)
            cases.to_csv(folder/f'{name}_cases.csv', index=False, encoding='utf-8-sig')
            print('FULL', score, flush=True)
        del model, tok
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(OUT/'official_metrics.csv', index=False)
    # Development gate is pre-specified in the user guide; test never changes this choice.
    b, c = [r for r in records if r['split']=='dev']
    accepted = choice['selected'] != 'marked_l1' and c['A_MAE'] < b['A_MAE'] and c['high_A_MAE'] < b['high_A_MAE'] and c['mean_MAE'] <= b['mean_MAE']+.003
    write_json(OUT/'deployment.json', dict(candidate=candidate, cv_pass=choice['selected']!='marked_l1',
               development_pass=accepted, accepted=accepted,
               current_demo='round2 candidate' if accepted else 'existing marked seed42 epoch2',
               test_used_for_selection=False, test_is_exploratory=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['cv','full'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(3)
    (cv if args.command=='cv' else full)(torch.device(args.device))
