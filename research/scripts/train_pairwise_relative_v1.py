#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped-batch relative-intensity ablation for SIGHAN Training ONLY.

Run both matched group sampler control and relative loss, using a frozen CV manifest.
No official Validation/Testing data are read.
"""
import gc
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import train_ablation_v2 as base

CONTROL = "pair_group_control"
RELATIVE = "pair_group_relative_02"
LAMBDA = {CONTROL: 0.0, RELATIVE: 0.20}
MIN_DIFF = 0.75
SCRIPT_VERSION = "v1.0"

# Keep the existing manifest validator, file integrity audit, per-fold OOF
# saving, and aggregations. Only override the train_one() implementation.
for name in (CONTROL, RELATIVE):
    if name in base.METHODS:
        raise RuntimeError(f"Method collision: {name}")
    base.METHODS[name] = ("l1", 0.0, False)


def build_training_plan(rows, y, *, seed, epochs, batch_size):
    """Deterministic per-epoch shuffle of WHOLE sentence groups.

    The same plan is used by both ablations, even if a sentence has 2-5 aspects.
    """
    group_map = rows.groupby(
        ["id", "text"], sort=False, dropna=False
    ).indices
    groups = [np.asarray(locations, dtype=np.int64)
              for locations in group_map.values()]
    if len(groups) == 0:
        raise ValueError("No training rows")
    if any(len(group) > batch_size for group in groups):
        raise ValueError(
            "Batch size is smaller than a sentence group. "
            "Increase --batch-size to at least the max aspect count."
        )

    generator = torch.Generator().manual_seed(seed)
    all_epochs = []
    count_selected = 0
    active_selected = 0

    for _ in range(epochs):
        order = torch.randperm(
            len(groups), generator=generator
        ).tolist()
        batches = []
        current = []
        member_groups = []

        def finalize():
            nonlocal count_selected, active_selected
            if not current:
                return
            idx = torch.tensor(current, dtype=torch.long)
            assert 1 <= len(idx) <= batch_size
            pair_i, pair_j, masks = [], [], []
            # Relative labels only for DIFFERENT aspects within same sentence.
            for member_positions in member_groups:
                for i, j in itertools.combinations(member_positions, 2):
                    dv = abs(float(y[current[i], 0] - y[current[j], 0]))
                    da = abs(float(y[current[i], 1] - y[current[j], 1]))
                    flags = [float(dv >= MIN_DIFF), float(da >= MIN_DIFF)]
                    if any(flags):
                        pair_i.append(i)
                        pair_j.append(j)
                        masks.append(flags)
            count_selected += len(pair_i)
            active_selected += sum(int(x[0] + x[1]) for x in masks)
            batches.append((
                idx,
                torch.tensor(pair_i, dtype=torch.long),
                torch.tensor(pair_j, dtype=torch.long),
                torch.tensor(masks, dtype=torch.float32).reshape(-1, 2),
            ))

        for gi in order:
            group = groups[gi].tolist()
            if current and len(current) + len(group) > batch_size:
                finalize()
                current, member_groups = [], []
            positions = list(range(len(current), len(current) + len(group)))
            member_groups.append(positions)
            current.extend(group)
        finalize()

        joined = [int(i) for batch in batches for i in batch[0].tolist()]
        assert len(joined) == len(y)
        assert sorted(joined) == list(range(len(y))), (
            "Every training sample must appear exactly once per epoch"
        )
        all_epochs.append(batches)

    assert count_selected > 0 and active_selected > 0, (
        "No eligible within-sentence aspect differences found"
    )
    return all_epochs, {
        "sentence_groups": len(groups),
        "steps": sum(len(b) for b in all_epochs),
        "selected_pairs_all_epochs": count_selected,
        "active_dimension_pairs_all_epochs": active_selected,
        "max_group_size": max(map(len, groups)),
    }


def pairwise_difference_loss(pred, true, i, j, mask):
    """Compare predicted and target (V_i-V_j,A_i-A_j), masked by >=0.75."""
    if i.numel() == 0:
        return pred.new_zeros((), dtype=torch.float32)
    d_pred = pred.float().index_select(0, i) - pred.float().index_select(0, j)
    d_true = true.float().index_select(0, i) - true.float().index_select(0, j)
    return ((d_pred - d_true).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def train_one(train_rows, eval_rows, *, method, seed, model_path,
              project, epochs, batch_size, device):
    from transformers import AutoModel, set_seed, get_linear_schedule_with_warmup
    from refine_experiment import tokenizer_for, encode, infer
    from round2_experiment import ArousalModel
    from bert_experiment import token_batch
    from training_utils_v2 import training_loss, fold_weights

    if method not in LAMBDA:
        raise ValueError(f"Unsupported method: {method}")
    if batch_size < 5:
        raise ValueError("Use --batch-size 8 (or at least 5)")
    set_seed(seed)
    tokenizer = tokenizer_for("marked")
    tokens = encode(train_rows, "marked", tokenizer)
    y = torch.tensor(train_rows[["V", "A"]].to_numpy(dtype=np.float32))
    _, ce_weights, _, counts = fold_weights(y, False)
    plan, stats = build_training_plan(
        train_rows, y, seed=seed, epochs=epochs, batch_size=batch_size
    )
    print("PAIRWISE TRAIN FOLD AUDIT",
          {"method": method, "seed": seed, **stats}, flush=True)

    encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model = ArousalModel(encoder, means=y.mean(0)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(stats["steps"] * 0.1),
        num_training_steps=stats["steps"]
    )
    amp = device.type == "cuda"
    dtype = (torch.bfloat16 if amp and torch.cuda.is_bf16_supported()
             else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    history = []
    pair_weight = LAMBDA[method]

    for epoch, batches in enumerate(plan, 1):
        model.train()
        start = time.perf_counter()
        accumulated = 0.0
        accumulated_l1 = 0.0
        accumulated_relative = 0.0
        batch_pair_count = 0

        for idx, pi, pj, mask in batches:
            optimizer.zero_grad(set_to_none=True)
            target = y[idx].to(device)
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=amp
            ):
                values, aux_logits = model(
                    return_aux=True, **token_batch(tokens, idx, device)
                )
                loss_l1 = training_loss(
                    values, target, aux_logits=aux_logits,
                    aux_weight=0.0, kind="l1"
                )
                if pair_weight and pi.numel():
                    pi_cuda = pi.to(device)
                    pj_cuda = pj.to(device)
                    mask_cuda = mask.to(device)
                    relative = pairwise_difference_loss(
                        values, target, pi_cuda, pj_cuda, mask_cuda
                    )
                else:
                    relative = values.float().new_zeros(())
                loss = loss_l1 + pair_weight * relative
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss in {method}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            n = len(idx)
            accumulated += float(loss.detach()) * n
            accumulated_l1 += float(loss_l1.detach()) * n
            accumulated_relative += float(relative.detach()) * n
            batch_pair_count += len(pi)

        row = {
            "epoch": epoch,
            "train_loss": accumulated / len(y),
            "l1_component": accumulated_l1 / len(y),
            "relative_component": accumulated_relative / len(y),
            "active_pairs": batch_pair_count,
            "steps": len(batches),
            "seconds": round(time.perf_counter() - start, 2),
        }
        print("TRAIN", method, seed, row, flush=True)
        history.append(row)

    predictions = infer(
        model, eval_rows, "marked", tokenizer, device,
        batch_size=batch_size
    )
    if predictions.shape != (len(eval_rows), 2):
        raise ValueError("Prediction shape mismatch")
    del model, optimizer, scheduler, tokens, tokenizer
    gc.collect()
    if amp:
        torch.cuda.empty_cache()
    return predictions, history, counts.tolist(), ce_weights.tolist()


def self_test():
    import pandas as pd
    sizes = [2, 5, 1, 3, 2]
    records = []
    for group_id, size in enumerate(sizes):
        for aspect in range(size):
            records.append({
                "id": str(group_id), "text": "sentence " + str(group_id),
                "aspect": str(aspect),
                "V": float(4.5 + aspect * 0.9),
                "A": float(4.0 + aspect * 0.8),
            })
    rows = pd.DataFrame(records)
    y = torch.tensor(rows[["V", "A"]].to_numpy(dtype=np.float32))
    p1, stats = build_training_plan(
        rows, y, seed=42, epochs=2, batch_size=8
    )
    p2, _ = build_training_plan(
        rows, y, seed=42, epochs=2, batch_size=8
    )
    assert all(torch.equal(a[0], b[0])
               for e1, e2 in zip(p1, p2)
               for a, b in zip(e1, e2))
    b = next(b for e in p1 for b in e if b[1].numel())
    ix, pi, pj, mask = b
    targets = y[ix]
    perfect = pairwise_difference_loss(
        targets, targets, pi, pj, mask
    )
    assert float(perfect) == 0.0
    perturbed = targets.clone()
    perturbed[pi[0], 0] += 0.7
    worse = pairwise_difference_loss(
        perturbed, targets, pi, pj, mask
    )
    assert float(worse) > 0
    print("PAIRWISE SELF TEST PASSED", stats, flush=True)


if __name__ == "__main__":
    # Retain original --action check/run/aggregate and all manifest audits.
    import sys
    if "--action" in sys.argv and sys.argv[
        sys.argv.index("--action") + 1
    ] == "self-test":
        self_test()
    else:
        base.train_one = train_one
        base.main()
