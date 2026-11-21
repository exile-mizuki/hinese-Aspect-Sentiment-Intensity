import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from train_opinion_aware_v2 import verify_manifest, METHODS
from training_utils_v2 import fold_weights, training_loss, scored_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("baseline", "aware"), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--project", type=Path, default=Path("技术成果核对表"))
    ap.add_argument(
        "--manifest", type=Path,
        default=Path("optimization_v2/validation/proposed_3fold_manifest.csv")
    )
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument(
        "--out", type=Path,
        default=Path("optimization_v2/full_dev_gate_v1")
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import (
        AutoModel, get_linear_schedule_with_warmup, set_seed
    )
    from refine_experiment import tokenizer_for, encode, infer
    from bert_experiment import token_batch
    from round2_experiment import ArousalModel
    from opinion_aware_model_v2 import OpinionAwareModel
    from opinion_supervision_v2 import build_opinion_labels
    from main import inputs, truth
    from window_experiment import extract_scores

    rows, _ = verify_manifest(args.project, args.manifest)
    assert len(rows) == 7476, "Unexpected training count"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    out = args.out / args.arch / f"seed_{args.seed}"
    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {out}"
        )

    set_seed(args.seed)
    tok = tokenizer_for("marked")
    tokens = encode(rows, "marked", tok)
    y = torch.tensor(
        rows[["V", "A"]].to_numpy(dtype=np.float32)
    )

    kind, aux_weight, use_weight = METHODS["marked_l1"]
    _, ce_weights, a_weights, _ = fold_weights(y, use_weight)

    opinion_labels = None
    opinion_stats = None

    if args.arch == "aware":
        opinion_labels, opinion_stats = build_opinion_labels(
            rows, tok, tokens
        )

        if opinion_stats.get("aggregate_va_verified", 0) != len(rows):
            raise RuntimeError("Opinion annotation mismatch")

        print(
            "OPINION_COVERAGE",
            opinion_stats["supervised_examples"],
            "/", len(rows),
            flush=True
        )

    encoder = AutoModel.from_pretrained(
        str(args.model_path),
        local_files_only=True
    )

    if args.arch == "aware":
        model = OpinionAwareModel(
            encoder,
            means=y.mean(0),
            start_id=tok.convert_tokens_to_ids("[unused1]"),
            end_id=tok.convert_tokens_to_ids("[unused2]"),
            sep_id=tok.sep_token_id
        ).to(device)
    else:
        model = ArousalModel(
            encoder, means=y.mean(0)
        ).to(device)

    steps = args.epochs * math.ceil(
        len(y) / args.batch_size
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=.01
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(steps * .1),
        steps
    )

    amp = device.type == "cuda"
    dtype = (
        torch.bfloat16
        if amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp and dtype == torch.float16
    )

    generator = torch.Generator().manual_seed(args.seed)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tic = time.perf_counter()
        total_loss = 0.0
        total_opinion_loss = 0.0

        batches = torch.randperm(
            len(y), generator=generator
        ).split(args.batch_size)

        for idx in batches:
            optimizer.zero_grad(set_to_none=True)
            batch = token_batch(tokens, idx, device)

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=amp
            ):
                if args.arch == "aware":
                    values, aux, logits = model(
                        return_opinion=True, **batch
                    )
                else:
                    values, aux = model(
                        return_aux=True, **batch
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

                opinion_loss = va_loss.new_zeros(())

                if args.arch == "aware":
                    full_gold = opinion_labels[idx]
                    length = logits.shape[1]

                    if full_gold[:, length:].ne(-100).any().item():
                        raise RuntimeError(
                            "Opinion labels would be cropped"
                        )

                    gold = full_gold[:, :length].to(device)
                    valid = gold.ne(-100)

                    per_token = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.float(),
                        gold.clamp_min(0).float(),
                        reduction="none"
                    )

                    token_counts = valid.sum(1)

                    sample_loss = (
                        (per_token * valid.float()).sum(1)
                        / token_counts.clamp_min(1)
                    )

                    opinion_loss = (
                        sample_loss.sum()
                        / token_counts.gt(0).sum().clamp_min(1)
                    )

                loss = va_loss + 0.1 * opinion_loss

            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            if scaler.get_scale() >= previous_scale:
                scheduler.step()

            total_loss += float(loss.detach()) * len(idx)
            total_opinion_loss += (
                float(opinion_loss.detach()) * len(idx)
            )

        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(y),
            "train_opinion_loss": total_opinion_loss / len(y),
            "seconds": round(time.perf_counter() - tic, 2)
        }

        print(
            "TRAIN_FULL", args.arch, args.seed,
            record, flush=True
        )
        history.append(record)

    # Only evaluate Validation. Testing is never loaded.
    dev_rows = inputs("Validation")
    dev_true = extract_scores(
        truth("Validation"), dev_rows
    )

    dev_pred = infer(
        model, dev_rows, "marked", tok,
        device, batch_size=args.batch_size
    )

    if (
        dev_pred.shape != dev_true.shape
        or dev_pred.shape[1] != 2
    ):
        raise RuntimeError(
            f"Dev shape mismatch: "
            f"{dev_pred.shape} vs {dev_true.shape}"
        )

    if (
        not np.isfinite(dev_true).all()
        or not np.isfinite(dev_pred).all()
    ):
        raise RuntimeError("Nonfinite Dev data")

    metrics = scored_metrics(dev_true, dev_pred)

    print(
        "DEV_ONLY", args.arch, args.seed,
        metrics, flush=True
    )

    out.mkdir(parents=True, exist_ok=False)

    cases = dev_rows.copy()
    cases[["true_V", "true_A"]] = dev_true
    cases[["pred_V", "pred_A"]] = np.round(
        np.clip(dev_pred, 1, 9), 2
    )

    cases.to_csv(
        out / "dev_cases.csv",
        index=False,
        encoding="utf-8-sig"
    )

    report = {
        "architecture": args.arch,
        "seed": args.seed,
        "train_rows": len(rows),
        "dev_rows": len(dev_rows),
        "method": "marked_l1",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "opinion_weight": (
            0.1 if args.arch == "aware" else 0.0
        ),
        "metrics": metrics,
        "history": history,
        "opinion_stats": opinion_stats
    }

    (out / "dev_metrics.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=float
        ),
        encoding="utf-8"
    )

    torch.save(
        {
            "architecture": args.arch,
            "seed": args.seed,
            "method": "marked_l1",
            "epoch": args.epochs,
            "model": model.state_dict()
        },
        out / "model.pt"
    )

    print("SAVED_DEV_ONLY", out, flush=True)


if __name__ == "__main__":
    main()
