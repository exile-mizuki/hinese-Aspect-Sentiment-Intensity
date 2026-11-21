import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class VADataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.tokens = tokenizer(
            frame["text"].astype(str).tolist(),
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            frame[["V", "A"]].to_numpy(dtype=np.float32)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            {k: v[index] for k, v in self.tokens.items()},
            self.labels[index],
        )


class SentenceVA(nn.Module):
    def __init__(self, model_path, means):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, 2)

        # 从数据均值开始预测，避免初始分数接近零。
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(
                torch.tensor(means, dtype=torch.float32)
            )

    def forward(self, **tokens):
        cls = self.encoder(
            **tokens
        ).last_hidden_state[:, 0, :]
        return self.head(cls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_path = Path(args.data).resolve()
    model_path = Path(args.model_path).resolve()
    out = Path(args.out).resolve()

    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Output is not empty; refusing overwrite: {out}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    frame = pd.read_csv(
        data_path,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    for col in ("text", "V", "A"):
        if col not in frame.columns:
            raise ValueError(f"Missing column: {col}")

    for col in ("V", "A"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")

    assert frame["text"].astype(str).str.strip().ne("").all()
    assert np.isfinite(frame[["V", "A"]].to_numpy()).all()
    assert frame[["V", "A"]].ge(1).all().all()
    assert frame[["V", "A"]].le(9).all().all()

    print("=== EMOBANK PRETRAINING ===", flush=True)
    print("Data:", data_path, flush=True)
    print("Samples:", len(frame), flush=True)
    print("Device:", device, flush=True)
    print("Seed:", args.seed, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    dataset = VADataset(frame, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = SentenceVA(
        model_path,
        frame[["V", "A"]].mean().to_numpy(dtype=np.float32),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": args.encoder_lr,
            },
            {
                "params": model.head.parameters(),
                "lr": args.head_lr,
            },
        ],
        weight_decay=0.01,
    )

    total_steps = args.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    history = []
    model.train()

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        count = 0

        for tokens, labels in loader:
            tokens = {
                k: v.to(device, non_blocking=True)
                for k, v in tokens.items()
            }
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if bf16:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    pred = model(**tokens)
                    loss = torch.nn.functional.l1_loss(
                        pred.float(), labels
                    )
            else:
                pred = model(**tokens)
                loss = torch.nn.functional.l1_loss(
                    pred.float(), labels
                )

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            scheduler.step()

            running += float(loss.item()) * len(labels)
            count += len(labels)

        train_mae = running / count
        history.append({
            "epoch": epoch,
            "train_l1": train_mae,
        })

        print(
            f"EPOCH {epoch}/{args.epochs} "
            f"train_l1={train_mae:.6f}",
            flush=True,
        )

    out.mkdir(parents=True, exist_ok=False)

    # 只把编码器作为下一阶段 SIGHAN 的初始化。
    model.encoder.save_pretrained(
        out, safe_serialization=True
    )
    tokenizer.save_pretrained(out)

    # 辅助任务回归头单独保存，方便审计，不自动迁移。
    torch.save(
        model.head.state_dict(),
        out / "emobank_aux_head.pt",
    )

    sha = hashlib.sha256(
        data_path.read_bytes()
    ).hexdigest()

    report = {
        "data": str(data_path),
        "data_sha256": sha,
        "samples": len(frame),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "history": history,
        "downstream": "SIGHAN aspect-level V/A",
        "transfer": "encoder_only",
    }
    (out / "pretraining_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ENCODER SAVED:", out, flush=True)
    print("EMOBANK PRETRAINING FINISHED", flush=True)


if __name__ == "__main__":
    main()
