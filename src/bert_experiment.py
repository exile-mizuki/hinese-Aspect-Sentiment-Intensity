"""对照实验：中文 BERT 仅整句 / 方面及整句输入，联合预测 V 和 A。"""
import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from main import O, inputs, save_pred, train, truth
from scoring import task1_score


class IntensityModel(nn.Module):
    def __init__(self, encoder, label_means=None):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.valence = nn.Linear(hidden_size, 1)
        self.arousal = nn.Linear(hidden_size, 1)
        if label_means is not None:
            for head, mean in zip((self.valence, self.arousal), label_means):
                nn.init.normal_(head.weight, std=0.01)
                nn.init.constant_(head.bias, float(mean))

    def forward(self, **tokens):
        cls = self.encoder(**tokens).last_hidden_state[:, 0, :]
        return torch.cat((self.valence(cls), self.arousal(cls)), dim=-1)


def encode(rows, mode, tokenizer, max_length):
    text = rows.text.tolist()
    if mode == "aspect":
        fields = (rows.aspect.tolist(), text)
    else:
        fields = (text,)
    return tokenizer(*fields, padding=True, truncation=True, max_length=max_length,
                     pad_to_multiple_of=8, return_tensors="pt")


def token_batch(tokens, indexes, device):
    selected = {key: value[indexes] for key, value in tokens.items()}
    lengths = selected["attention_mask"].sum(dim=1)
    width = 8 * math.ceil(int(lengths.max()) / 8)
    return {key: value[:, :width].contiguous().to(device) for key, value in selected.items()}


def infer_rows(model, rows, mode, tokenizer, max_length, batch_size, device):
    # 对相同整句只计算一次，句子对照的各方面应得到严格相同的分数。
    columns = ["text", "aspect"] if mode == "aspect" else ["text"]
    unique = rows[columns].drop_duplicates(ignore_index=True)
    tokens = encode(unique, mode, tokenizer, max_length)
    pieces = []
    model.eval()
    with torch.inference_mode():
        for indexes in torch.arange(len(unique)).split(batch_size):
            # 评估使用 float32，避免半精度造成两位小数得分的额外量化。
            pieces.append(model(**token_batch(tokens, indexes, device)).float().cpu().numpy())
    unique[["V", "A"]] = np.concatenate(pieces)
    return rows[columns].merge(unique, on=columns, how="left", validate="many_to_one",
                               sort=False)[["V", "A"]].to_numpy()


def scored_predictions(rows, values, gold):
    if not np.isfinite(values).all():
        raise ValueError("模型输出包含非有限值")
    pred = save_pred(rows, values, None)
    if not pred.ID.is_unique or pred.ID.tolist() != gold.ID.tolist():
        raise ValueError("预测句子 ID 或顺序与官方真值不一致")
    pattern = re.compile(r"\(([^,]+),[\d.]+#[\d.]+\)")
    for hypothesis, reference in zip(pred.Intensity, gold.Intensity):
        if pattern.findall(hypothesis) != pattern.findall(reference):
            raise ValueError(f"方面覆盖或顺序不一致：{hypothesis} / {reference}")
    return pred, task1_score(pred, gold, {})


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def path_for_mode(root, mode, smoke):
    return root / (("smoke_" if smoke else "") + "bert_" + mode)


def checkpoint_path(args, mode):
    return path_for_mode(args.checkpoint_dir, mode, args.smoke) / "best.pt"


def make_device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def run_experiment(args, mode, device, tokenizer):
    set_seed(args.seed)
    started = time.monotonic()
    rows, _ = train()
    dev_rows = inputs("Validation")
    dev_gold = truth("Validation")
    if args.smoke:
        rows = rows.iloc[:64].copy()
        dev_gold = dev_gold.head(12)
        dev_rows = dev_rows[dev_rows.id.isin(dev_gold.ID)]
    y = torch.tensor(rows[["V", "A"]].to_numpy(), dtype=torch.float32)
    tokens = encode(rows, mode, tokenizer, args.max_length)
    model = IntensityModel(AutoModel.from_pretrained(args.model_path), y.mean(0)).to(device)
    folder = path_for_mode(args.output_dir, mode, args.smoke)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(args, mode)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.smoke else args.epochs
    steps = epochs * math.ceil(len(y) / args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    best_mae = math.inf
    best_epoch = 0
    best_prediction = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_started = time.monotonic()
        running_loss = torch.zeros((), device=device)
        for indexes in torch.randperm(len(y), generator=generator).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                pred = model(**token_batch(tokens, indexes, device))
                loss = nn.functional.l1_loss(pred.float(), y[indexes].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            running_loss += loss.detach() * len(indexes)
        dev_values = infer_rows(model, dev_rows, mode, tokenizer, args.max_length, args.batch_size, device)
        prediction, scores = scored_predictions(dev_rows, dev_values, dev_gold)
        mean_mae = (scores["Valance MAE"] + scores["Arousal MAE"]) / 2
        entry = dict(epoch=epoch, train_mean_l1=running_loss.item()/len(y), dev_mean_MAE=mean_mae,
                     seconds=round(time.monotonic() - epoch_started, 2), **scores)
        history.append(entry)
        print(f"{mode} epoch {epoch}/{epochs} | train {entry['train_mean_l1']:.4f} | "
              f"dev mean MAE {mean_mae:.4f} | {entry['seconds']:.1f}s", flush=True)
        if mean_mae < best_mae:
            best_mae, best_epoch, best_prediction = mean_mae, epoch, prediction
            temporary = checkpoint.with_name(f"best.{os.getpid()}.tmp")
            torch.save({"model": model.state_dict(), "mode": mode, "max_length": args.max_length,
                        "epoch": epoch, "seed": args.seed}, temporary)
            os.replace(temporary, checkpoint)
    if best_prediction is None:
        raise RuntimeError("未产生有效的开发集检查点")
    del optimizer, scheduler
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    del state
    test_rows = inputs("Testing")
    test_gold = truth("Testing")
    if args.smoke:
        test_gold = test_gold.head(12)
        test_rows = test_rows[test_rows.id.isin(test_gold.ID)]
    values = infer_rows(model, test_rows, mode, tokenizer, args.max_length, args.batch_size, device)
    test_prediction, test_scores = scored_predictions(test_rows, values, test_gold)
    dev_scores = task1_score(best_prediction, dev_gold, {})
    for split, pred, scores in [("dev", best_prediction, dev_scores), ("test", test_prediction, test_scores)]:
        pred.to_csv(folder / f"{split}_predictions.txt", sep=" ", index=False, lineterminator="\n")
        write_json(folder / f"{split}_scores.json", scores)
    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    write_json(folder / "run_log.json", {
        "method": "bert_" + mode, "smoke": args.smoke, "pretrained": str(args.model_path),
        "seed": args.seed, "epochs": epochs, "batch_size": args.batch_size,
        "max_length": args.max_length, "learning_rate": args.learning_rate,
        "warmup_steps": int(0.1 * steps), "optimizer": "AdamW", "weight_decay": 0.01,
        "gradient_clip_norm": 1.0, "loss": "mean of V and A L1", "selection": "minimum dev mean MAE",
        "selected_epoch": best_epoch, "history": history, "train_rows": len(rows),
        "dev_rows": len(dev_rows), "test_rows": len(test_rows), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_precision": str(amp_dtype) if amp else "float32", "torch": str(torch.__version__),
        "transformers": transformers.__version__, "seconds": round(time.monotonic() - started, 2),
        "checkpoint": str(checkpoint.resolve()),
    })
    print(f"{mode} best epoch={best_epoch}, test={test_scores}", flush=True)


def demo(args, device, tokenizer):
    state = torch.load(checkpoint_path(args, args.mode), map_location="cpu", weights_only=True)
    if state["mode"] != args.mode:
        raise ValueError("检查点与输入方案不匹配")
    model = IntensityModel(AutoModel.from_config(AutoConfig.from_pretrained(args.model_path)))
    model.load_state_dict(state["model"])
    del state["model"]
    model.to(device)
    rows = pd.DataFrame({"aspect": args.predict_aspects, "text": args.predict_text})
    values = infer_rows(model, rows, args.mode, tokenizer, state["max_length"], args.batch_size, device)
    for aspect, (v, a) in zip(rows.aspect, values.clip(1, 9)):
        print(f"{aspect}  V={v:.2f}  A={a:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--mode", choices=["both", "sentence", "aspect"], default="both")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=O.parent / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=O)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="64 训练方面及 12 句验证/测试，写入单独目录")
    parser.add_argument("--predict-text")
    parser.add_argument("--predict-aspects", nargs="+")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_length) <= 0 or args.max_length % 8:
        parser.error("轮数和批次须大于 0，最大长度须为 8 的正整数倍")
    if bool(args.predict_text) != bool(args.predict_aspects) or (args.predict_text and args.mode == "both"):
        parser.error("演示需指定一个 mode，并同时提供 --predict-text 和 --predict-aspects")
    device = make_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.predict_text:
        demo(args, device, tokenizer)
    else:
        modes = ("sentence", "aspect") if args.mode == "both" else (args.mode,)
        for mode in modes:
            run_experiment(args, mode, device, tokenizer)


if __name__ == "__main__":
    main()
