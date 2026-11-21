import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("未检测到 CUDA，请先检查 PyTorch GPU 环境")

device = torch.device("cuda")
source = Path("optimization_v2/external_data/ASAP/aux_train_5000_safe.csv")
bert_dir = Path("模型缓存/chinese-bert-wwm-ext")
output = Path("optimization_v2/external_data/ASAP/aux_model_v1")

df = pd.read_csv(source, encoding="utf-8-sig")
aspects = [c for c in df.columns if c not in ("id", "review", "star")]
labels = df[aspects].to_numpy(dtype=np.int64)

assert len(df) == 5000, f"预期 5000 条，实际 {len(df)}"
assert df["id"].nunique() == 5000
assert len(aspects) == 18, f"预期 18 个方面，实际 {len(aspects)}"
assert set(np.unique(labels)).issubset({-2, -1, 0, 1})

rng = np.random.default_rng(SEED)
order = rng.permutation(len(df))
val_idx = order[:500]
train_idx = order[500:]

# 只有冒烟测试时才缩小数据量
if args.smoke:
    train_idx = train_idx[:16]
    val_idx = val_idx[:8]

tokenizer = AutoTokenizer.from_pretrained(
    str(bert_dir), local_files_only=True
)

def make_windows(index):
    text = str(df.iloc[int(index)]["review"])
    tokens = tokenizer.encode(text, add_special_tokens=False)
    windows = []
    start = 0

    while True:
        part = tokens[start:start + 254]
        windows.append(
            tokenizer.build_inputs_with_special_tokens(part)
        )
        if start + 254 >= len(tokens):
            break
        start += 192

    batch = tokenizer.pad(
        {"input_ids": windows},
        padding=True,
        return_tensors="pt"
    )
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        torch.tensor(labels[int(index)],
                     dtype=torch.long, device=device)
    )

class ASAPAuxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            str(bert_dir), local_files_only=True
        )
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.mention = nn.Linear(hidden, 18)
        self.polarity = nn.Linear(hidden, 18 * 3)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 多窗口属于同一条评论，先聚合，再做评论级预测
        pooled = output.last_hidden_state[:, 0].mean(dim=0)
        pooled = self.dropout(pooled)
        mention_logits = self.mention(pooled)
        polarity_logits = self.polarity(pooled).reshape(18, 3)
        return mention_logits, polarity_logits

model = ASAPAuxModel().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-5, weight_decay=0.01
)
scaler = torch.amp.GradScaler("cuda")

# 提及检测使用适度类别权重，避免 -2 占多数
full_train_idx = order[500:]
train_labels = labels[full_train_idx]
positive = (train_labels != -2).sum(axis=0)
negative = (train_labels == -2).sum(axis=0)
weights = np.clip(
    negative / np.maximum(positive, 1), 0.5, 3.0
)
pos_weight = torch.tensor(
    weights, dtype=torch.float32, device=device
)

def calculate_loss(mention_logits, polarity_logits, target):
    mentioned = target != -2

    mention_loss = F.binary_cross_entropy_with_logits(
        mention_logits,
        mentioned.float(),
        pos_weight=pos_weight
    )
    # 仅对真正提及的方面计算情感分类损失
    polarity_loss = F.cross_entropy(
        polarity_logits[mentioned],
        target[mentioned] + 1
    )
    return 0.5 * mention_loss + polarity_loss

print("=== ASAP AUXILIARY TRAINING ===", flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Train reviews:", len(train_idx), flush=True)
print("Validation reviews:", len(val_idx), flush=True)
print("Aspects:", len(aspects), flush=True)
print("Smoke test:", args.smoke, flush=True)

accumulation = 4

for epoch in range(args.epochs):
    model.train()
    epoch_order = np.random.default_rng(
        SEED + epoch
    ).permutation(train_idx)

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    for step, index in enumerate(epoch_order, start=1):
        ids, mask, target = make_windows(index)

        with torch.autocast("cuda", dtype=torch.float16):
            m_logits, p_logits = model(ids, mask)
            loss = calculate_loss(m_logits, p_logits, target)

        if not torch.isfinite(loss):
            raise RuntimeError(f"出现无效损失，样本索引：{index}")

        running_loss += loss.item()
        scaler.scale(loss / accumulation).backward()

        if step % accumulation == 0 or step == len(epoch_order):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0 or step == len(epoch_order):
            print(
                f"Epoch {epoch+1} | "
                f"{step}/{len(epoch_order)} | "
                f"loss={running_loss/step:.4f}",
                flush=True
            )

    # ASAP 内部留出集，仅用来检查辅助任务学习情况
    model.eval()
    losses = []
    tp = fp = fn = correct = sentiment_total = 0

    with torch.inference_mode():
        for index in val_idx:
            ids, mask, target = make_windows(index)
            with torch.autocast("cuda", dtype=torch.float16):
                m_logits, p_logits = model(ids, mask)
                val_loss = calculate_loss(
                    m_logits, p_logits, target
                )

            losses.append(val_loss.item())

            truth = target != -2
            prediction = torch.sigmoid(m_logits) >= 0.5
            tp += (prediction & truth).sum().item()
            fp += (prediction & ~truth).sum().item()
            fn += (~prediction & truth).sum().item()

            correct += (
                p_logits[truth].argmax(dim=-1)
                == target[truth] + 1
            ).sum().item()
            sentiment_total += truth.sum().item()

    mention_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    sentiment_acc = correct / max(sentiment_total, 1)
    print(
        f"VALIDATION | loss={np.mean(losses):.4f} | "
        f"mention_F1={mention_f1:.4f} | "
        f"sentiment_acc={sentiment_acc:.4f}",
        flush=True
    )

if not args.smoke:
    output.mkdir(parents=True, exist_ok=True)
    encoder_dir = output / "encoder"
    model.encoder.save_pretrained(str(encoder_dir))
    tokenizer.save_pretrained(str(encoder_dir))

    torch.save({
        "mention": model.mention.state_dict(),
        "polarity": model.polarity.state_dict(),
        "aspects": aspects,
    }, output / "aux_heads.pt")

    manifest = {
        "seed": SEED,
        "epochs": args.epochs,
        "window_tokens": 256,
        "window_stride": 192,
        "train_ids": df.iloc[full_train_idx]["id"].astype(str).tolist(),
        "val_ids": df.iloc[order[:500]]["id"].astype(str).tolist(),
        "source_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "status": "ASAP auxiliary training; not SIGHAN VA evaluation"
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("Saved encoder:", encoder_dir, flush=True)

print("FINISHED", flush=True)
