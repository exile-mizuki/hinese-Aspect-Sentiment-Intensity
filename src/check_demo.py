"""Load the saved model locally on CPU and test five illustrative inputs."""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from bert_experiment import IntensityModel, infer_rows
from main import O

EXAMPLES = [
    ("单方面", "牛肉很嫩。", ["牛肉"]),
    ("多方面", "面条有嚼劲，汤头鲜美，服务也很热情。", ["面条", "汤头", "服务"]),
    ("转折", "蚵仔和小管新鲜，但肉燥普通。", ["蚵仔", "小管", "肉燥"]),
    ("否定", "鸡肉一点也不好吃。", ["鸡肉"]),
    ("跨分句", "牛肉的份量看起来不算少，吃起来也蛮软嫩的。", ["牛肉的份量", "牛肉"]),
]


def main():
    root = Path(__file__).resolve().parents[1]
    out = O / "demo_check"
    out.mkdir(exist_ok=True)
    started = time.perf_counter()
    tokenizer_dir = root / "demo_tokenizer"
    checkpoint = root / "checkpoints" / "bert_aspect" / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["mode"] != "aspect":
        raise ValueError("Expected the aspect-pair checkpoint.")
    model = IntensityModel(AutoModel.from_config(AutoConfig.from_pretrained(tokenizer_dir, local_files_only=True)))
    model.load_state_dict(state["model"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    rows = pd.DataFrame([dict(type=kind, text=text, aspect=aspect)
                         for kind, text, aspects in EXAMPLES for aspect in aspects])
    values = infer_rows(model, rows, "aspect", tokenizer, state["max_length"], 32, torch.device("cpu"))
    rows[["V", "A"]] = values.clip(1, 9).round(2)
    rows.to_csv(out / "demo_predictions.csv", index=False, encoding="utf-8-sig")
    record = dict(device="cpu", local_files_only=True, examples=len(EXAMPLES),
                  aspect_rows=len(rows), seconds=round(time.perf_counter()-started, 2),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  note="Illustrative inputs; no human reference scores, excluded from official evaluation.")
    (out / "run_log.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()

