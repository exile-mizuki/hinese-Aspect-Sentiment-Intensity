"""用相同官方输入、真值和评分脚本复核各方法，包括同句严格排序。"""
import json

import numpy as np
import pandas as pd

from main import O, inputs, truth
from scoring import task1_score
from window_experiment import diagnosis, extract_scores

METHODS = ("sentence_only", "baseline", "improved", "window_experiment", "bert_sentence", "bert_aspect")


def main():
    out = O / "comparison"
    out.mkdir(exist_ok=True)
    metrics, pairs, slices = [], [], []
    for split, tag in (("dev", "Validation"), ("test", "Testing")):
        rows = inputs(tag)
        reference = truth(tag)
        rows["multi"] = rows.groupby("id").aspect.transform("nunique") > 1
        rows[["true_V", "true_A"]] = extract_scores(reference, rows)
        for method in METHODS:
            folder = O / method
            predictions = pd.read_csv(folder / f"{split}_predictions.txt", sep=" ")
            if predictions.ID.tolist() != reference.ID.tolist():
                raise ValueError(f"{method}: sentence ID mismatch")
            values = extract_scores(predictions, rows)
            if not np.isfinite(values).all() or not ((1 <= values) & (values <= 9)).all():
                raise ValueError(f"{method}: invalid values")
            rows[[f"{method}_V", f"{method}_A"]] = values
            scores = task1_score(predictions, reference, {})
            if method == "window_experiment":
                saved = pd.read_csv(folder / "metrics.csv").query("method == 'next_clause' and split == @split").iloc[0]
            else:
                saved = json.loads((folder / f"{split}_scores.json").read_text(encoding="utf-8"))
            for key, value in scores.items():
                if not np.isclose(value, saved[key], atol=1e-9, rtol=0):
                    raise ValueError(f"{method}: {split} {key} differs from its run")
            metrics.append(dict(split=split, method=method, **scores))
        p, s = diagnosis(rows, split, METHODS)
        pairs.extend(p)
        slices.extend(s)
        rows.to_csv(out / f"{split}_cases.csv", index=False, encoding="utf-8-sig")
    for name, records in (("metrics", metrics), ("aspect_order", pairs), ("single_multi_metrics", slices)):
        pd.DataFrame(records).to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()

