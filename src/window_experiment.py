"""Compare the saved matched-clause model with a one-clause look-ahead window."""
import hashlib
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from main import D, O, inputs, local, save_pred, train, truth
from scoring import task1_score


OUTPUT = O / "window_experiment"
ALPHAS = (0.1, 1, 5, 15)
DELIMITERS = re.compile(r"[，,。；;！？!?]")
RATINGS = re.compile(r"\(([^,]+),([\d.]+)#([\d.]+)\)")


def extended_clause(sentence, aspect):
    """Gather matching clauses and their next nonempty clause in source order."""
    if aspect == "NULL":
        return sentence
    clauses = [clause for clause in DELIMITERS.split(sentence) if clause.strip()]
    selected = set()
    for i, clause in enumerate(clauses):
        if aspect in clause:
            selected.add(i)
            if i + 1 < len(clauses):
                selected.add(i + 1)
    return " ".join(clauses[i] for i in sorted(selected)) if selected else sentence


def vectorize(splits):
    blocks = [[] for _ in splits]
    for column in ("text", "aspect", "extended_clause"):
        tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3),
                                sublinear_tf=True, min_df=1)
        blocks[0].append(tfidf.fit_transform(splits[0][column]))
        for idx, part in enumerate(splits[1:], 1):
            blocks[idx].append(tfidf.transform(part[column]))
    return [hstack(parts, format="csr") for parts in blocks]


def extract_scores(packed, expected):
    keys, ratings = [], []
    for row in packed.itertuples(index=False):
        for aspect, v, a in RATINGS.findall(row.Intensity):
            keys.append((row.ID, aspect))
            ratings.append((float(v), float(a)))
    frame = pd.DataFrame(ratings, columns=["V", "A"],
                         index=pd.MultiIndex.from_tuples(keys, names=["id", "aspect"]))
    index = pd.MultiIndex.from_frame(expected[["id", "aspect"]])
    if not frame.index.is_unique or not index.is_unique or set(frame.index) != set(index):
        raise ValueError("Prediction/reference IDs or aspects do not match official inputs.")
    return frame.loc[index].to_numpy()


def diagnosis(frame, split, methods=("matched_clause", "next_clause")):
    pairs, slices = [], []
    for method in methods:
        for group, subset in frame.groupby("multi"):
            row = dict(split=split, method=method, group="multi" if group else "single",
                       sentences=subset.id.nunique(), aspects=len(subset))
            for dim in ("V", "A"):
                row[dim + "_MAE"] = (subset[method + "_" + dim] - subset["true_" + dim]).abs().mean()
            slices.append(row)
        for dim in ("V", "A"):
            correct = tied = reversed_count = 0
            for _, group in frame[frame.multi].groupby("id"):
                labels = group[["true_" + dim, method + "_" + dim]].to_numpy()
                for (left, pred_left), (right, pred_right) in combinations(labels, 2):
                    if left == right:
                        continue
                    sign = np.sign((left - right) * (pred_left - pred_right))
                    correct += int(sign > 0)
                    tied += int(sign == 0)
                    reversed_count += int(sign < 0)
            n = correct + tied + reversed_count
            pairs.append(dict(split=split, method=method, dimension=dim, eligible_pairs=n,
                              correct=correct, tied=tied, reversed=reversed_count,
                              strict_correct_rate=correct / n if n else None))
    return pairs, slices


def main():
    start = time.perf_counter()
    OUTPUT.mkdir(exist_ok=True)
    training, raw_count = train()
    dev, test = inputs("Validation"), inputs("Testing")
    splits = (training, dev, test)
    for frame in splits:
        frame["extended_clause"] = [extended_clause(r.text, r.aspect) for r in frame.itertuples()]
    train_x, dev_x, test_x = vectorize(splits)
    reference = truth("Validation")
    log = []
    best_loss = float("inf")
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5).fit(train_x, training[["V", "A"]])
        pred = save_pred(dev, model.predict(dev_x), None)
        scores = task1_score(pred, reference, {})
        mean_mae = (scores["Valance MAE"] + scores["Arousal MAE"]) / 2
        log.append(dict(alpha=alpha, dev_mean_MAE=mean_mae, **scores))
        if mean_mae < best_loss:
            best_loss, best_alpha, best_model = mean_mae, alpha, model

    old_folder = O / "improved"
    old_alpha = json.loads((old_folder / "run_log.json").read_text())["selected_alpha"]
    summary, order, slices = [], [], []
    for split, frame, matrix, gold in (("dev", dev, dev_x, reference),
                                       ("test", test, test_x, truth("Testing"))):
        old = pd.read_csv(old_folder / f"{split}_predictions.txt", sep=" ")
        new = save_pred(frame, best_model.predict(matrix), None)
        new.to_csv(OUTPUT / f"{split}_predictions.txt", sep=" ", index=False, lineterminator="\n")
        cases = frame.copy()
        cases["matched_clause"] = [local(r.text, r.aspect) for r in frame.itertuples()]
        cases["multi"] = cases.groupby("id").aspect.transform("nunique") > 1
        cases[["true_V", "true_A"]] = extract_scores(gold, frame)
        for method, pred, alpha in (("matched_clause", old, old_alpha),
                                    ("next_clause", new, best_alpha)):
            cases[[method + "_V", method + "_A"]] = extract_scores(pred, frame)
            scores = task1_score(pred, gold, {})
            if method == "matched_clause":
                original = json.loads((old_folder / f"{split}_scores.json").read_text())
                if any(not np.isclose(value, original[k]) for k, value in scores.items()):
                    raise ValueError("Saved matched-clause scores could not be reproduced.")
            summary.append(dict(method=method, split=split, alpha=alpha, **scores))
        cases.to_csv(OUTPUT / f"{split}_cases.csv", index=False, encoding="utf-8-sig")
        pair_rows, slice_rows = diagnosis(cases, split)
        order.extend(pair_rows)
        slices.extend(slice_rows)

    for name, records in (("metrics", summary), ("aspect_order", order),
                          ("single_multi_metrics", slices)):
        pd.DataFrame(records).to_csv(OUTPUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    record = dict(selection="minimum mean of V and A official development MAE", selected_alpha=best_alpha,
                  candidates=log, reused_baseline="results/improved", raw_training_tuples=raw_count,
                  instances=[len(f) for f in splits], splits=["train", "dev", "test"],
                  tfidf="separate train-only char 1-3-gram per sentence, aspect, and local context; sublinear TF",
                  estimator="Ridge(solver='lsqr', tol=1e-5), threadpool_limits(1)",
                  local_context="matching clauses + following nonempty clause; union in original order; unmatched or NULL uses whole sentence",
                  python=sys.version, sklearn=sklearn.__version__, scipy=scipy.__version__,
                  numpy=np.__version__, pandas=pd.__version__, elapsed_seconds=round(time.perf_counter() - start, 2),
                  data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob("*")) if p.is_file()},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUTPUT / "run_log.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(order).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

