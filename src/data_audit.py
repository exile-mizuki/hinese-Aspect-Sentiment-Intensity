"""Check official splits and aspect aggregation, saving reproducible statistics."""
import hashlib
import json
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import D, O, inputs, train, truth
from window_experiment import extract_scores

OUT = O / "data_audit"


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def main():
    OUT.mkdir(exist_ok=True)
    raw_records, issues = [], []
    for path in sorted(D.glob("*Training*.json")):
        for record in json.loads(path.read_text(encoding="utf-8-sig")):
            if len(record["Aspect"]) != len(record["Intensity"]):
                issues.append({"file": path.name, "id": record["ID"], "issue": "aspect/intensity count mismatch"})
                continue
            for i, (aspect, value) in enumerate(zip(record["Aspect"], record["Intensity"])):
                try:
                    v, a = map(float, value.split("#"))
                except (ValueError, AttributeError):
                    v, a = np.nan, np.nan
                raw_records.append(dict(source=path.name, id=record["ID"], text=record["Sentence"],
                                        aspect=aspect, opinion=record["Opinion"][i], V=v, A=a))
    raw = pd.DataFrame(raw_records)
    save(raw.groupby("source", as_index=False).agg(
        raw_annotations=("id", "size"), sentences=("id", "nunique")
    ), "raw_source_counts.csv")
    bad = raw[~np.isfinite(raw[["V", "A"]]).all(axis=1) | ~raw[["V", "A"]].apply(lambda c: c.between(1, 9)).all(axis=1)]
    save(bad, "invalid_raw_labels.csv")
    (OUT / "structural_issues.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
    if issues or len(bad):
        raise ValueError("Invalid training rows were recorded; inspect before aggregation.")
    training, nraw = train()
    assert nraw == len(raw)
    frames = {"train": training}
    for split, tag in (("dev", "Validation"), ("test", "Testing")):
        rows = inputs(tag)
        rows[["V", "A"]] = extract_scores(truth(tag), rows)
        frames[split] = rows
    stats = []
    for name, df in frames.items():
        counts = df.groupby("id").aspect.nunique()
        multi_ids = counts[counts > 1].index
        stats.append({
            "数据集": name, "句子数": df.id.nunique(), "方面实例数": len(df),
            "空文本实例数": df.text.str.strip().eq("").sum(),
            "NULL方面实例数": df.aspect.eq("NULL").sum(),
            "空方面实例数": df.aspect.str.strip().eq("").sum(),
            "缺失标签实例数": df[["V", "A"]].isna().any(axis=1).sum(),
            "非有限或越界标签实例数": (~np.isfinite(df[["V", "A"]]).all(axis=1) | ~df[["V", "A"]].apply(lambda c: c.between(1, 9)).all(axis=1)).sum(),
            "重复ID方面实例数": df.duplicated(["id", "aspect"]).sum(),
            "重复文本方面实例数": df.duplicated(["text", "aspect"]).sum(),
            "多方面句子数": len(multi_ids), "多方面句比例": len(multi_ids) / df.id.nunique(),
            "多方面句涉及实例数": df.id.isin(multi_ids).sum(),
        })
    summary = pd.DataFrame(stats)
    save(summary, "split_statistics.csv")
    summary_overlaps, overlaps = [], []
    for (left, ldf), (right, rdf) in combinations(frames.items(), 2):
        lkeys = set(zip(ldf.text, ldf.aspect))
        rkeys = set(zip(rdf.text, rdf.aspect))
        intersection = lkeys & rkeys
        summary_overlaps.append(dict(left=left, right=right, exact_text_aspect_pairs=len(intersection)))
        for text, aspect in sorted(intersection):
            lg = ldf[(ldf.text == text) & (ldf.aspect == aspect)]
            rg = rdf[(rdf.text == text) & (rdf.aspect == aspect)]
            overlaps.append(dict(left=left, right=right, text=text, aspect=aspect,
                                 left_ids=";".join(lg.id), right_ids=";".join(rg.id),
                                 left_labels=";".join(f"{r.V}#{r.A}" for r in lg.itertuples()),
                                 right_labels=";".join(f"{r.V}#{r.A}" for r in rg.itertuples())))
    save(pd.DataFrame(summary_overlaps), "cross_split_overlap_counts.csv")
    save(pd.DataFrame(overlaps), "cross_split_overlap_details.csv")
    counts = raw.groupby(["id", "text", "aspect"], sort=False).size()
    example_id, example_text, example_aspect = counts[counts > 1].index[0]
    example = raw[(raw.id == example_id) & (raw.aspect == example_aspect)].copy()
    save(example, "aggregation_example_raw.csv")
    average = example.groupby(["id", "text", "aspect"], as_index=False)[["V", "A"]].mean()
    save(average, "aggregation_example_mean.csv")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    hist_rows = []
    bins = np.arange(1, 9.5, .5)
    for ax, dim, title, color in zip(axes, ("V", "A"), ("愉悦度 V", "唤醒度 A"), ("#16776E", "#457AA0")):
        values, edges = np.histogram(training[dim], bins=bins)
        ax.hist(training[dim], bins=bins, color=color, edgecolor="white")
        ax.set(xlabel=f"{title}（1～9 分）", ylabel="方面实例数", title=title, xlim=(1, 9))
        ax.spines[["top", "right"]].set_visible(False)
        hist_rows.extend(dict(dimension=dim, lower=lo, upper=hi, count=int(n)) for lo, hi, n in zip(edges[:-1], edges[1:], values))
    fig.suptitle("训练集情感强度分布 · 同方面取均值后的 7,476 个实例")
    fig.savefig(OUT / "label_distribution.png", dpi=180)
    plt.close(fig)
    save(pd.DataFrame(hist_rows), "distribution_counts.csv")
    (OUT / "audit_log.json").write_text(json.dumps({
        "raw_annotations": len(raw), "merged_instances": len(training),
        "group_key": ["id", "text", "aspect"], "overlap_rule": "exact text and aspect; no normalization",
        "raw_invalid_labels": len(bad), "structural_issues": len(issues),
        "data_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(D.iterdir()) if f.is_file()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    r = average.iloc[0]
    note = (f"# 数据检查结果\n\n训练集 {len(raw):,} 条标注按句子 ID、文本和方面取两维均值，合并成 {len(training):,} 个实例。"
            "统计表中的空文本、NULL 和标签检查均按方面实例计数，多方面比例按句子计数。NULL 表示隐含方面，保留参与训练。\n\n"
            f"取均值的实例为 {example_id}，原句“{example_text}”，方面“{example_aspect}”。"
            f"{len(example)} 条原始标注的 V 均值为 {r.V:.4f}，A 均值为 {r.A:.4f}。原始意见与评分见 aggregation_example_raw.csv。\n\n"
            "跨划分重复按完全一致的“文本＋方面”识别，数量及 ID 见 cross_split_overlap_counts.csv 和 cross_split_overlap_details.csv。主实验沿用官方划分，重复情况作为数据限制披露。\n\n"
            "label_distribution.png 的分数已按方面合并，横轴每 0.5 分分箱，最后一箱包含 9 分。分布反映课程语料的标签分布。\n")
    (OUT / "数据说明.md").write_text(note, encoding="utf-8")
    print(summary.to_string(index=False))
    print(pd.DataFrame(summary_overlaps).to_string(index=False))


if __name__ == "__main__":
    main()

