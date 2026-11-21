"""Re-score six methods and three seeded BERT pairs, with figures for the course project."""
import contextlib
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_experiments import METHODS, main as compare_all
from main import O, inputs, truth
from scoring import task1_score
from window_experiment import extract_scores

OUT = O / "experiment_analysis"
SEEDS = (42, 17, 2026)
BERT_METHODS = ("bert_sentence", "bert_aspect")
METRICS = ("Valance MAE", "Valance PCC", "Arousal MAE", "Arousal PCC")
LABELS = ("V-MAE ↓", "V-PCC ↑", "A-MAE ↓", "A-PCC ↑")
METHOD_NAMES = {
    "sentence_only": "岭回归\n仅整句", "baseline": "岭回归\n整句＋方面",
    "improved": "岭回归\n加目标分句", "window_experiment": "岭回归\n再加下一句",
    "bert_sentence": "BERT\n仅整句", "bert_aspect": "BERT\n方面句对",
}
PARAMETERS = ("epochs", "batch_size", "max_length", "learning_rate", "warmup_steps",
              "optimizer", "weight_decay", "gradient_clip_norm", "loss", "selection",
              "train_rows", "dev_rows", "test_rows")
COLORS = ("#94B9B1", "#779E99", "#487F79", "#95A4A6", "#6397BF", "#16776E")


def save_csv(data, name):
    data.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def save_figure(fig, name):
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)


def folder_for(seed, method):
    return O / method if seed == 42 else O / "repeats" / f"seed_{seed}" / method


def read_and_verify_runs():
    records, configurations, hashes = [], [], {}
    expected_parameters = None
    for seed in SEEDS:
        for method in BERT_METHODS:
            folder = folder_for(seed, method)
            log_path = folder / "run_log.json"
            if not log_path.exists():
                raise FileNotFoundError(f"Finish this run first: {log_path}")
            log = json.loads(log_path.read_text(encoding="utf-8"))
            if log["seed"] != seed or log["method"] != method or log["smoke"]:
                raise ValueError(f"Unexpected run identity: {folder}")
            parameters = {key: log[key] for key in PARAMETERS}
            if expected_parameters is None:
                expected_parameters = parameters
            if parameters != expected_parameters:
                raise ValueError(f"Training settings differ: {folder}")
            best = min(log["history"], key=lambda row: row["dev_mean_MAE"])
            if best["epoch"] != log["selected_epoch"]:
                raise ValueError(f"Selected epoch does not minimize dev MAE: {folder}")
            configurations.append(dict(seed=seed, method=method,
                                       selected_epoch=log["selected_epoch"], **parameters))
            hashes[str(log_path.relative_to(O))] = hashlib.sha256(log_path.read_bytes()).hexdigest()
            for split, official_name in (("dev", "Validation"), ("test", "Testing")):
                rows, gold = inputs(official_name), truth(official_name)
                pred_path = folder / f"{split}_predictions.txt"
                pred = pd.read_csv(pred_path, sep=" ", keep_default_na=False)
                if not pred.ID.is_unique or pred.ID.tolist() != gold.ID.tolist():
                    raise ValueError(f"Sentence coverage/order mismatch: {pred_path}")
                values = extract_scores(pred, rows)
                if not np.isfinite(values).all() or not ((values >= 1) & (values <= 9)).all():
                    raise ValueError(f"Invalid scores: {pred_path}")
                recomputed = task1_score(pred, gold, {})
                saved = json.loads((folder / f"{split}_scores.json").read_text(encoding="utf-8"))
                for key in METRICS:
                    if not np.isclose(recomputed[key], saved[key], atol=1e-9, rtol=0):
                        raise ValueError(f"Saved/recomputed metric mismatch: {folder} {key}")
                    if split == "dev" and not np.isclose(recomputed[key], best[key], atol=1e-9, rtol=0):
                        raise ValueError(f"Prediction/best-epoch metric mismatch: {folder} {key}")
                records.append(dict(seed=seed, method=method, split=split, selected_epoch=log["selected_epoch"],
                                    **{key: float(recomputed[key]) for key in METRICS}))
                hashes[str(pred_path.relative_to(O))] = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    return pd.DataFrame(records), configurations, hashes


def summarize(records):
    summary = []
    for (split, method), group in records.groupby(["split", "method"], sort=False):
        for metric in METRICS:
            summary.append(dict(split=split, method=method, metric=metric, n_seeds=len(group),
                                mean=group[metric].mean(), sample_sd=group[metric].std(ddof=1),
                                minimum=group[metric].min(), maximum=group[metric].max()))
    paired = []
    for (split, seed), group in records.groupby(["split", "seed"], sort=False):
        lookup = group.set_index("method")
        for metric in METRICS:
            sentence, aspect = lookup.loc["bert_sentence", metric], lookup.loc["bert_aspect", metric]
            delta = aspect - sentence
            paired.append(dict(split=split, seed=seed, metric=metric, sentence_value=sentence,
                               aspect_value=aspect, aspect_minus_sentence=delta,
                               improvement=bool(delta < 0 if "MAE" in metric else delta > 0)))
    return pd.DataFrame(summary), pd.DataFrame(paired)


def plot_six_methods():
    metrics = pd.read_csv(O / "comparison" / "metrics.csv")
    test = metrics[metrics.split == "test"].set_index("method").loc[list(METHODS)]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for ax, metric, label in zip(axes.flat, METRICS, LABELS):
        bars = ax.bar(np.arange(len(METHODS)), test[metric], color=COLORS, width=.62)
        ax.bar_label(bars, fmt="%.4f", fontsize=10, padding=3)
        ax.set_xticks(np.arange(len(METHODS)), [METHOD_NAMES[m] for m in METHODS], fontsize=10)
        ax.set(title=label, ylabel="分" if "MAE" in metric else "相关系数")
        ax.set_ylim(0, 1 if "PCC" in metric else float(test[metric].max()) * 1.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("六组方法的官方测试指标 · BERT 为随机种子 42 的归档结果", fontsize=16)
    save_figure(fig, "six_method_metrics.png")
    save_csv(metrics, "six_method_metrics.csv")


def plot_training():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    histories = []
    for ax, method in zip(axes, BERT_METHODS):
        hist = pd.read_csv(O / method / "training_history.csv")
        log = json.loads((O / method / "run_log.json").read_text(encoding="utf-8"))
        hist["method"], hist["seed"] = method, 42
        histories.append(hist)
        ax.plot(hist.epoch, hist.train_mean_l1, "o-", color="#6397BF", label="训练平均 L1")
        ax.plot(hist.epoch, hist.dev_mean_MAE, "s-", color="#16776E", label="开发集平均 MAE")
        best = hist[hist.epoch == log["selected_epoch"]].iloc[0]
        ax.scatter([best.epoch], [best.dev_mean_MAE], marker="*", color="#B86B32", s=180, zorder=5, label="选中检查点")
        ax.set(title=METHOD_NAMES[method].replace("\n", " · "), xlabel="训练轮次", ylabel="分")
        ax.set_xticks([1, 2, 3])
        ax.set_ylim(0, float(hist[["train_mean_l1", "dev_mean_MAE"]].max().max()) * 1.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        ax.legend(fontsize=10)
    fig.suptitle("随机种子 42 的训练与选模 · 按开发集平均 MAE 选择第 2 轮", fontsize=14)
    save_figure(fig, "training_curves_seed42.png")
    save_csv(pd.concat(histories, ignore_index=True), "training_curves_seed42.csv")


def plot_repeats(records, summary):
    test = records[records.split == "test"]
    means = summary[summary.split == "test"].set_index(["method", "metric"])
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), constrained_layout=True)
    offsets = (-.07, 0, .07)
    markers = ("o", "s", "^")
    for ax, metric, label in zip(axes.flat, METRICS, LABELS):
        for seed, offset, marker in zip(SEEDS, offsets, markers):
            values = test[test.seed == seed].set_index("method").loc[list(BERT_METHODS), metric]
            ax.plot(np.array([0, 1]) + offset, values, marker=marker, linestyle="--",
                    alpha=.62, linewidth=1, label=f"种子 {seed}")
        for x, method in enumerate(BERT_METHODS):
            row = means.loc[(method, metric)]
            ax.errorbar(x + .18, row["mean"], yerr=row["sample_sd"], fmt="D",
                        color="#173E40", capsize=6, markersize=6,
                        label="均值 ± 样本标准差" if x == 0 else None)
        ax.set_xticks([0, 1], ["BERT 仅整句", "BERT 方面句对"])
        ax.set(xlim=(-.4, 1.48), title=label, ylabel="分" if "MAE" in metric else "相关系数")
        ax.set_ylim(0, 1 if "PCC" in metric else float(test[metric].max()) * 1.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=10)
    fig.suptitle("三次测试结果 · 连线配对相同随机种子，误差棒表示运行间波动", fontsize=14)
    save_figure(fig, "three_seed_metrics.png")


def write_report(summary, paired, configurations):
    test = summary[summary.split == "test"].set_index(["method", "metric"])
    lines = ["# 实验复核与随机种子重复实验", "",
             "六组原有预测和三种子 BERT 的每份开发、测试预测均通过官方评分复算。句子 ID、方面覆盖和数值范围均检查通过。",
             "", "## 三次测试结果", "",
             "|模型|V-MAE|V-PCC|A-MAE|A-PCC|",
             "|---|---:|---:|---:|---:|"]
    for method in BERT_METHODS:
        cells = [f"{test.loc[(method, metric), 'mean']:.4f} ± {test.loc[(method, metric), 'sample_sd']:.4f}" for metric in METRICS]
        lines.append("|" + METHOD_NAMES[method].replace("\n", " ") + "|" + "|".join(cells) + "|")
    lines += ["", "表中为种子 42、17、2026 的均值和样本标准差（ddof=1）。每个模型都使用相同的种子列表和训练设置，每次按开发集两维平均 MAE 选定检查点。",
              "", "## 配对比较", "",
              "|指标|方面句对减仅整句的平均差值|改善的种子数|",
              "|---|---:|---:|"]
    for metric in METRICS:
        subset = paired[(paired.split == "test") & (paired.metric == metric)]
        lines.append(f"|{metric}|{subset.aspect_minus_sentence.mean():+.4f}|{int(subset.improvement.sum())}/{len(subset)}|")
    lines += ["", "MAE 差值为负表示误差下降，PCC 差值为正表示相关性上升。三次训练在测试集四项指标上均呈现方面句对优于仅整句的方向。",
              "", "六组图展示了两步变化。字符模型增加目标分句后优于整句加方面词基线，继续扩大到下一分句后四项总体测试指标下降。BERT 的语义表示进一步降低误差，在同一个 BERT 结构中显式输入方面词又带来额外改善。",
              "", "种子 42 的训练曲线显示，两种 BERT 的训练损失到第 3 轮继续下降，但开发集两维平均 MAE 在第 2 轮最低。选模使用开发集检查点，随后计算测试结果。训练曲线为训练过程的平均 L1，开发集 MAE 来自裁剪并保留两位小数后的官方预测。",
              "", "本次只有三个随机种子，误差棒表示运行间的样本标准差。开发集仅 100 句、115 个方面，未进行显著性检验。前一阶段已分析过测试案例，新增重复实验用于检验固定方案的运行稳定性。",
              "", "## 可复查文件", "",
              "- seed_metrics.csv 保存每个种子、模型、划分的四项指标及选中轮次。",
              "- seed_summary.csv 保存开发集与测试集的均值、样本标准差、最小值和最大值。",
              "- paired_differences.csv 按同一种子比较方面句对与整句模型。",
              "- verification_log.json 记录配置一致性和预测文件 SHA-256，six_method_verification.txt 保存原六组复核输出。",
              "- six_method_metrics.png、training_curves_seed42.png、three_seed_metrics.png 可直接用于汇报。"]
    (OUT / "实验分析说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False,
                         "font.size": 11, "figure.facecolor": "white"})
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        compare_all()
    (OUT / "six_method_verification.txt").write_text(captured.getvalue(), encoding="utf-8")
    records, configs, hashes = read_and_verify_runs()
    summary, paired = summarize(records)
    save_csv(records, "seed_metrics.csv")
    save_csv(summary, "seed_summary.csv")
    save_csv(paired, "paired_differences.csv")
    save_csv(pd.DataFrame(configs), "run_configurations.csv")
    plot_six_methods()
    plot_training()
    plot_repeats(records, summary)
    write_report(summary, paired, configs)
    (OUT / "verification_log.json").write_text(json.dumps({
        "seeds": list(SEEDS), "methods": list(BERT_METHODS), "n_model_runs": len(configs),
        "n_prediction_files": len(records), "standard_deviation_ddof": 1,
        "all_official_scores_recomputed": True, "all_settings_consistent": True,
        "prediction_sha256_and_logs": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary[summary.split == "test"].to_string(index=False))
    print(paired[paired.split == "test"].to_string(index=False))
    print("Saved experiment analysis to", OUT)


if __name__ == "__main__":
    main()

