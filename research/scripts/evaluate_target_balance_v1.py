"""Audit fixed Train-only OOF results; never load official Dev/Test or train models.

Compare the same seed, rows and frozen folds. Bootstrap resamples connected review
groups and is conditional on these trained folds; it is NOT a multi-seed CI.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def training_rows(project):
    files = sorted((project / "data").glob("*Training*.json"))
    if not files:
        raise FileNotFoundError("No Training JSON under actual project/data")
    records = []
    for path in files:
        for row in read_json(path):
            if len(row["Aspect"]) != len(row["Intensity"]):
                raise ValueError("Aspect/Intensity length mismatch")
            for aspect, intensity in zip(row["Aspect"], row["Intensity"]):
                v, a = map(float, intensity.split("#"))
                records.append([row["ID"], row["Sentence"], aspect, v, a])
    rows = pd.DataFrame(records, columns=["id", "text", "aspect", "V", "A"])
    rows = rows.groupby(["id", "text", "aspect"], sort=False, as_index=False)[["V", "A"]].mean()
    return rows, {p.name: sha(p) for p in files}, len(records)


def assert_rows(actual, expected, label):
    if len(actual) != len(expected):
        raise ValueError(f"{label}: row count mismatch")
    for col in ("id", "text", "aspect"):
        if not np.array_equal(actual[col].to_numpy(), expected[col].to_numpy()):
            raise ValueError(f"{label}: {col} or row order mismatch")
    if not np.allclose(actual[["V", "A"]].to_numpy(float),
                       expected[["V", "A"]].to_numpy(float), atol=1e-6, rtol=0):
        raise ValueError(f"{label}: source label mismatch")


def load_predictions(folder, manifest, source_hashes, manifest_hash, seed):
    frames, configs, files, warnings = [], [], {}, []
    common = {"manifest_sha256": manifest_hash, "rows": len(manifest),
              "fold_count": int(manifest.fold.nunique()), "epochs": 3,
              "batch_size": 8, "learning_rate": 2e-5, "max_length": 128,
              "input_variant": "marked", "dev_test_access": False,
              "method": "marked_l1", "method_config": ["l1", 0.0, False], "seed": seed}
    for fold in sorted(manifest.fold.unique()):
        run = folder / f"seed_{seed}" / f"fold_{fold}"
        pred_file, config_file, history_file = [run / name for name in
            ("heldout_predictions.csv", "config.json", "history.json")]
        # config is written last by the runner; all three must exist.
        if not all(p.is_file() for p in (pred_file, config_file, history_file)):
            raise FileNotFoundError(f"Incomplete fold; wait for SAVED: {run}")
        config, history = read_json(config_file), read_json(history_file)
        for key, value in {**common, "fold": int(fold)}.items():
            if config.get(key) != value:
                raise ValueError(f"{run}: config mismatch {key}: {config.get(key)!r} != {value!r}")
        recorded_hashes = config.get("training_source_file_sha256")
        if not recorded_hashes:
            warnings.append(f"{run}: source hashes missing in original config; current data/manifest verified independently; original config retained")
        elif recorded_hashes != source_hashes:
            raise ValueError(f"{run}: recorded Training JSON hashes differ")
        if [r["epoch"] for r in history["history"]] != [1, 2, 3]:
            raise ValueError(f"{run}: incomplete fixed-epoch training history")
        expected = manifest.loc[manifest.fold == fold].reset_index(drop=True)
        frame = read_csv(pred_file)
        assert_rows(frame, expected, str(pred_file))
        for col in ("fold", "group"):
            if not np.array_equal(frame[col].to_numpy(), expected[col].to_numpy()):
                raise ValueError(f"{run}: {col} mismatch")
        predictions = frame[["pred_V", "pred_A"]].to_numpy(float)
        if not np.isfinite(predictions).all() or not ((predictions >= 1) & (predictions <= 9)).all():
            raise ValueError(f"{run}: invalid predictions")
        # Align by frozen row index, not an assumed-unique id/aspect join.
        frame["row_index"] = expected.row_index.to_numpy()
        frames.append(frame)
        configs.append(config)
        for path in (pred_file, config_file, history_file):
            files[str(path)] = sha(path)
    joined = pd.concat(frames).sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(joined.row_index.to_numpy(), np.arange(len(manifest))):
        raise ValueError("Missing/duplicate OOF rows")
    return joined, configs, files, warnings


def cluster_ci(difference, groups, repetitions, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        return None
    counts = np.bincount(inverse)
    sums = np.stack([np.bincount(inverse, weights=difference[:, i])
                     for i in range(difference.shape[1])], axis=1)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repetitions, 100):
        sampled = rng.integers(0, len(unique), size=(min(100, repetitions-start), len(unique)))
        draws.append(sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)[:, None])
    intervals = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
    return {name: [float(intervals[0, j]), float(intervals[1, j])]
            for j, name in enumerate(("V_MAE", "A_MAE", "mean_MAE"))}


def evaluate(args):
    root, project = args.root.resolve(), args.project.resolve()
    manifest_path = root / "validation" / "proposed_3fold_manifest.csv"
    m = read_csv(manifest_path)
    rows, hashes, raw_n = training_rows(project)
    assert_rows(m, rows, "manifest vs actual Training JSON")
    if not np.array_equal(m.row_index.to_numpy(), np.arange(len(rows))):
        raise ValueError("Invalid frozen row_index")
    if set(m.fold) != {0, 1, 2} or m.groupby("group").fold.nunique().max() != 1:
        raise ValueError("Invalid frozen 3-fold grouping")
    review = m.id.str.split(":").str[0]
    for fold in (0, 1, 2):
        held = m.fold == fold
        if set(review[held]) & set(review[~held]) or set(m.text[held]) & set(m.text[~held]):
            raise ValueError("Cross-fold review/text leakage")
    baseline, bc, bf, bw = load_predictions(root / "training_v2" / "marked_l1", m, hashes, sha(manifest_path), args.seed)
    candidate, cc, cf, cw = load_predictions(root / "target_balanced_v1" / "training_v2" / "marked_l1", m, hashes, sha(manifest_path), args.seed)
    for left, right in zip(bc, cc):
        if Path(left["model_path"]).resolve() != Path(right["model_path"]).resolve():
            raise ValueError("Pretrained encoder paths differ")
        if right.get("architecture") != "bert_target_balanced_v1_m125_extreme3":
            raise ValueError("Unexpected candidate architecture")
    y = m[["V", "A"]].to_numpy(float)
    be = np.abs(baseline[["pred_V", "pred_A"]].to_numpy(float)-y)
    ce = np.abs(candidate[["pred_V", "pred_A"]].to_numpy(float)-y)
    aspect_n = m.groupby(["id", "text"], dropna=False).aspect.transform("nunique").to_numpy()
    subsets = {"OVERALL": np.ones(len(m), bool), "SINGLE": aspect_n == 1,
               "MULTI": aspect_n > 1, "LOW_V": y[:, 0] < 3,
               "HIGH_A": y[:, 1] >= 6.5, "EXTREME_A": y[:, 1] >= 7.5,
               "MULTI_EXTREME_A": (aspect_n > 1) & (y[:, 1] >= 7.5)}
    metrics, flat = {}, []
    for i, (name, mask) in enumerate(subsets.items()):
        if not mask.any():
            raise ValueError(f"Empty expected subgroup: {name}")
        base = np.r_[be[mask].mean(0), be[mask].mean()]
        cand = np.r_[ce[mask].mean(0), ce[mask].mean()]
        diff = ce[mask]-be[mask]
        diff = np.column_stack([diff, diff.mean(1)])
        names = ["V_MAE", "A_MAE", "mean_MAE"]
        result = {"n": int(mask.sum()), "groups": int(m.loc[mask, "group"].nunique()),
                  "baseline": dict(zip(names, base.tolist())),
                  "candidate": dict(zip(names, cand.tolist())),
                  "delta_candidate_minus_baseline": dict(zip(names, (cand-base).tolist())),
                  "paired_group_bootstrap_95ci": cluster_ci(diff, m.loc[mask, "group"].to_numpy(), args.bootstrap, 20261120+i)}
        metrics[name] = result
        flat.append({"subgroup": name, "n": result["n"], "groups": result["groups"],
                     **{f"{system}_{metric}": result[system][metric]
                        for system in ("baseline", "candidate", "delta_candidate_minus_baseline") for metric in names}})
    delta = lambda subset, metric: metrics[subset]["delta_candidate_minus_baseline"][metric]
    checks = {"overall_mean_delta_le_0.003": delta("OVERALL", "mean_MAE") <= .003,
              "extreme_A_MAE_decreases": delta("EXTREME_A", "A_MAE") < 0,
              "low_V_MAE_decreases": delta("LOW_V", "V_MAE") < 0}
    result = {"seed": args.seed, "raw_training_tuples": raw_n, "aggregated_training_rows": len(m),
              "manifest_sha256": sha(manifest_path), "actual_training_sha256": hashes,
              "input_artifact_sha256": {**bf, **cf}, "provenance_warnings": bw+cw,
              "official_dev_test_read": False, "original_files_modified": False,
              "bootstrap_repetitions": args.bootstrap,
              "bootstrap_scope": "Paired connected-group resampling of fixed OOF errors; conditional on fitted models; not a multi-seed or retraining CI; subgroup CIs exploratory without multiplicity correction",
              "metrics": metrics, "exploratory_screen_checks": checks,
              "exploratory_screen_pass": all(checks.values()), "deployment_accepted": False,
              "screen_scope": "Follow-up screen declared during this seed-42 run, before complete OOF results. Not the original full experiment preregistration. Require matched multi-seed confirmation and a separate deployment decision; no auto-promotion."}
    out = root / "target_balanced_v1" / "analysis"
    out.mkdir(exist_ok=True)
    (out / f"audit_seed{args.seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(flat).to_csv(out / "subgroup_comparison.csv", index=False, encoding="utf-8-sig")
    lines = ["分层加权实验：训练集三折独立复核", f"随机种子：{args.seed}；方面实例：{len(m)}；未读取官方 Dev/Test。",
             "误差越低越好；delta 为候选减基线，负数表示改善。", ""]
    for name, values in metrics.items():
        lines.append(f"{name}: n={values['n']}, groups={values['groups']}")
        for metric in ("V_MAE", "A_MAE", "mean_MAE"):
            lo, hi = values["paired_group_bootstrap_95ci"][metric]
            lines.append(f"  {metric}: {values['baseline'][metric]:.6f} -> {values['candidate'][metric]:.6f}; delta={values['delta_candidate_minus_baseline'][metric]:+.6f}; paired group 95% CI [{lo:+.6f}, {hi:+.6f}]")
    lines += ["", "初筛条件：整体 MAE 增量不超过 0.003，极高 A 的 A_MAE 降低，低 V 的 V_MAE 降低。",
              f"初筛结果：{'通过，仍需多种子确认' if all(checks.values()) else '未通过，不替换当前正式模型'}。",
              "该条件在本轮训练进行期间、完整三折结果产出前声明；不是原始完整预注册。",
              "单种子与小子集结果不能证明稳定提升。Bootstrap 只反映固定模型下的分组抽样变化；未涵盖重新训练的不确定性。",
              "追溯缺项：" + ("部分原始 config 的源文件哈希为空。已独立核对当前 Training JSON、冻结分折表和每条预测；原文件保留，不能追溯证明训练当时哈希。" if cw else "无新增缺项。")]
    (out / "复核结论.txt").write_text("\n".join(lines), encoding="utf-8-sig")
    print("\n".join(lines))
    print("Saved:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", type=Path, default=default_root)
    ap.add_argument("--project", type=Path, default=default_root.parent / "中文方面情感强度预测")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    if args.bootstrap < 100:
        ap.error("bootstrap must be at least 100")
    evaluate(args)
