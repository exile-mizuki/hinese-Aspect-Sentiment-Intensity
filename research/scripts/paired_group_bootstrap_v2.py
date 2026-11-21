#!/usr/bin/env python3
"""Paired connected-review-group bootstrap for local, train-only OOF predictions.

This script reads ONLY validation/proposed_3fold_manifest.csv and
training_v2/<method>/seed_<seed>/fold_<fold>/heldout_predictions.csv.
It NEVER reads official Dev/Test data, and never trains/changes models.

Positive gain means the candidate reduces MAE. CIs are conditional on the
observed seeds/folds and are exploratory, not a proof of external generalization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ('marked_l1', 'a_mse', 'aux_0.05')
PAIRS = (('marked_l1', 'a_mse'), ('a_mse', 'aux_0.05'), ('marked_l1', 'aux_0.05'))
METRICS = ('V_MAE', 'A_MAE', 'mean_MAE', 'high_A_MAE')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_manifest(path):
    require(path.is_file(), f'Cannot find grouped CV manifest: {path}')
    m = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                    dtype={'id': str, 'aspect': str, 'text': str})
    needed = {'row_index', 'id', 'aspect', 'text', 'V', 'A', 'group', 'fold'}
    require(needed.issubset(m.columns), f'Missing manifest fields: {needed - set(m.columns)}')
    require(not m.duplicated(['id', 'aspect']).any(), 'Duplicate id+aspect in manifest; cannot align predictions')
    m = m.sort_values('row_index').reset_index(drop=True)
    require(np.array_equal(m.row_index.to_numpy(), np.arange(len(m))), 'Manifest row_index is not continuous')
    require(m.groupby('group').fold.nunique().max() == 1, 'Group leakage between folds')
    require(np.isfinite(m[['V', 'A']].to_numpy(dtype=float)).all(), 'Bad labels in manifest')
    return m


def load_predictions(root, manifest, method, seed):
    frames = []
    for fold in sorted(manifest.fold.unique()):
        path = root / 'training_v2' / method / f'seed_{seed}' / f'fold_{int(fold)}' / 'heldout_predictions.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing CV prediction file: {path}')
        frame = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                            dtype={'id': str, 'aspect': str, 'text': str})
        needed = {'id', 'aspect', 'text', 'V', 'A', 'fold', 'group', 'pred_V', 'pred_A'}
        require(needed.issubset(frame.columns), f'{path}: missing fields {needed - set(frame.columns)}')
        require((frame.fold.astype(int) == int(fold)).all(), f'{path}: wrong fold assignments')
        frames.append(frame[list(needed)].copy())
    p = pd.concat(frames, ignore_index=True)
    require(len(p) == len(manifest), f'{method}, seed {seed}: expected {len(manifest)}, got {len(p)} rows')
    require(not p.duplicated(['id', 'aspect']).any(), f'{method}, seed {seed}: repeated id+aspect')
    joined = manifest.merge(p, on=['id', 'aspect'], how='left', sort=False,
                            suffixes=('_true', '_pred'), validate='one_to_one', indicator=True)
    require((joined._merge == 'both').all(), f'{method}, seed {seed}: mismatched prediction keys')
    for c in ['text', 'fold', 'group']:
        require((joined[f'{c}_true'] == joined[f'{c}_pred']).all(),
                f'{method}, seed {seed}: mismatched {c}; source rows are not aligned')
    for c in ['V', 'A']:
        require(np.allclose(joined[f'{c}_true'].astype(float), joined[f'{c}_pred'].astype(float),
                            atol=1e-6, rtol=1e-6),
                f'{method}, seed {seed}: mismatched ground-truth labels')
    pred = joined[['pred_V', 'pred_A']].to_numpy(dtype=float)
    require(np.isfinite(pred).all(), f'{method}, seed {seed}: nonfinite predictions')
    return np.round(np.clip(pred, 1, 9), 2)


def calc_errors(y, pred):
    return np.abs(y - pred)


def metric_values(err, high):
    return {'V_MAE': float(err[:, 0].mean()),
            'A_MAE': float(err[:, 1].mean()),
            'mean_MAE': float(err.mean()),
            'high_A_MAE': float(err[high, 1].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('optimization_v2'),
                    help='Path to optimization_v2 folder')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--resamples', type=int, default=5000)
    ap.add_argument('--bootstrap-seed', type=int, default=20261120)
    args = ap.parse_args()
    require(args.resamples >= 100, '--resamples must be at least 100')
    require(len(set(args.seeds)) == len(args.seeds), 'Repeated seed values')
    root = args.root.resolve()
    manifest = read_manifest(root / 'validation' / 'proposed_3fold_manifest.csv')
    y = manifest[['V', 'A']].to_numpy(dtype=float)
    high = y[:, 1] >= 6.5
    require(high.any(), 'No high-A examples')
    ids, inverse = np.unique(manifest['group'].to_numpy(), return_inverse=True)
    ng = len(ids)
    predictions, errors, summaries = {}, {}, []
    for method in METHODS:
        for seed in args.seeds:
            pred = load_predictions(root, manifest, method, seed)
            predictions[(method, seed)] = pred
            err = calc_errors(y, pred)
            errors[(method, seed)] = err
            summaries.append({'method': method, 'seed': seed, **metric_values(err, high)})
    summary = pd.DataFrame(summaries)
    print('\n=== 逐种子折外验证指标（越低越好） ===')
    print(summary.round(6).to_string(index=False))

    out_dir = root / 'validation' / 'posttraining_bootstrap'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'recomputed_seed_metrics.csv', index=False, encoding='utf-8-sig')
    rng = np.random.default_rng(args.bootstrap_seed)
    draw = rng.integers(0, ng, size=(args.resamples, ng))
    n_group = np.bincount(inverse, minlength=ng)
    high_n_group = np.bincount(inverse, weights=high.astype(float), minlength=ng)
    rows = []
    per_seed = []

    for baseline, candidate in PAIRS:
        delta = np.mean([errors[(baseline, seed)] - errors[(candidate, seed)]
                         for seed in args.seeds], axis=0)
        for seed in args.seeds:
            single = errors[(baseline, seed)] - errors[(candidate, seed)]
            per_seed.append({'baseline': baseline, 'candidate': candidate, 'seed': seed,
                             **{f'gain_{metric}': val for metric, val in metric_values(single, high).items()}})
        weights = {
            'V_MAE': delta[:, 0],
            'A_MAE': delta[:, 1],
            'mean_MAE': delta.mean(axis=1),
            'high_A_MAE': delta[:, 1] * high,
        }
        for metric in METRICS:
            sums = np.bincount(inverse, weights=weights[metric], minlength=ng)
            counts = high_n_group if metric == 'high_A_MAE' else n_group
            observed = float(sums.sum() / counts.sum())
            denominator = counts[draw].sum(axis=1)
            require(np.all(denominator > 0), 'Encountered empty bootstrap denominator')
            sampled_gain = sums[draw].sum(axis=1) / denominator
            low, high_ci = np.quantile(sampled_gain, [0.025, 0.975])
            rows.append({'baseline': baseline, 'candidate': candidate, 'metric': metric,
                         'MAE_reduction_positive_is_better': observed,
                         'CI95_low': float(low), 'CI95_high': float(high_ci),
                         'CI_excludes_zero': bool(low > 0 or high_ci < 0),
                         'connected_groups': ng, 'seeds': ','.join(map(str, args.seeds)),
                         'bootstrap_resamples': args.resamples})
        if baseline == 'marked_l1' and candidate == 'a_mse':
            regressions = manifest.loc[high, ['id', 'text', 'aspect', 'A', 'group', 'fold']].copy()
            regressions['avg_baseline_abs_error_A'] = np.mean([
                errors[(baseline, seed)][high, 1] for seed in args.seeds], axis=0)
            regressions['avg_candidate_abs_error_A'] = np.mean([
                errors[(candidate, seed)][high, 1] for seed in args.seeds], axis=0)
            regressions['candidate_minus_baseline_error'] = (
                regressions['avg_candidate_abs_error_A'] - regressions['avg_baseline_abs_error_A'])
            regressions.sort_values('candidate_minus_baseline_error', ascending=False).head(30).to_csv(
                out_dir / 'a_mse_high_A_top30_regressions.csv', index=False, encoding='utf-8-sig')
    result = pd.DataFrame(rows)
    print('\n=== 按关联评论组重采样的配对 Bootstrap：正数表示候选降低了 MAE ===')
    print(result[['baseline', 'candidate', 'metric', 'MAE_reduction_positive_is_better',
                  'CI95_low', 'CI95_high', 'CI_excludes_zero']].round(6).to_string(index=False))
    result.to_csv(out_dir / 'paired_cluster_bootstrap.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(per_seed).to_csv(out_dir / 'per_seed_paired_gains.csv', index=False, encoding='utf-8-sig')
    (out_dir / 'interpretation_limits.json').write_text(json.dumps({
        'source': 'TRAIN-only grouped OOF prediction CSVs; official Dev/Test untouched',
        'bootstrap_unit': 'manifest.group (connected component of reviews and exact texts)',
        'resamples': args.resamples, 'bootstrap_random_seed': args.bootstrap_seed,
        'seeds_included': args.seeds, 'instances': len(manifest),
        'high_A_definition': 'A >= 6.5', 'group_count': ng,
        'scope_warning': 'Intervals reflect grouped data resampling conditional on 3 saved training seeds; they do NOT capture unseen training seeds, fold design uncertainty, hyperparameter-selection bias, or new data distribution. Grouped OOF CV folds share training records.',
        'adoption_warning': 'A candidate must still pass the precommitted independent development-set gate, without tuning to Test.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n完成。结果目录：{out_dir}')
    print('未使用 Dev/Test；未重训；未修改原始预测。')


if __name__ == '__main__':
    main()
