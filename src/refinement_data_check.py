"""Read-only train/dev input diagnostics for the fixed refinement experiment.

Writes two diagnostic artifacts; never loads test inputs or test references.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from main import D, O, inputs, train


def summarize_lengths(lengths):
    values = np.asarray(lengths, dtype=int)
    return {
        'p50': float(np.quantile(values, .50)),
        'p90': float(np.quantile(values, .90)),
        'p95': float(np.quantile(values, .95)),
        'p99': float(np.quantile(values, .99)),
        'maximum': int(values.max()),
        'over_80': int((values > 80).sum()),
        'over_128': int((values > 128).sum()),
    }


def mark(text, aspect):
    if not aspect or aspect == 'NULL':
        return text
    return text.replace(aspect, '[unused1]' + aspect + '[unused2]')


def main():
    root = Path(__file__).resolve().parents[1]
    out = O / 'refinement'
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(root / 'demo_tokenizer', local_files_only=True)
    marker_tokenizer = AutoTokenizer.from_pretrained(root / 'demo_tokenizer', local_files_only=True)
    vocab_size = len(marker_tokenizer)
    marker_tokenizer.add_special_tokens({'additional_special_tokens': ['[unused1]', '[unused2]']})
    assert len(marker_tokenizer) == vocab_size
    for marker in ('[unused1]', '[unused2]'):
        assert marker_tokenizer.encode(marker, add_special_tokens=False) == [marker_tokenizer.convert_tokens_to_ids(marker)]
    training, raw_count = train()
    frames = {'train': training, 'dev': inputs('Validation')}
    report = {
        'scope': 'training and development inputs only; no test data read',
        'length_definition': 'aspect-text pair token count including special tokens, no padding or truncation',
        'percentile_method': 'numpy.quantile default linear interpolation',
        'match_rule': 'literal case-sensitive Python str.count, non-overlapping occurrences',
        'marker_rule': 'mark all non-overlapping exact aspect occurrences; NULL/unmatched text unchanged',
        'raw_training_annotations': raw_count,
        'splits': {},
    }
    for split, rows in frames.items():
        encoded = tokenizer(rows.aspect.tolist(), rows.text.tolist(), truncation=False, padding=False)
        marked = marker_tokenizer(rows.aspect.tolist(), [mark(r.text, r.aspect) for r in rows.itertuples()],
                                  truncation=False, padding=False)
        null = rows.aspect.eq('NULL')
        occurrences = np.array([r.text.count(r.aspect) if r.aspect and r.aspect != 'NULL' else 0
                                for r in rows.itertuples()])
        unmatched = (~null.to_numpy()) & (occurrences == 0)
        report['splits'][split] = {
            'sentences': int(rows.id.nunique()),
            'aspect_instances': len(rows),
            'pair_lengths': summarize_lengths([len(ids) for ids in encoded['input_ids']]),
            'marked_pair_lengths': summarize_lengths([len(ids) for ids in marked['input_ids']]),
            'null_aspects': int(null.sum()),
            'unmatched_including_null': int((occurrences == 0).sum()),
            'unmatched_excluding_null': int(unmatched.sum()),
            'multiple_matches': int((occurrences > 1).sum()),
            'unmatched_non_null_rows': rows.loc[unmatched, ['id', 'text', 'aspect']].to_dict('records'),
        }
    report['training_label_bins'] = {
        'definition': '[1,2), [2,3), [3,4), [4,5), [5,6), [6,7), [7,8), [8,9]',
        'population': 'aggregated training aspect instances',
        'V': np.histogram(training.V, bins=np.arange(1, 10))[0].tolist(),
        'A': np.histogram(training.A, bins=np.arange(1, 10))[0].tolist(),
    }
    assert sum(report['training_label_bins']['V']) == len(training)
    assert sum(report['training_label_bins']['A']) == len(training)
    source_paths = sorted([*D.glob('*Training*.json'), *D.glob('*Validation*Simplified.txt')])
    report['input_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths}
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'data_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [
        '# 改进实验的数据诊断', '',
        '本诊断仅读取训练数据和开发集输入，未读取测试集。长度按“方面词＋评论”句对分词统计，包含特殊词元，不进行补齐或截断。分位数采用 NumPy 默认线性插值。', '',
        '## 原句对长度', '',
        '|划分|方面实例数|P50|P90|P95|P99|最大长度|超过80|超过128|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for split, info in report['splits'].items():
        length = info['pair_lengths']
        lines.append(f"|{split}|{info['aspect_instances']}|" + '|'.join(f'{length[k]:g}' for k in
                      ('p50', 'p90', 'p95', 'p99', 'maximum', 'over_80', 'over_128')) + '|')
    lines += ['', '训练和开发句对均未超过原先的 80 词元上限。因此，延长输入至 128 在这两个划分上没有增加原文信息，control_long 只作为负对照，不作为改善的原因。', '',
              '## 方面匹配与标记', '',
              '|划分|NULL方面|未命中含NULL|未命中不含NULL|多次命中|标记后最大长度|标记后超过128|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for split, info in report['splits'].items():
        lines.append(f"|{split}|{info['null_aspects']}|{info['unmatched_including_null']}|"
                     f"{info['unmatched_excluding_null']}|{info['multiple_matches']}|"
                     f"{info['marked_pair_lengths']['maximum']}|{info['marked_pair_lengths']['over_128']}|")
    lines += ['', '匹配采用完全一致的字符串匹配；多次命中按不重叠位置计数，正文中的所有命中均加标记。NULL 及未命中的方面保留原句。标记词注册为特殊词元，并检查每个标记编码为单一 ID，复用已有词表。']
    for split, info in report['splits'].items():
        for row in info['unmatched_non_null_rows']:
            lines.append(f"\n未命中的显式方面实例：{split} 的 {row['id']}，原文“{row['text']}”，方面“{row['aspect']}”。记录此情况并采用无标记回退，不用评分标签推测文字纠错。")
    lines += ['', '## 训练标签分布', '',
              '以合并后的方面实例为单位，每维按 1 分宽度分箱。最后一箱包含 9 分。', '',
              '|维度|1至2|2至3|3至4|4至5|5至6|6至7|7至8|8至9|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dim in ('V', 'A'):
        lines.append('|' + dim + '|' + '|'.join(map(str, report['training_label_bins'][dim])) + '|')
    lines += ['', 'V 标签主要集中于 6～7 分，A 标签主要集中于 5～7 分。稀有强度区间样本较少，频率重加权可作为固定候选，但会改变训练损失的侧重，不能预先保证整体 MAE 改善。权重只由训练标签计算，最终仍按开发集官方两维平均 MAE 选模。', '',
              '完整数值、定义及源文件 SHA-256 保存在 data_check.json。可在项目目录执行 python src/refinement_data_check.py 重现本诊断。']
    (out / '数据诊断.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report['splits'], ensure_ascii=False, indent=2))
    print('Saved train/dev diagnostics to', out)


if __name__ == '__main__':
    main()
