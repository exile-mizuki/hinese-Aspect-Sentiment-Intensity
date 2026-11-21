import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "中文方面情感强度预测" / "data"


def load_raw_annotations():
    groups = defaultdict(list)

    files = sorted(DATA.glob(
        "SIGHAN2024_dimABSA_TrainingSet*_Simplified.json"
    ))

    if len(files) != 2:
        raise RuntimeError(f"Expected 2 training JSON files, got {len(files)}")

    for path in files:
        entries = json.loads(path.read_text(encoding="utf-8-sig"))

        for entry in entries:
            arrays = [
                entry["Aspect"],
                entry["Opinion"],
                entry["OpinionFromTo"],
                entry["Intensity"]
            ]

            if len(set(map(len, arrays))) != 1:
                raise ValueError(f"Misaligned raw tuple: {entry['ID']}")

            key_prefix = (
                str(entry["ID"]),
                str(entry["Sentence"])
            )

            for aspect, opinion, position, intensity in zip(*arrays):
                v, a = map(float, str(intensity).split("#"))

                key = key_prefix + (str(aspect),)

                groups[key].append({
                    "V": v,
                    "A": a,
                    "opinion": str(opinion),
                    "position": str(position)
                })

    return groups


def marked_position_map(original, marked):
    markers = ("[unused1]", "[unused2]")
    positions = []
    i = 0

    while i < len(marked):
        marker = next(
            (m for m in markers if marked.startswith(m, i)),
            None
        )

        if marker is not None:
            i += len(marker)
            continue

        positions.append(i)
        i += 1

    recovered = "".join(marked[i] for i in positions)

    if recovered != original:
        raise ValueError("Marked text changed original characters")

    return positions


def opinion_span(text, opinion, position):
    match = re.fullmatch(r"\s*(\d+)\s*#\s*(\d+)\s*", position)

    if match is None:
        return None

    # Original coordinates: 1-based inclusive
    start = int(match.group(1)) - 1
    end = int(match.group(2))

    if not (0 <= start < end <= len(text)):
        return None

    if text[start:end] != opinion:
        return None

    return start, end


def build_opinion_labels(rows, tokenizer, original_tokens):
    from refine_experiment import marked_text

    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required for offset_mapping")

    raw_groups = load_raw_annotations()

    marked = [
        marked_text(r.text, r.aspect)
        for r in rows.itertuples()
    ]

    encoded = tokenizer(
        rows.aspect.tolist(),
        marked,
        truncation=True,
        max_length=128,
        padding=True,
        pad_to_multiple_of=8,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    for field in ("input_ids", "attention_mask", "token_type_ids"):
        if not torch.equal(encoded[field], original_tokens[field]):
            raise ValueError(f"Tokenizer mismatch: {field}")

    labels = torch.full_like(
        encoded["input_ids"], -100, dtype=torch.long
    )

    marker_ids = {
        tokenizer.convert_tokens_to_ids("[unused1]"),
        tokenizer.convert_tokens_to_ids("[unused2]"),
        tokenizer.sep_token_id
    }

    stats = Counter()
    worst_va_error = 0.0
    invalid_examples = []

    for i, row in enumerate(rows.itertuples()):
        stats["total"] += 1

        key = (str(row.id), str(row.text), str(row.aspect))
        annotations = raw_groups.get(key)

        if not annotations:
            stats["missing_raw_group"] += 1
            continue

        stats["matched_raw_group"] += 1

        if len(annotations) > 1:
            stats["merged_multi_opinion_rows"] += 1

        mean_v = sum(a["V"] for a in annotations) / len(annotations)
        mean_a = sum(a["A"] for a in annotations) / len(annotations)

        error = max(
            abs(mean_v - float(row.V)),
            abs(mean_a - float(row.A))
        )

        worst_va_error = max(worst_va_error, error)

        if error > 0.0002:
            stats["aggregate_va_mismatch"] += 1
            if len(invalid_examples) < 5:
                invalid_examples.append((row.id, "VA", error))
            continue

        stats["aggregate_va_verified"] += 1

        spans = []
        valid_record = True

        for ann in annotations:
            op = ann["opinion"].strip()

            if not op or op.upper() == "NULL":
                stats["null_opinions"] += 1
                continue

            span = opinion_span(
                str(row.text), ann["opinion"], ann["position"]
            )

            if span is None:
                stats["invalid_opinion_offsets"] += 1
                valid_record = False

                if len(invalid_examples) < 5:
                    invalid_examples.append((
                        row.id, "OFFSET", ann["position"]
                    ))
                break

            spans.append(span)
            stats["validated_opinion_spans"] += 1

        if not valid_record or not spans:
            stats["unsupervised_due_to_spans"] += 1
            continue

        positions = marked_position_map(
            str(row.text), marked[i]
        )

        mapped_spans = [
            (positions[s], positions[e - 1] + 1)
            for s, e in spans
        ]

        valid_tokens = []
        positive_tokens = set()

        offsets = encoded["offset_mapping"][i].tolist()
        types = encoded["token_type_ids"][i].tolist()
        attention = encoded["attention_mask"][i].tolist()
        ids = encoded["input_ids"][i].tolist()

        for j, ((left, right), segment, active, token_id) in enumerate(
            zip(offsets, types, attention, ids)
        ):
            if not (
                active
                and segment == 1
                and right > left
                and token_id not in marker_ids
            ):
                continue

            valid_tokens.append(j)

            if any(
                right > start and left < end
                for start, end in mapped_spans
            ):
                positive_tokens.add(j)

        if not positive_tokens:
            stats["opinion_truncated_or_unmapped"] += 1
            continue

        labels[i, valid_tokens] = 0
        labels[i, sorted(positive_tokens)] = 1

        stats["supervised_examples"] += 1
        stats["positive_tokens"] += len(positive_tokens)
        stats["negative_tokens"] += (
            len(valid_tokens) - len(positive_tokens)
        )

    result = dict(stats)
    result["max_aggregate_va_error"] = worst_va_error
    result["invalid_examples"] = invalid_examples

    return labels, result
