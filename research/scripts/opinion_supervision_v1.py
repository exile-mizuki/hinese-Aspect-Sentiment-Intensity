import json
from pathlib import Path
from collections import defaultdict, Counter

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "中文方面情感强度预测" / "data"


def make_key(id_, text, aspect, v, a):
    return (
        str(id_), str(text), str(aspect),
        round(float(v), 4), round(float(a), 4)
    )


def load_raw_opinions():
    mapping = defaultdict(set)

    files = sorted(DATA.glob(
        "SIGHAN2024_dimABSA_TrainingSet*_Simplified.json"
    ))

    if len(files) != 2:
        raise RuntimeError(
            f"Expected 2 raw training JSON files, found {len(files)}"
        )

    for path in files:
        entries = json.loads(path.read_text(encoding="utf-8-sig"))

        for entry in entries:
            aspects = entry["Aspect"]
            opinions = entry["Opinion"]
            intensities = entry["Intensity"]

            if not (
                len(aspects) == len(opinions) == len(intensities)
            ):
                raise ValueError(
                    f"Misaligned annotations: {entry['ID']}"
                )

            for aspect, opinion, intensity in zip(
                aspects, opinions, intensities
            ):
                v, a = map(float, str(intensity).split("#"))
                key = make_key(
                    entry["ID"],
                    entry["Sentence"],
                    aspect, v, a
                )
                mapping[key].add(str(opinion))

    return mapping


def original_char_positions(original, marked):
    markers = ("[unused1]", "[unused2]")
    positions = []
    i = 0

    while i < len(marked):
        matched = next(
            (m for m in markers if marked.startswith(m, i)),
            None
        )
        if matched is not None:
            i += len(matched)
        else:
            positions.append(i)
            i += 1

    recovered = "".join(marked[i] for i in positions)
    if recovered != original:
        raise ValueError(
            "Marked text does not preserve the original sentence"
        )

    return positions


def build_opinion_labels(rows, tokenizer, original_tokens):
    from refine_experiment import marked_text

    if not tokenizer.is_fast:
        raise RuntimeError(
            "Opinion labels require a fast tokenizer with offsets"
        )

    gold = load_raw_opinions()

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
        if not torch.equal(
            encoded[field], original_tokens[field]
        ):
            raise ValueError(
                f"Tokenizer mismatch in {field}"
            )

    labels = torch.full_like(
        encoded["input_ids"], -100, dtype=torch.long
    )

    marker_ids = {
        tokenizer.convert_tokens_to_ids("[unused1]"),
        tokenizer.convert_tokens_to_ids("[unused2]"),
        tokenizer.sep_token_id
    }

    stats = Counter()

    for i, row in enumerate(rows.itertuples()):
        stats["total"] += 1

        key = make_key(
            row.id, row.text, row.aspect, row.V, row.A
        )

        candidates = gold.get(key, set())

        if not candidates:
            stats["no_exact_raw_match"] += 1
            continue

        stats["exact_raw_match"] += 1

        if len(candidates) != 1:
            stats["ambiguous_gold_opinion"] += 1
            continue

        opinion = next(iter(candidates))

        if not opinion or opinion.upper() == "NULL":
            stats["empty_opinion"] += 1
            continue

        occurrences = []
        start = 0

        while True:
            at = row.text.find(opinion, start)
            if at < 0:
                break
            occurrences.append(at)
            start = at + 1

        if len(occurrences) != 1:
            stats["ambiguous_or_missing_position"] += 1
            continue

        start = occurrences[0]
        end = start + len(opinion)

        position_map = original_char_positions(
            row.text, marked[i]
        )

        mapped_start = position_map[start]
        mapped_end = position_map[end - 1] + 1

        valid = []
        positive = []

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

            valid.append(j)

            if right > mapped_start and left < mapped_end:
                positive.append(j)

        if not positive:
            stats["opinion_truncated_or_unmapped"] += 1
            continue

        labels[i, valid] = 0
        labels[i, positive] = 1

        stats["supervised_examples"] += 1
        stats["positive_tokens"] += len(positive)
        stats["negative_tokens"] += len(valid) - len(positive)

    return labels, dict(stats)
