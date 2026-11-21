"""Reproduce 12 manually reviewed error cases from saved official-split predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "comparison"
OUT = ROOT / "results" / "case_analysis"
METHODS = ("improved", "bert_sentence", "bert_aspect")
METHOD_NAMES = {
    "improved": "岭回归目标分句",
    "bert_sentence": "BERT仅整句",
    "bert_aspect": "BERT方面句对",
}

# Each entry identifies a real source row. All numeric labels and predictions are read below.
CASES = [
    dict(id="R2103:S002", aspect="服务态度", group="预测改善", phenomenon="转折后评价对象变化",
         recommended=True,
         analysis="“服务态度不错”评价服务，“食物真的不好吃”评价食物。方面句对对服务的预测更接近人工分数，说明这条输入中显式方面有助于区分转折前后的对象。"),
    dict(id="R3423:S013", aspect="面", group="预测改善", phenomenon="多方面正负评价并存",
         recommended=False,
         analysis="前半句“小菜种类有点少”偏负面，后半句“光吃面就让人大大满足”明确赞赏面。方面句对更贴近面的正面评分，唤醒度仍低于人工评分。"),
    dict(id="R0907:S024", aspect="面条", group="预测改善", phenomenon="局部优点与整体偏好并存",
         recommended=False,
         analysis="“面条有咬劲”描述面条的局部优点，结尾“整体滋味非我所好”表达整体偏好。方面句对对面条的两维评分均更接近人工值，能够在这一例中体现对象差异。"),
    dict(id="R0921:S004", aspect="蛋饼", group="预测改善", phenomenon="同一方面包含相反意见",
         recommended=False,
         analysis="同一蛋饼既被评价“非常的油”“满嘴油二只手也油”，又被评价“又香又嫩”。人工愉悦度整体偏低。方面句对降低了前两种方法的偏正面预测，但仍高于人工值。"),
    dict(id="R1358:S031", aspect="豆花", group="仍有明显误差", phenomenon="比较表达的方向含糊",
         recommended=False,
         analysis="“完全不在同一个档次”只写出比较差异，单句没有明确说明档次更高还是更低。人工评分偏正面，两个BERT预测却偏负面。该例提示单句比较表达可能需要上下文来确定方向。"),
    dict(id="R3453:S040", aspect="握寿司", group="仍有明显误差", phenomenon="隐含的新鲜度不满",
         recommended=False,
         analysis="“看起来也放很久”通过放置时间表达对握寿司状态的担忧，句中缺少直接的“不好吃”。三个方法的愉悦度都高于人工评分，说明这类隐含负面评价仍难判断。"),
    dict(id="R0060:S023", aspect="味道", group="仍有明显误差", phenomenon="强烈负面情绪的强度低估",
         recommended=True,
         analysis="“超级不推荐”“超级淡”重复强调不满。人工愉悦度很低且唤醒度很高，方面句对虽给出偏负面的V，仍低估了负面程度和情绪激烈程度。高A也可以伴随负面评价。"),
    dict(id="R0832:S025", aspect="用油量", group="仍有明显误差", phenomenon="数量描述的评价含义",
         recommended=False,
         analysis="“用油量高，盒底不少残油”以用量和残留描述餐品。人工愉悦度偏低，方面句对却预测到中性以上。此处需结合餐饮语境理解“高”“不少残油”的评价含义。"),
    dict(id="R3420:S008", aspect="蚌面", peer="口味", group="同句方面混淆或打平",
         phenomenon="局部与整体评价层级",
         recommended=False,
         analysis="“口味还行”与“整碗都是油”分别提供局部和整体评价。人工给蚌面的V低于口味，方面句对的V顺序相反。仅整句模型将两方面打平，分句法也没有恢复人工顺序。"),
    dict(id="R0841:S027", aspect="猪肝", peer="猪肝口感", group="同句方面混淆或打平",
         phenomenon="方面名称包含关系",
         recommended=False,
         analysis="“猪肝口感不差”肯定口感，“可惜厚薄不一”指出缺点。人工给猪肝口感的V高于猪肝整体。方面句对对两个名称仅给出很小差距，并将顺序反向，显示重叠方面名仍会混淆。"),
    dict(id="R2120:S005", aspect="土鸡", peer="酱汁", group="同句方面混淆或打平",
         phenomenon="长句中多个对象的意见分配",
         recommended=True,
         analysis="“土鸡超小盘”与“酱汁蛮好吃”评价不同对象，句中还插入对糖醋鱼口感的批评。人工给土鸡的V低于酱汁，方面句对却将土鸡略排在酱汁前。此例中分句岭回归保留了正确顺序。"),
    dict(id="R0902:S030", aspect="味道", peer="咸蛋黄", group="同句方面混淆或打平",
         phenomenon="相邻短语中的方面边界",
         recommended=False,
         analysis="“咸蛋黄味道很好”将食材名和味道紧密相连。人工给味道的V高于咸蛋黄，方面句对在两位小数下给两者相同V。显式提供不同方面词后，正式提交的分数仍可能打平。"),
]


def get_row(frames, split, sentence_id, aspect):
    frame = frames[split]
    found = frame[(frame.id == sentence_id) & (frame.aspect == aspect)]
    if len(found) != 1:
        raise ValueError(f"Expected one source row: {split} {sentence_id} {aspect}, got {len(found)}")
    return found.iloc[0]


def scores(row, prefix=""):
    result = {prefix + "人工V": float(row.true_V), prefix + "人工A": float(row.true_A)}
    for method in METHODS:
        label = prefix + METHOD_NAMES[method]
        errors = []
        for dim in ("V", "A"):
            predicted = float(row[f"{method}_{dim}"])
            error = abs(predicted - float(row[f"true_{dim}"]))
            result[f"{label}_{dim}"] = predicted
            result[f"{label}_{dim}绝对误差"] = round(error, 6)
            errors.append(error)
        result[f"{label}_两维平均绝对误差"] = round(float(np.mean(errors)), 6)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {
        split: pd.read_csv(SOURCE / f"{split}_cases.csv", keep_default_na=False)
        for split in ("dev", "test")
    }
    records, pair_records = [], []
    for number, entry in enumerate(CASES, 1):
        split = "test"
        row = get_row(frames, split, entry["id"], entry["aspect"])
        numeric = scores(row)
        record = {
            "案例号": f"C{number:02d}", "分组": entry["group"], "来源划分": split,
            "句子ID": entry["id"], "原文": row.text, "方面词": entry["aspect"],
            "现象": entry["phenomenon"], "推荐展示": "是" if entry["recommended"] else "否",
            **numeric, "原因分析": entry["analysis"],
        }
        if entry["group"] == "预测改善":
            assert numeric["BERT方面句对_两维平均绝对误差"] < min(
                numeric["岭回归目标分句_两维平均绝对误差"],
                numeric["BERT仅整句_两维平均绝对误差"])
        if entry["group"] == "仍有明显误差":
            assert numeric["BERT方面句对_两维平均绝对误差"] >= 1
        if "peer" in entry:
            peer = get_row(frames, split, entry["id"], entry["peer"])
            assert row.text == peer.text
            true_gap = float(row.true_V - peer.true_V)
            predicted_gap = float(row.bert_aspect_V - peer.bert_aspect_V)
            assert abs(true_gap) >= .5 and true_gap * predicted_gap <= 0
            pair = {
                "案例号": record["案例号"], "来源划分": split, "句子ID": entry["id"],
                "原文": row.text, "主方面": entry["aspect"], "对照方面": entry["peer"],
                "主方面人工V": float(row.true_V), "对照方面人工V": float(peer.true_V),
                "人工V差值": round(true_gap, 6),
                **scores(peer, "对照方面_"),
            }
            for method in METHODS:
                gap = float(row[f"{method}_V"] - peer[f"{method}_V"])
                label = METHOD_NAMES[method]
                pair[f"{label}_主方面V"] = float(row[f"{method}_V"])
                pair[f"{label}_对照方面V"] = float(peer[f"{method}_V"])
                pair[f"{label}_V差值"] = round(gap, 6)
                pair[f"{label}_排序"] = "打平" if gap == 0 else "正确" if gap * true_gap > 0 else "反向"
            pair_records.append(pair)
            record["对照方面词"] = entry["peer"]
            record["BERT方面句对_主减对照V"] = round(predicted_gap, 6)
            record["人工_主减对照V"] = round(true_gap, 6)
        else:
            record["对照方面词"] = ""
            record["BERT方面句对_主减对照V"] = None
            record["人工_主减对照V"] = None
        records.append(record)
    frame = pd.DataFrame(records)
    assert len(frame) == 12 and frame.句子ID.nunique() == 12
    assert frame.分组.value_counts().to_dict() == {
        "预测改善": 4, "仍有明显误差": 4, "同句方面混淆或打平": 4}
    frame.to_csv(OUT / "12个案例.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pair_records).to_csv(OUT / "同句方面对照.csv", index=False, encoding="utf-8-sig")
    (OUT / "cases.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    metadata = {
        "selection": "purposive review of frozen seed-42 test predictions with AI-assisted text interpretation",
        "count": 12, "unique_sentences": 12,
        "groups": {"improved": 4, "large_error": 4, "aspect_order_error_or_tie": 4},
        "improvement_rule": "BERT aspect mean absolute V/A error lower than both comparison methods",
        "large_error_rule": "BERT aspect mean absolute V/A error at least 1.0",
        "pair_rule": "absolute true V difference >= 0.5; BERT aspect V order reversed or tied",
        "error_formula": "abs(prediction - gold); mean absolute error=(V_error+A_error)/2",
        "model_seed": 42,
        "notes": "Purposeful explanatory examples. Group counts do not estimate population prevalence.",
        "sources_sha256": {f"{s}_cases.csv": hashlib.sha256((SOURCE / f"{s}_cases.csv").read_bytes()).hexdigest()
                          for s in frames},
    }
    (OUT / "selection_log.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    chosen = frame[frame.推荐展示 == "是"]
    intro = (
        "# 典型案例分析\n\n"
        "本次从已冻结的随机种子 42 测试集预测中，定向整理 12 个句子—方面实例，来自 12 条不同评论。"
        "分为预测改善、仍有明显误差、同句方面混淆或打平三组，每组 4 例。"
        "先依据误差和同句排序筛查，再逐条阅读原文确定现象，所有标签和预测由脚本按 ID 与方面词从源表读取。"
        "分组数量由案例设计确定，总体表现应结合完整测试集指标判断。\n\n"
        "改善组的句对 BERT 两维平均绝对误差，均低于分句岭回归和仅整句 BERT。"
        "例如“服务态度不错”与“食物真的不好吃”在同一句并存，提供服务这一方面后，预测更贴近人工评价。"
        "面条局部口感与整体滋味偏好并存的例子也显示，整体情绪和指定方面的评分可以不同。\n\n"
        "明显误差组的句对 BERT 两维平均绝对误差均达到 1 分。"
        "“不在同一个档次”的比较方向依赖上下文，“握寿司看起来也放很久”通过状态暗示不满；"
        "“用油量高、盒底不少残油”需要餐饮语境中的评价知识。"
        "“超级不推荐、味道超级淡”则体现强烈负面情绪，模型仍同时低估了负面程度和唤醒度。"
        "上述解释依据文字与输出差异，属于错误原因分析。\n\n"
        "同句组另外保留了对照方面的评分。四对人工 V 差距均至少 0.5 分，句对 BERT 有三对顺序反向、一对打平。"
        "重叠名称“猪肝”与“猪肝口感”、整体“蚌面”与局部“口味”，以及土鸡、糖醋鱼、酱汁同时出现的长句，"
        "都体现了对象和意见分配的难点。不同方面均已作为输入提供，分数仍可能接近甚至次序相反。"
        "因此可在后续开发集实验中继续检查方面边界、整体与局部层级和长句对象切换。\n\n"
        "推荐展示以下三例，分别说明效果、强度误差和方面混淆。更多数值与逐例解释见《12个案例.csv》，"
        "四组同句比较的完整数据见《同句方面对照.csv》。\n\n"
    )
    lines = [intro]
    for r in chosen.to_dict("records"):
        lines.append(
            f"## {r['案例号']} · {r['现象']}\n\n"
            f"原文“{r['原文']}”，方面“{r['方面词']}”，测试集 {r['句子ID']}。\n\n"
            f"人工 V/A 为 {r['人工V']:.2f}/{r['人工A']:.2f}；"
            f"分句岭回归为 {r['岭回归目标分句_V']:.2f}/{r['岭回归目标分句_A']:.2f}，"
            f"仅整句 BERT 为 {r['BERT仅整句_V']:.2f}/{r['BERT仅整句_A']:.2f}，"
            f"句对 BERT 为 {r['BERT方面句对_V']:.2f}/{r['BERT方面句对_A']:.2f}。\n\n"
            f"{r['原因分析']}\n\n"
        )
    (OUT / "案例分析.md").write_text("".join(lines), encoding="utf-8")
    print(frame[["案例号", "分组", "句子ID", "方面词", "BERT方面句对_两维平均绝对误差"]].to_string(index=False))
    print("Created 12 cases and 4 within-sentence comparisons.")


if __name__ == "__main__":
    main()
