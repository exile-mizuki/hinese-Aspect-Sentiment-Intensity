# 观感｜中文方面情感强度预测

SIGHAN 2024 DimABSA 课程设计：输入中文餐饮评论及目标方面，预测 V（情感效价）与 A（唤醒度）两个 1～9 连续分数。本仓库展示**已完成的代码、研究记录和网页**。

## 从哪里开始看

1. `web/` 与 `src/web_demo.py`：本地交互演示，支持多方面输入、V/A 可视化和结果导出。网页截图在 `research/web_screenshots/`。
2. `research/release/`：正式模型冻结记录、实验登记表。正式模型为 Chinese BERT WWM Ext `marked_l1` seed 42，训练 3 轮。
3. `research/experiments/`：各轮折外汇总指标。比较时请区分单种子、多种子、三折 Group CV 与官方开发集。
4. `research/docs/`、`research/project_results/`：数据审计、优化过程、报告图表。
5. `research/scripts/`：研究实验脚本；源码保留历史方法，当前演示不需要全部运行。

## 为什么没有数据和权重

正式模型约 409 MB；网页可选候选模型也约 409 MB。普通 GitHub 仓库单文件限制为 100 MB。官方训练数据、测试真值、第三方 EmoBank 数据和 40 条人工评分材料需要分别核查来源与使用范围，因此没有放进公开仓库。模型元数据与验证结果保留在 `research/`，成绩不是手工估计。

仅克隆此仓库**不能直接运行模型**。完整离线演示请使用课程交付包；如果有权公开分发模型，可将经许可核对的模型作为 GitHub Release 附件，并在这里补下载说明。不要把大 ZIP 提交到仓库。

## 使用说明

准备 Python 3.12 环境和 `requirements-bert.txt` 依赖。依原项目目录结构放置 `中文方面情感强度预测/data/`、`demo_tokenizer/`、`checkpoints/round2/candidate.pt` 以及 `optimization_v2/final_model/marked_l1/seed_42/model.pt`、配置和 `optimization_v2/release_v1/baseline_manifest.json`，再启动本地网页。

历史 Test 已被查看；后续模型比较是探索性结果。AI 建议分数不作为人工真值。官方数据来源：[SIGHAN2024-dimABSA](https://github.com/NYCU-NLP/SIGHAN2024-dimABSA)。
