---
title: DASC RULER 复现实验
description: RULER 准确率实验的最小复现、正式 sweep 与对照实验索引
---

# DASC RULER 复现实验

本目录区分“快速链路验证”和“论文正式统计”。2026-08-22 已按 N=30、一次 warm-up 加三次 replay 的正式协议精确复现 KDA RULER-S1 16k 的 dense 与 DASC-NR Wmax=16 数据点；两者均为 1.00。

## 文档索引

- [最小联合 A/B](minimal-joint-ab.md)：已实际复现，建议先跑。
- [论文数据点：KDA S1-16k](paper-kda-s1-16k.md)：N=30 正式协议，已精确复现。
- [4k/8k/16k 完整 sweep](small-context-suite.md)：论文 RULER 主表对应入口。
- [32k/64k/128k 长上下文 sweep](long-context-suite.md)：补充实验。
- [GDN CWE/MK3 定点复跑](gdn-cwe-mk3-rerun.md)：修复输出截断后的 GDN 对照。
- [INT8 与随机选择对照](int8-random-controls.md)：量化、随机掩码和 GDN sweep。

## 论文正式口径

| 项目 | 配置 |
| --- | --- |
| 硬件 | 2 × NVIDIA H20，TP=2 |
| KDA 模型 | Kimi-Linear-48B-A3B-Instruct |
| GDN 模型 | Qwen3-Next-80B-A3B-Instruct-NVFP4 |
| 数据 | RULER 全 13 个子任务，4k/8k/16k |
| 样本 | 每个 task-length-arm 使用相同的 30 个唯一样本 |
| 重放 | 1 次缓存填充，随后 3 次 measured replay |
| DASC 默认 | ragged、no-reconstruction、BF16 recurrent checkpoint |
| Wmax | 16、64、128、512、1024 |

正式统计只报告三个 replay round；warm-up 准确率和 token 计数不进入论文结果。三个 replay 不把独立样本数从 30 扩成 90。

## 公共入口

- 样本生成与评分：`datasets/ruler/ruler_gen.py`
- 生成器依赖数据：`datasets/ruler/data/`（词表、essay、SQuAD、HotpotQA）
- 正式 suite 客户端：`experiments/scripts/bench_idea1_accuracy.py`
- 联合准确率/TTFT 客户端：`experiments/scripts/bench_ruler_joint.py`
- 已复现论文目标与对照值：[KDA RULER-S1 16k](paper-kda-s1-16k.md)

运行前确认模型目录存在、两张 GPU 空闲，并使用未被占用的端口。多个 suite 脚本会按端口执行 `pkill`，共享机器上必须先检查端口归属。
