---
title: RULER 最小联合 A/B
description: KDA 16k dense 与 DASC-NR Wmax=16 的快速准确率和缓存链路验证
---

# RULER 最小联合 A/B

## 目的

用最少的服务启动验证四件事：RULER 样本能生成、模型输出能评分、重复请求能命中缓存、DASC-NR 在相同输入上与 dense 行为一致。它是 smoke test，不替代论文全量表。

对应脚本：

- 驱动：`experiments/scripts/run_ruler_joint_ab.sh`
- 客户端：`experiments/scripts/bench_ruler_joint.py`
- 数据与评分：`datasets/ruler/ruler_gen.py`

## 本次配置

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-08-22 |
| 模型 | Kimi-Linear-48B-A3B-Instruct |
| GPU / 并行 | 2 × H20 / TP=2 |
| 上下文 | 16k |
| 任务 | s1、mk1、vt、cwe |
| arms | dense、NR_16 |
| 每任务样本 | 2，base seed=42 |
| 每样本请求 | 2：第一次 miss，第二次 hit |
| checkpoint slots | 64，大于 8 个唯一样本 |
| DASC | Wmax=16、ragged=1、re-prefill=0、route A |
| 采样 | temperature=0 |
| 总请求 | 每臂 16 |

## 命令

```bash
cd /path/to/DASC_H20
ARMS="dense NR_16" \
  N=2 NUMTRIES=2 CKPT_SIZE=64 PORT=30017 \
  LOGDIR="$PWD/experiments/results/ruler_joint_minimal_fixed_20260822" \
  bash experiments/scripts/run_ruler_joint_ab.sh
```

只检查配置可运行：

```bash
bash experiments/scripts/run_ruler_joint_ab.sh --dry-run
```

## 实测结果

| arm | overall accuracy | miss TTFT | hit TTFT | request hit rate | 请求数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8625 | 897.1 ms | 75.6 ms | 0.500 | 16 |
| NR_16 | 0.8625 | 936.8 ms | 80.3 ms | 0.500 | 16 |

逐任务准确率两臂完全相同：S1=1.00、MK1=1.00、CWE=1.00、VT=0.45。VT 仅有两个样本，其中一个模型回答失败，因此总体值明显低于论文大样本结果；不能据此判断精度回退。两臂输入字符总数均为 1,132,604，交叉检查通过。

结果文件：

- `experiments/results/ruler_joint_minimal_fixed_20260822/result_dense_joint.json`
- `experiments/results/ruler_joint_minimal_fixed_20260822/result_NR_16_joint.json`
- 同目录下 `server_*.log` 和 `probe_*.log` 保存服务与客户端日志。

## 当前脚本兼容修复

当前 `/generate` SSE 返回累计字段 `text`，旧联合客户端只读取 `text_output`，会得到空文本并把所有准确率记为 0。客户端现优先读取 `text`，同时保留旧字段兼容。目录 `experiments/results/ruler_joint_minimal_20260822` 是修复前诊断结果，不应作为实验数据。

## 验收

- JSON 中 `output_text` 非空。
- 两臂的 prompt 总量相同。
- 第二次请求 `cached_tokens > 0`，请求命中率为 0.5。
- dense 与 NR_16 的逐样本/逐任务分数一致。
- 服务退出后没有遗留 launch_server 进程。

联合脚本把 warm-up 和 hit 都计入 overall accuracy，并且只有 4 个任务；论文正式口径是 13 个任务、N=30、丢弃 warm-up 后统计 3 个 replay round。小样本 TTFT 只用于确认链路，不用于复现论文性能表。
