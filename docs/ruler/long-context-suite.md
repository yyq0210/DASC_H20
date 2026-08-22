---
title: RULER 32k/64k/128k 长上下文 Sweep
description: RULER 长上下文补充实验的配置、脚本和资源限制
---

# RULER 32k/64k/128k 长上下文 Sweep

对应 `experiments/scripts/run_ruler_suite_longctx.sh`。这是 4k/8k/16k 主实验的补充，不是论文主表的最小复现。

## 默认配置

13 个任务，长度 32k/64k/128k，Wmax=16/64/128/512/1024，4 tries；32k/64k 每项 N=30，128k 每项 N=20。为控制长上下文 KV 占用，client parallel=4、max-running-requests=4。KDA 使用 mem fraction 0.85，GDN 使用 0.90。

## 命令

```bash
cd /path/to/DASC_H20
BACKEND=kda \
  LOGDIR="$PWD/experiments/results/ruler_longctx_kda" \
  bash experiments/scripts/run_ruler_suite_longctx.sh

BACKEND=gdn \
  LOGDIR="$PWD/experiments/results/ruler_longctx_gdn" \
  bash experiments/scripts/run_ruler_suite_longctx.sh
```

与 small-context 脚本不同，这个脚本支持外部 LOGDIR，建议始终使用独立目录。结果为 `acc_results.jsonl`、`summary_longctx.txt`、`out_*.jsonl`、`bench_*.log` 和 `server_longctx_*.log`。

## 验收与风险

- 128k 使用 N=20，不能与 N=30 的区间直接等同。
- 检查服务日志无 OOM，且每个 arm 的问题数符合配置。
- 检查 warm-up 后三个 replay 的稳定性和 cached/new token 统计。
- 脚本同样按端口执行 `pkill`，并按 arm 名 resume-skip。
- 单次完整 sweep 很大；先用 [最小联合 A/B](minimal-joint-ab.md) 验证环境。
