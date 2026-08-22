---
title: RULER INT8 与随机选择对照
description: KDA/GDN 状态量化、随机掩码和 GDN 修正 sweep 的实验说明
---

# RULER INT8 与随机选择对照

对应 `experiments/scripts/run_ruler_int8_random_gdn.sh`。默认一次运行包含五个 phase，规模很大，应按 phase 分开执行。

## Phase

| Phase | 默认规模 | 关键设置 |
| --- | ---: | --- |
| KDA INT8 | 1 launch | Wmax=0，STATE_QUANT_MODE=int8 |
| KDA random | 25 launches | 5 Wmax × 5 seeds，count-matched random drop |
| GDN INT8 | 1 launch | Wmax=0，STATE_QUANT_MODE=int8 |
| GDN random | 25 launches | 5 Wmax × 5 seeds |
| GDN corrected sweep | 6 launches | dense + 5 Wmax，max-new-tokens=256 |

每次 launch 默认覆盖 13 任务 × 3 长度 × 30 样本 × 4 tries，即 4,680 个请求；全部默认 phase 共 58 次服务启动。

## 建议分阶段命令

KDA INT8：

```bash
RUN_KDA_INT8=1 RUN_KDA_RANDOM=0 \
RUN_GDN_INT8=0 RUN_GDN_RANDOM=0 RUN_GDN=0 \
bash experiments/scripts/run_ruler_int8_random_gdn.sh
```

KDA 论文单 seed 随机对照：

```bash
RUN_KDA_INT8=0 RUN_KDA_RANDOM=1 \
RUN_GDN_INT8=0 RUN_GDN_RANDOM=0 RUN_GDN=0 \
SEEDS=1234 bash experiments/scripts/run_ruler_int8_random_gdn.sh
```

GDN 修正 sweep：

```bash
RUN_KDA_INT8=0 RUN_KDA_RANDOM=0 \
RUN_GDN_INT8=0 RUN_GDN_RANDOM=0 RUN_GDN=1 \
bash experiments/scripts/run_ruler_int8_random_gdn.sh
```

## 公共配置与输出

默认任务为全部 13 项，长度 4k/8k/16k，N=30，4 tries，max-new-tokens=256，checkpoint slots=200。KDA parallel=8/mem fraction=0.85；GDN parallel=4/max running=4/mem fraction=0.90。

KDA 写入 `idea1_accuracy_logs/ruler_int8_random/`，GDN 写入 `idea1_accuracy_logs/ruler_gdn_rerun_256/`。

## 口径与陷阱

- 脚本默认随机 seeds 是 42、123、7、555、31337；论文附录的 RULER random-drop 表使用 seed=1234。复现该表必须显式设置 `SEEDS=1234`。
- Wmax=0 的 INT8 arm 是全局 state checkpoint 量化基线，不等同于某个 DASC Wmax arm。
- 所有 phase 共用固定结果目录和 arm 名 resume；变更 N、tries 或 seeds 后可能错误跳过旧记录。
- 脚本硬编码模型路径与结果目录，运行前先核对。
- random 对照应与 DASC 使用同一批实例，并按实例做 paired comparison；三个 replay round 不能当作 90 个独立样本。
