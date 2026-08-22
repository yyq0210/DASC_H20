# DASC H20 experiments

This repository contains only the H20 experiment assets: datasets/fixed workloads,
scripts, documentation, and reproduced results. SGLang source code and history are
intentionally not vendored here.

The repository has three content directories:

- `datasets/`: fixed serving workloads plus the RULER generator and its required corpora.
- `experiments/`: runnable scripts and checked-in raw reproduction results.
- `docs/`: one configuration/reproduction guide per experiment family.

## Runtime dependency

The experiments use the separate SGLang repository:

- Source: <https://github.com/yyq0210/sglang>
- Branch: `yyq/kimi-k3-dasc`
- Verified commit: `f05a0725a7e85cabc0d0149c9ebe16d6218e933b`

Install that checkout in editable mode before running scripts from this repository:

```bash
git clone --branch yyq/kimi-k3-dasc https://github.com/yyq0210/sglang.git /path/to/sglang
export SGLANG_REPO=/path/to/sglang
python -m pip install -e "$SGLANG_REPO/python"
cd /path/to/DASC_H20
```

For the serving experiments, copy the checked-in fixed workloads to the paths expected by
the drivers, or override the workload option documented for each experiment:

```bash
cp datasets/serving/arena_shared_prefix_N40.jsonl /tmp/arena_shared_prefix_N40.jsonl
cp datasets/serving/arena_shared_prefix_N200.jsonl /tmp/arena_shared_prefix_N200.jsonl
```

## Start here

- Performance experiments: [`docs/performance/README.md`](docs/performance/README.md)
- RULER experiments: [`docs/ruler/README.md`](docs/ruler/README.md)
- Formal reproduced paper point: [`docs/ruler/paper-kda-s1-16k.md`](docs/ruler/paper-kda-s1-16k.md)

## Included results

| Result directory | Status |
| --- | --- |
| `experiments/results/fixed_graph_dense_3seed_20260822` | KDA matched-HBM dense, S=96, 3 seeds |
| `experiments/results/fixed_graph_padded_3seed_20260822` | KDA matched-HBM padded DASC, S=96, 3 seeds |
| `experiments/results/diag_prefill_graph_3seed_20260822` | KDA matched-HBM ragged DASC, S=96, 3 seeds |
| `experiments/results/minperf_kda_s96_20260822` | Initial diagnostic run; capacity valid, prefill graph not locked |
| `experiments/results/ruler_joint_minimal_fixed_20260822` | RULER 16k smoke test after SSE compatibility fix |
| `experiments/results/ruler_paper_kda_s1_16k_20260822` | Formal N=30 paper-point reproduction |

## Fixed inputs

- `datasets/serving/arena_shared_prefix_N40.jsonl`: matched-HBM serving input.
- `datasets/serving/arena_shared_prefix_N200.jsonl`: Wmax sweep input.
- Adjacent `*.stats.json` files record construction statistics.
- `datasets/ruler/data/`: the minimal vocabularies and corpora required by all 13 RULER tasks.

Model weights are intentionally excluded. Scripts look under `SGLANG_REPO` by default;
override `MODEL` when weights are stored elsewhere.

## Reproduced headline values

- KDA matched-HBM S=96, ragged versus dense: hit rate +66.28%, TTFT -66.02%, throughput +42.55%.
- KDA RULER-S1 16k: paper dense/DASC-NR Wmax=16 = 1.00/1.00; reproduced replay = 1.0000/1.0000.

All result logs are raw experiment artifacts. The documentation states which directories are
formal results and which are diagnostic runs.
