#!/usr/bin/env python3
"""Aggregate + plot the EQUAL-HBM serving control sweep (run_equal_hbm.sh output).

Reads per-pass result JSONs  equal_hbm/result_*_seed*.json  written by
bench_arena_shared_prefix.sh (MEASURE_SEEDS path), groups by (backend, mode,
budget-S), and reports mean +/- std over the 3 seeds for the three headline
metrics:  _token_hit_rate, mean_ttft_ms, output_throughput.

Primary output is a PLAIN-TEXT table (paper-citable). If matplotlib is available
it ALSO writes one figure (2 rows KDA/GDN x 3 cols hit-rate/TTFT/throughput,
x = HBM budget GiB, 3 lines dense/padded/ragged with std error bars).

Budget bookkeeping: the intended dense-equivalent slot count S is recovered from
each record as  round(_ckpt_slots * _bytes_per_slot / bytes_per_slot[dense]) so
records from different modes at the same budget group together. We also verify
measured (_bytes_per_slot * _ckpt_slots) ~= target budget and warn on drift,
which would mean the constant-bytes/slot assumption failed.

  Usage: python experiments/scripts/plot_equal_hbm.py [--logdir DIR] [--out FIG]
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

# measured constant bytes/slot per (backend,mode) -- used only to recover the
# intended budget-S bucket and to compute the dense-equivalent reference.
BPS = {
    ("kda", "dense"): 20971520,
    ("kda", "padded"): 11929600,
    ("kda", "ragged"): 7570944,
    ("kda", "int8"): 5324800,  # offline est; live-calibrated value injected per-record
    (
        "kda",
        "int8fair",
    ): 10649600,  # offline est; head-aware pool + STATE_QUANT_MODE=int8 (32 heads)
    ("gdn", "dense"): 37748736,
    ("gdn", "padded"): 37748736,
    ("gdn", "ragged"): 27983872,
}
MODES = ("dense", "padded", "ragged", "int8", "int8fair")
BACKENDS = ("kda", "gdn")
S_LIST = (48, 96, 144)
METRICS = [
    ("_token_hit_rate", "hit_rate", 1.0),
    ("mean_ttft_ms", "TTFT_ms", 1.0),
    ("output_throughput", "thr_tok_s", 1.0),
]


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0, len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var), len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logdir",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "gdn_prefix_hitrate_logs",
            "equal_hbm",
        ),
    )
    ap.add_argument(
        "--out", default=None, help="figure path (default: <logdir>/equal_hbm.pdf)"
    )
    args = ap.parse_args()
    logdir = os.path.abspath(args.logdir)
    out = args.out or os.path.join(logdir, "equal_hbm.pdf")

    files = sorted(glob.glob(os.path.join(logdir, "result_*_seed*.json")))
    if not files:
        print(f"[plot] no result_*_seed*.json under {logdir}")
        return

    # group[(backend,mode,S)] = {metric: [values...], "budget_gb": .., "slots": ..,
    #                            "in_tok": set(), "hbm_meas_gb": [..]}
    group = defaultdict(lambda: defaultdict(list))
    drift_warn = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"[plot] skip unreadable {os.path.basename(f)}: {e}")
            continue
        bk = d.get("_backend")
        md = d.get("_dasc_mode")
        slots = d.get("_ckpt_slots")
        bps = d.get("_bytes_per_slot") or BPS.get((bk, md))
        if bk not in BACKENDS or md not in MODES or not slots or not bps:
            continue
        bps_dense = BPS[(bk, "dense")]
        budget_bytes = slots * bps
        S_raw = round(budget_bytes / bps_dense)
        # snap to nearest standard budget point so measured-bytes/slot drift
        # (e.g. 47/95/105) doesn't split same-budget records into separate groups.
        S = min(S_LIST, key=lambda s: abs(s - S_raw)) if S_LIST else S_raw
        key = (bk, md, S)
        g = group[key]
        for jkey, _, _ in METRICS:
            g[jkey].append(d.get(jkey))
        g["_slots"].append(slots)
        g["_budget_gb"].append(budget_bytes / 1e9)
        g["_hbm_gib"].append(budget_bytes / 2**30)
        g["_in_tok"].append(d.get("total_input_tokens"))
        g["_seed"].append(d.get("_seed"))

        # budget-consistency check: dense-equivalent budget should be ~ S*bps_dense.
        target = S * bps_dense
        rel = abs(budget_bytes - target) / target if target else 0
        if rel > 0.06:  # >6% drift -> bytes/slot assumption suspect
            drift_warn.append(
                (os.path.basename(f), bk, md, S, budget_bytes, target, rel)
            )

    # ---- text table ----
    print(
        "\n================ EQUAL-HBM serving control (mean +/- std over seeds) ================"
    )
    print(
        "workload: ShareGPT-V3 N40 shared-prefix, num-prompts=184 (all), 3 seeds {42,123,7}"
    )
    print(
        "budget fixed per S: bytes = S * bytes_per_slot[dense]; slots = round(budget/bytes_per_slot[mode])"
    )
    print("-" * 100)
    hdr = (
        f"{'bk':<4}{'mode':<8}{'S':>4}{'HBM_GiB':>9}{'slots':>7}{'n':>3}  "
        f"{'hit_rate':>18}{'TTFT_ms':>18}{'thr_tok/s':>18}   in_tok"
    )
    for bk in BACKENDS:
        print(hdr)
        for S in sorted({k[2] for k in group if k[0] == bk}):
            for md in MODES:
                key = (bk, md, S)
                if key not in group:
                    continue
                g = group[key]
                slots = int(round(sum(g["_slots"]) / len(g["_slots"])))
                hbm = sum(g["_hbm_gib"]) / len(g["_hbm_gib"])
                cells = []
                for jkey, _, _ in METRICS:
                    m, s, n = mean_std(g[jkey])
                    if m is None:
                        cells.append(f"{'NA':>18}")
                    elif jkey == "_token_hit_rate":
                        cells.append(f"{m:>10.4f}+/-{s:<6.4f}")
                    else:
                        cells.append(f"{m:>9.1f}+/-{s:<7.1f}")
                intoks = sorted({t for t in g["_in_tok"] if t is not None})
                intok_str = str(intoks[0]) if len(intoks) == 1 else f"VARIES{intoks}"
                n = mean_std(g["_token_hit_rate"])[2]
                print(
                    f"{bk:<4}{md:<8}{S:>4}{hbm:>9.2f}{slots:>7}{n:>3}  "
                    f"{cells[0]}{cells[1]}{cells[2]}   {intok_str}"
                )
        print("-" * 100)

    if drift_warn:
        print(
            "\n[WARN] budget drift (measured bytes/slot * slots vs target S*bytes_per_slot[dense] > 6%):"
        )
        for fn, bk, md, S, bb, tg, rel in drift_warn:
            print(
                f"   {fn}: {bk}/{md} S={S} measured={bb/1e9:.3f}GB target={tg/1e9:.3f}GB drift={rel*100:.1f}%"
            )
    else:
        print(
            "\n[ok] budget consistency verified (measured bytes/slot * slots ~= target for all records)."
        )

    # invariant reminder: within a backend, input tokens should be identical across
    # ALL arms/seeds (same workload) -> print the global set.
    all_intok = sorted(
        {t for g in group.values() for t in g["_in_tok"] if t is not None}
    )
    print(
        f"[invariant] distinct total_input_tokens across all records: {all_intok} "
        f"({'OK identical' if len(all_intok) == 1 else 'WARN differs'})"
    )

    # ---- figure (optional) ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(
            f"\n[plot] matplotlib unavailable ({e}); text table above is the citable output."
        )
        return

    colors = {
        "dense": "#888888",
        "padded": "#1f77b4",
        "ragged": "#d62728",
        "int8": "#2ca02c",
        "int8fair": "#ff7f0e",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    metric_titles = [
        "cached-token hit rate",
        "mean TTFT (ms)",
        "output throughput (tok/s)",
    ]
    for r, bk in enumerate(BACKENDS):
        Ss = sorted({k[2] for k in group if k[0] == bk})
        for c, (jkey, _, _) in enumerate(METRICS):
            ax = axes[r][c]
            for md in MODES:
                xs, ys, es = [], [], []
                for S in Ss:
                    key = (bk, md, S)
                    if key not in group:
                        continue
                    g = group[key]
                    hbm = sum(g["_hbm_gib"]) / len(g["_hbm_gib"])
                    m, s, n = mean_std(g[jkey])
                    if m is None:
                        continue
                    xs.append(hbm)
                    ys.append(m)
                    es.append(s)
                if xs:
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=es,
                        marker="o",
                        capsize=3,
                        color=colors[md],
                        label=md,
                    )
            ax.set_xlabel("checkpoint-pool HBM budget (GiB)")
            if c == 0:
                ax.set_ylabel(f"{bk.upper()}")
            if r == 0:
                ax.set_title(metric_titles[c])
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        "Equal-HBM control: compression converts fixed memory into hit-rate/latency "
        "(ShareGPT-V3 N40, mean+/-std over 3 seeds)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    png = os.path.splitext(out)[0] + ".png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"\n[plot] wrote {out} and {png}")


if __name__ == "__main__":
    main()
