#!/usr/bin/env python3
"""Aggregate + plot the W_MAX sweep serving experiment (run_wmax_sweep.sh output).

Two figures decompose the DASC benefit:

  Figure A (Group 1, fixed slots — isolates recon cost):
    x = W, y = paired ΔTTFT(W) = TTFT_WR(W) − TTFT_NR(W) with error bars (paired
    delta std over seeds). Also absolute TTFT_NR / TTFT_WR curves. The paired
    delta is computed PER SEED (same request order; only NR→WR differs) so it
    isolates the reconstruction overhead at fixed slot count.

  Figure B (Groups 2+3, fixed HBM — compression benefit + net):
    3 panels (hit rate / TTFT / throughput), x = W, 4 curves
    (Dense horizontal, INT8 horizontal, DASC-NR, DASC-WR) with error bars.
    Slots per W annotated.
    Gain signs (positive = benefit):
      G_NR(W) = T_dense − T_NR(W)    (compression benefit, no recon)
      C_WR(W) = T_WR(W) − T_NR(W)   (recon overhead)
      G_net(W) = T_dense − T_WR(W)  (net benefit)

P0 guards:
  - FLOOR budget cross-check: measured _bytes_per_slot * _ckpt_slots ≈ HBM budget
    for G23; warns on drift.
  - Paired-delta completeness: if any NR/WR pair at a W is missing seeds, the
    paired delta for that W is REFUSED (not silently averaged over fewer seeds).
  - Accuracy boundary: W=128 marked as the validated operating point;
    W>=256 marked as scaling-study-only (no e2e accuracy data).

  Usage: python experiments/scripts/plot_wmax_sweep.py [--logdir DIR]
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

W_SWEEP = (16, 64, 128, 512, 1024)
VALIDATED_W = 128  # KDA end-task WR validated to W=128; >=256 is scaling-study-only
METRICS = [
    ("mean_ttft_ms", "TTFT (ms)", 1.0),
    ("output_throughput", "throughput (tok/s)", 1.0),
    ("_token_hit_rate", "cached-token hit rate", 1.0),
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
            os.path.dirname(__file__), "..", "..", "experiments", "results", "wmax_sweep"
        ),
    )
    ap.add_argument(
        "--slots",
        type=int,
        default=None,
        help="G1 fixed-slot count (= G23 dense slots). Auto-detected from the dense record if omitted.",
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    logdir = os.path.abspath(args.logdir)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "..", "..", "docs", "iclr_paper"
    )
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(logdir, "result_*_seed*.json")))
    if not files:
        print(f"[plot] no result_*_seed*.json under {logdir}")
        return

    # ---- load + classify ----
    # dense (G23 baseline) tells us SLOTS; ragged with slots==SLOTS -> G1, else G23.
    dense_slots = None
    recs = []  # each: dict with fields
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"[plot] skip unreadable {os.path.basename(f)}: {e}")
            continue
        md = d.get("_dasc_mode")
        w = d.get("_wmax")
        slots = d.get("_ckpt_slots")
        bps = d.get("_bytes_per_slot")
        recon = d.get("_recon")  # True/False/None
        seed = d.get("_seed")
        if md is None or slots is None:
            continue
        # NR vs WR: recon True => WR; recon False/None and (w==0 or not recon) => NR.
        # dense has w==0; treat as its own arm.
        is_wr = bool(recon)
        recs.append(
            {
                "f": os.path.basename(f),
                "mode": md,
                "w": w,
                "slots": slots,
                "bps": bps,
                "is_wr": is_wr,
                "seed": seed,
                "ttft": d.get("mean_ttft_ms"),
                "ttft_p95": d.get("p95_ttft_ms"),
                "thr": d.get("output_throughput"),
                "hit": d.get("_token_hit_rate"),
                "in_tok": d.get("total_input_tokens"),
                "peak_mem": d.get("_peak_mem_gb"),
                "recon_n": d.get("_recon_n_hits"),
                "replay_mean": d.get("_replay_tokens_mean"),
                "replay_p95": d.get("_replay_tokens_p95"),
            }
        )
        if md == "dense" and dense_slots is None:
            dense_slots = slots

    S = args.slots or dense_slots
    if S is None:
        print("[plot] cannot determine G1 fixed-slot count; pass --slots")
        return

    # split: G1 (ragged, slots==S) vs G23 (dense + ragged slots!=S)
    g1 = [r for r in recs if r["mode"] == "ragged" and r["slots"] == S]
    g23 = [r for r in recs if r["mode"] == "dense" or r["slots"] != S]
    print(
        f"[load] {len(recs)} records: G1={len(g1)} G23={len(g23)} S(G1 fixed slots / G23 dense slots)={S}"
    )

    # ---- aggregate per (arm, W) ----
    def agg(rs, w_filter=None, wr=None):
        out = {}
        for r in rs:
            if w_filter is not None and r["w"] != w_filter:
                continue
            if wr is not None and r["is_wr"] != wr:
                continue
            out.setdefault(r["w"], []).append(r)
        return out

    # G1: NR/WR per W, keyed by seed for paired delta
    g1_nr = defaultdict(dict)  # g1_nr[W][seed] = rec
    g1_wr = defaultdict(dict)
    for r in g1:
        (g1_wr if r["is_wr"] else g1_nr)[r["w"]][r["seed"]] = r

    # G23: dense (W=None arm) + int8fair (W-independent horizontal) + DASC NR/WR per W
    g23_dense = [r for r in g23 if r["mode"] == "dense"]
    g23_int8 = [r for r in g23 if r["mode"] == "int8fair"]
    g23_nr = defaultdict(dict)
    g23_wr = defaultdict(dict)
    for r in g23:
        if r["mode"] in ("dense", "int8fair"):
            continue
        (g23_wr if r["is_wr"] else g23_nr)[r["w"]][r["seed"]] = r

    # ---- text table ----
    print("\n================ W_MAX sweep (mean +/- std over seeds) ================")
    print(f"G1 fixed slots = {S}; G23 fixed HBM = {S} * dense_bps")
    print("-" * 110)

    def row_agg(rs, key):
        m, s, n = mean_std([r[key] for r in rs])
        return f"{m:>9.2f}+/-{s:<6.2f}" if m is not None else f"{'NA':>17}"

    print(
        f"{'grp':<4}{'W':>5}{'arm':>8}{'n':>3}  {'hit_rate':>18}{'TTFT_ms':>18}{'thr':>18}  "
        f"{'peak_mem':>9}{'recon_n':>9}{'replay_mean':>12}"
    )
    for W in sorted(g1_nr.keys() | g1_wr.keys()):
        for arm, src in (("NR", g1_nr), ("WR", g1_wr)):
            rs = list(src[W].values())
            if not rs:
                continue
            pk = rs[0]["peak_mem"]
            pk_s = f"{pk:.2f}" if pk is not None else "NA"
            rm = rs[0]["replay_mean"]
            rm_s = f"{rm:.0f}" if rm is not None else "NA"
            print(
                f"G1  {W:>5}{arm:>8}{len(rs):>3}  {row_agg(rs,'hit')}{row_agg(rs,'ttft')}{row_agg(rs,'thr')}  "
                f"{pk_s:>9}{(rs[0]['recon_n'] or 0):>9}{rm_s:>12}"
            )
    print("-" * 60)
    # G23 dense
    if g23_dense:
        rs = g23_dense
        print(
            f"G23 dense{'':>3}{'':>8}{len(rs):>3}  {row_agg(rs,'hit')}{row_agg(rs,'ttft')}{row_agg(rs,'thr')}"
        )
    # G23 int8fair (W-independent horizontal baseline)
    if g23_int8:
        rs = g23_int8
        slots_i8 = rs[0]["slots"]
        print(
            f"G23 int8{'':>4}{'':>8}{len(rs):>3}  {row_agg(rs,'hit')}{row_agg(rs,'ttft')}{row_agg(rs,'thr')}  "
            f"slots={slots_i8}"
        )
    for W in sorted(g23_nr.keys() | g23_wr.keys()):
        for arm, src in (("NR", g23_nr), ("WR", g23_wr)):
            rs = list(src[W].values())
            if not rs:
                continue
            slots = rs[0]["slots"]
            print(
                f"G23 {W:>5}{arm:>8}{len(rs):>3}  {row_agg(rs,'hit')}{row_agg(rs,'ttft')}{row_agg(rs,'thr')}  "
                f"slots={slots}"
            )

    # ---- P0: paired-delta completeness (Figure A needs NR+WR at same W, same seeds) ----
    print("\n[paired-delta completeness] G1 NR/WR per W:")
    delta_ok = True
    for W in W_SWEEP:
        nr_seeds = set(g1_nr.get(W, {}).keys())
        wr_seeds = set(g1_wr.get(W, {}).keys())
        common = nr_seeds & wr_seeds
        missing = (nr_seeds | wr_seeds) - common
        status = (
            "OK"
            if not missing and len(common) >= 2
            else f"REFUSE ({len(common)} paired, missing {missing})"
        )
        if "REFUSE" in status:
            delta_ok = False
        print(
            f"  W={W}: NR={len(nr_seeds)} WR={len(wr_seeds)} paired={len(common)} -> {status}"
        )

    # ---- P0: floor budget cross-check (G23) ----
    print("\n[floor-budget cross-check] G23 measured bps*slots vs budget:")
    dense_bps = g23_dense[0]["bps"] if g23_dense else None
    if dense_bps is None and g23:
        dense_bps = next(r["bps"] for r in g23 if r["bps"])
    if dense_bps:
        budget = S * dense_bps
        for W in W_SWEEP:
            rs = list(g23_nr.get(W, {}).values()) or list(g23_wr.get(W, {}).values())
            if not rs:
                continue
            bps = rs[0]["bps"]
            slots = rs[0]["slots"]
            used = bps * slots
            rel = abs(used - budget) / budget if budget else 0
            flag = "OK" if rel < 0.02 else f"DRIFT {rel*100:.1f}%"
            print(
                f"  W={W}: bps={bps} slots={slots} used={used} budget={budget} -> {flag}"
            )
    if g23_int8:
        bps = g23_int8[0]["bps"]
        slots = g23_int8[0]["slots"]
        used = bps * slots
        rel = abs(used - budget) / budget if budget else 0
        flag = "OK" if rel < 0.02 else f"DRIFT {rel*100:.1f}%"
        print(
            f"  int8fair: bps={bps} slots={slots} used={used} budget={budget} -> {flag}"
        )

    # ---- gain signs (G23) ----
    print(
        "\n[gain signs] (positive = benefit) G_NR=T_d-T_NR, C_WR=T_WR-T_NR, G_net=T_d-T_WR:"
    )
    if g23_dense:
        d_ttft, _, _ = mean_std([r["ttft"] for r in g23_dense])
        if g23_int8:
            i8_ttft, _, _ = mean_std([r["ttft"] for r in g23_int8])
            print(
                f"  int8fair: T_i8={i8_ttft:.1f}  G_i8=T_d-T_i8={d_ttft-i8_ttft:+.1f}"
            )
        for W in W_SWEEP:
            nr = list(g23_nr.get(W, {}).values())
            wr = list(g23_wr.get(W, {}).values())
            nr_m, _, _ = mean_std([r["ttft"] for r in nr]) if nr else (None, None, 0)
            wr_m, _, _ = mean_std([r["ttft"] for r in wr]) if wr else (None, None, 0)
            if d_ttft and nr_m and wr_m:
                print(
                    f"  W={W}: G_NR={d_ttft-nr_m:+.1f}  C_WR={wr_m-nr_m:+.1f}  G_net={d_ttft-wr_m:+.1f}  "
                    f"(T_d={d_ttft:.1f} T_NR={nr_m:.1f} T_WR={wr_m:.1f})"
                )

    # ---- figures ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(
            f"\n[plot] matplotlib unavailable ({e}); text table above is the citable output."
        )
        return

    # Figure A: G1 paired ΔTTFT vs W + absolute NR/WR curves
    figA, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ws, deltas, derrs = [], [], []
    abs_nr_m, abs_nr_s, abs_wr_m, abs_wr_s = [], [], [], []
    for W in W_SWEEP:
        nr = g1_nr.get(W, {})
        wr = g1_wr.get(W, {})
        common = sorted(set(nr) & set(wr))
        if len(common) < 2:
            continue
        d = [
            wr[s]["ttft"] - nr[s]["ttft"]
            for s in common
            if wr[s]["ttft"] is not None and nr[s]["ttft"] is not None
        ]
        if len(d) < 2:
            continue
        m, s, _ = mean_std(d)
        ws.append(W)
        deltas.append(m)
        derrs.append(s)
        nr_m, nr_s, _ = mean_std(
            [nr[s]["ttft"] for s in common if nr[s]["ttft"] is not None]
        )
        wr_m, wr_s, _ = mean_std(
            [wr[s]["ttft"] for s in common if wr[s]["ttft"] is not None]
        )
        abs_nr_m.append(nr_m)
        abs_nr_s.append(nr_s)
        abs_wr_m.append(wr_m)
        abs_wr_s.append(wr_s)
    ax1.errorbar(
        ws,
        deltas,
        yerr=derrs,
        marker="o",
        capsize=4,
        color="#d62728",
        label="ΔTTFT = WR−NR (paired)",
    )
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_xlabel("W_max (forced head-aware window)")
    ax1.set_ylabel("ΔTTFT (ms)  [positive = WR slower]")
    ax1.set_title("Group 1 (fixed slots): reconstruction overhead")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.errorbar(
        ws,
        abs_nr_m,
        yerr=abs_nr_s,
        marker="s",
        capsize=3,
        color="#1f77b4",
        label="NR (no recon)",
    )
    ax2.errorbar(
        ws,
        abs_wr_m,
        yerr=abs_wr_s,
        marker="^",
        capsize=3,
        color="#d62728",
        label="WR (recon)",
    )
    ax2.set_xlabel("W_max")
    ax2.set_ylabel("mean TTFT (ms)")
    ax2.set_xscale("log")
    ax2.set_title("Group 1: absolute TTFT (NR vs WR)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    figA.suptitle(
        "Figure A: fixed-slot reconstruction overhead (paired over seeds)", fontsize=12
    )
    figA.tight_layout(rect=[0, 0, 1, 0.96])
    fA = os.path.join(out_dir, "wmax_sweep_figA.pdf")
    figA.savefig(fA, bbox_inches="tight")
    figA.savefig(os.path.splitext(fA)[0] + ".png", dpi=130, bbox_inches="tight")
    print(f"\n[plot] wrote {fA}")

    # Figure B: G23 3-panel (hit / TTFT / thr), 3 curves (Dense / DASC-NR / DASC-WR)
    figB, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["cached-token hit rate", "mean TTFT (ms)", "output throughput (tok/s)"]
    keys = ["hit", "ttft", "thr"]
    dense_vals = {
        k: mean_std([r[k] for r in g23_dense]) if g23_dense else (None, None, 0)
        for k in keys
    }
    for c, (key, _, _) in enumerate(zip(keys, titles, [0] * 3)):
        ax = axes[c]
        # dense horizontal
        dm, ds, _ = dense_vals[key]
        if dm is not None:
            xs = [min(ws) if ws else 16, max((ws + [1024]))]
            ax.plot(
                xs, [dm] * 2, "--", color="#888888", lw=2, label="Dense (matched HBM)"
            )
            ax.fill_between(
                xs, [dm - ds] * 2, [dm + ds] * 2, color="#888888", alpha=0.15
            )
        # int8fair horizontal (W-independent uniform INT8 quantization baseline)
        if g23_int8:
            i8m, i8s, _ = mean_std([r[key] for r in g23_int8])
            if i8m is not None:
                ax.plot(
                    xs,
                    [i8m] * 2,
                    "--",
                    color="#2ca02c",
                    lw=2,
                    label="INT8 (matched HBM)",
                )
                ax.fill_between(
                    xs, [i8m - i8s] * 2, [i8m + i8s] * 2, color="#2ca02c", alpha=0.15
                )
        # NR + WR
        for arm, src, col, mk in (
            ("NR", g23_nr, "#1f77b4", "s"),
            ("WR", g23_wr, "#d62728", "^"),
        ):
            xs2, ys, es = [], [], []
            for W in W_SWEEP:
                rs = list(src.get(W, {}).values())
                m, s, n = mean_std([r[key] for r in rs]) if rs else (None, None, 0)
                if m is None:
                    continue
                xs2.append(W)
                ys.append(m)
                es.append(s)
            if xs2:
                ax.errorbar(
                    xs2,
                    ys,
                    yerr=es,
                    marker=mk,
                    capsize=3,
                    color=col,
                    label=f"DASC-{arm}",
                )
        ax.set_xlabel("W_max")
        ax.set_xscale("log")
        ax.set_title(titles[c])
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend(fontsize=8)
        # accuracy-boundary shading: validated to W=128, scaling-only beyond
        ax.axvspan(VALIDATED_W, 1024 * 1.5, color="orange", alpha=0.08)
        if c == 0:
            ax.text(
                VALIDATED_W * 1.3,
                ax.get_ylim()[0],
                "scaling-study\n(no e2e acc)",
                fontsize=7,
                color="orange",
                va="bottom",
            )
        # annotate slots per W
        if c == 2:
            for W in W_SWEEP:
                rs = list(g23_nr.get(W, {}).values())
                if rs:
                    ax.annotate(
                        f"{rs[0]['slots']}",
                        (W, ax.get_ylim()[1]),
                        fontsize=6,
                        ha="center",
                        color="gray",
                    )
    figB.suptitle(
        "Figure B: matched-HBM (compression benefit + net). Slots annotated on throughput panel.",
        fontsize=11,
    )
    figB.tight_layout(rect=[0, 0, 1, 0.96])
    fB = os.path.join(out_dir, "wmax_sweep_figB.pdf")
    figB.savefig(fB, bbox_inches="tight")
    figB.savefig(os.path.splitext(fB)[0] + ".png", dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {fB}")

    if not delta_ok:
        print(
            "\n[WARN] paired-delta REFUSED for some W (missing NR/WR seeds); "
            "Figure A omits those W. Rerun the missing arms before citing."
        )


if __name__ == "__main__":
    main()
