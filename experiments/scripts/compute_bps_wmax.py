#!/usr/bin/env python3
"""GPU-free bytes-per-slot calibration for the KDA per-channel head-aware checkpoint
across the w_max sweep. Loads the real Kimi-Linear-48B A_log/dt_bias weights and
builds the SAME plan the live server builds (HeadAwarePlan.build_plan -> dispatches
to build_plan_per_channel because dt_bias is [L, HV*d_k] wider than A_log [L, HV]),
then computes bytes_per_slot for the ragged and padded layouts DIRECTLY from the
plan (no store allocation, no GPU).

The per-slot byte count is a CONSTANT (independent of pool size) so this offline
table seeds the orchestrator's slot/HBM math; the harness re-measures
_bytes_per_slot per launch and plot_wmax_sweep.py flags drift.

Cross-checks (must match live-measured values):
  W=0  -> 20,971,520 (dense: all (head,col) global)
  W=16 ragged -> 7,570,944 (existing N40 KDA ragged measurement)

Output: experiments/results/wmax_sweep/bps_wmax_table.json  (sourced by run_wmax_sweep.sh)

  python experiments/scripts/compute_bps_wmax.py
  python experiments/scripts/compute_bps_wmax.py --model ./Kimi-Linear-48B-A3B-Instruct
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# SGLang is a separate checkout. Editable installation is preferred; SGLANG_REPO
# also supports running directly against that checkout.
_BUNDLE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SGLANG_REPO = os.environ.get("SGLANG_REPO", "")
if _SGLANG_REPO:
    sys.path.insert(0, os.path.join(_SGLANG_REPO, "python"))

import torch  # noqa: E402

from sglang.srt.mem_cache.mamba_checkpoint_pool import HeadAwarePlan  # noqa: E402

KDA_MODEL_DEFAULT = os.environ.get(
    "KDA_MODEL",
    os.path.join(_SGLANG_REPO or _BUNDLE_ROOT, "Kimi-Linear-48B-A3B-Instruct"),
)
# Kimi-Linear-48B KDA dims: HV=32, d_k=d_v=128, 20 KDA layers.
HV_DEFAULT = 32
D_K_DEFAULT = 128
D_V_DEFAULT = 128
W_SWEEP = [0, 16, 64, 128, 512, 1024]
# dense-equivalent slot counts the orchestrator may want to budget for; the table
# reports floor(budget/bps) for each so the dry-run can check saturation + max-slots.
S_BUDGETS = [16, 24, 32, 48, 64, 96, 128]
BF16_ELEM_BYTES = 2


def _load_real_kda_weights(model_dir, d_k):
    """Return (A_log [L, HV], dt_bias [L, HV*d_k]) over all KDA layers.

    Kimi-Linear stores A_log as a per-head scalar and dt_bias as [HV*d_k]
    (per-channel, head-major). We keep only the per-channel layers (dt_bias
    wider than A_log by exactly d_k) and stack them per layer.
    """
    from safetensors import safe_open

    a_by_prefix, dt_by_prefix = {}, {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            for key in st.keys():
                if key.endswith(".A_log"):
                    a_by_prefix[key[: -len("A_log")]] = st.get_tensor(key)
                elif key.endswith(".dt_bias"):
                    dt_by_prefix[key[: -len("dt_bias")]] = st.get_tensor(key)

    def _layer_idx(k):
        m = re.search(r"layers?\.(\d+)\.", k)
        return int(m.group(1)) if m else -1

    prefixes = sorted(set(a_by_prefix) & set(dt_by_prefix), key=_layer_idx)
    kda = []
    for p in prefixes:
        a = a_by_prefix[p].float().flatten()  # [HV]
        dt = dt_by_prefix[p].float().flatten()  # [HV*d_k] for KDA
        if dt.numel() == a.numel() * d_k:
            kda.append((a, dt))
    if not kda:
        raise SystemExit(
            f"no KDA (per-channel) A_log/dt_bias pairs found under {model_dir}"
        )
    A_log = torch.stack([a for a, _ in kda], dim=0)  # [L, HV]
    dt_bias = torch.stack([dt for _, dt in kda], dim=0)  # [L, HV*d_k]
    return A_log, dt_bias


def _bps_from_plan(plan, L, d_v):
    """bytes_per_slot for ragged + padded, computed directly from the plan.

    Mirrors HeadAwareCheckpointStore.mem_usage_bytes() // num_slots for the
    no-quantizer bf16 case (the live KDA default):
      ragged : state_buf_pc = [num_slots, total_units, d_v]  (bf16)
               -> bytes_per_slot = total_units * d_v * BF16
      padded : state_buf_pc = [L, num_slots, GU_max, d_v]    (bf16)
               -> bytes_per_slot = L * GU_max * d_v * BF16
    total_units = #global (head, d_k-col) pairs = sum_l n_global[l].
    """
    is_local = plan.w_chan > 0  # [L, HV, K]
    total_units = int((~is_local).sum().item())  # total global (head,col) pairs
    GU_max = int(plan.GU_max) if plan.GU_max else 0
    bps_ragged = total_units * d_v * BF16_ELEM_BYTES
    bps_padded = L * GU_max * d_v * BF16_ELEM_BYTES
    return bps_ragged, bps_padded, total_units, GU_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=KDA_MODEL_DEFAULT)
    ap.add_argument("--d-k", type=int, default=D_K_DEFAULT)
    ap.add_argument("--d-v", type=int, default=D_V_DEFAULT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A_log, dt_bias = _load_real_kda_weights(args.model, args.d_k)
    L, HV = A_log.shape
    assert dt_bias.shape == (
        L,
        HV * args.d_k,
    ), f"dt_bias {dt_bias.shape} != ({L}, {HV * args.d_k})"
    print(f"[weights] {L} KDA layers, HV={HV}, d_k={args.d_k}, d_v={args.d_v}")

    # dense (W=0) is the reference; all (head,col) global.
    dense_total_units = L * HV * args.d_k
    dense_bps = dense_total_units * args.d_v * BF16_ELEM_BYTES
    print(f"[dense] total_units={dense_total_units} bytes_per_slot={dense_bps}")

    table = {}
    per_layer_counts = {}
    for W in W_SWEEP:
        plan = HeadAwarePlan.build_plan(
            A_log=A_log,
            dt_bias=dt_bias,
            route="A",
            d_k=args.d_k,
            d_v=args.d_v,
            eps=1e-3,
            w_max=W,
            a_margin=0.3,
        )
        bps_ragged, bps_padded, total_units, GU_max = _bps_from_plan(plan, L, args.d_v)
        comp_ragged = dense_bps / bps_ragged if bps_ragged else float("inf")
        comp_padded = dense_bps / bps_padded if bps_padded else float("inf")
        # per-layer global (head,col) counts — for the ragged-diag cross-check.
        is_local = plan.w_chan > 0
        n_global_per_layer = (~is_local).sum(dim=(1, 2)).tolist()
        table[W] = {
            "bytes_per_slot_ragged": bps_ragged,
            "bytes_per_slot_padded": bps_padded,
            "total_units": total_units,
            "GU_max": GU_max,
            "compression_x_ragged": round(comp_ragged, 4),
            "compression_x_padded": round(comp_padded, 4),
            "n_global_per_layer": n_global_per_layer,
            "W_max_actual": int(plan.W_max),
        }
        per_layer_counts[W] = n_global_per_layer
        sat_w16 = "  <-- W=16 SATURATES at S>=64" if W == 16 else ""
        print(
            f"[w_max={W:>4}] ragged={bps_ragged:>10} padded={bps_padded:>10}  "
            f"units={total_units:>6} GU_max={GU_max:>5} comp_ragged={comp_ragged:>6.2f}x "
            f"comp_padded={comp_padded:>6.2f}x plan.W_max={int(plan.W_max)}{sat_w16}"
        )

    # ---- cross-checks ----
    # W=0 == dense == 20,971,520 is STRUCTURAL (all (head,col) global; code-version
    # independent) -> hard check. W=16 vs the prior live measurement (7,570,944 from
    # 2026-07-31) is a SOFT drift check: the working-tree plan builder may have
    # changed since that measurement, so a mismatch is informational, not a failure.
    # The live run re-measures _bytes_per_slot authoritatively per launch; the
    # orchestrator uses this offline table only as the initial CKPT_W estimate.
    ok0 = table[0]["bytes_per_slot_ragged"] == dense_bps == 20_971_520
    prior_w16 = 7_570_944
    drift16 = table[16]["bytes_per_slot_ragged"] - prior_w16
    print(
        "\n[cross-check] W=0 ragged == dense == 20,971,520 : "
        f"{'PASS' if ok0 else 'FAIL'} ({table[0]['bytes_per_slot_ragged']})"
    )
    print(
        f"[info] W=16 ragged vs prior live {prior_w16}: "
        f"{table[16]['bytes_per_slot_ragged']} (drift {drift16:+d}; "
        f"{'matches' if drift16 == 0 else 'code drifted since 2026-07-31; live re-measures'})"
    )

    # ---- fixed-HBM slot counts (FLOOR, never round) for the budgets the
    # orchestrator may use. Reports max slots so the dry-run can warn on huge pools.
    print("\n[floor slot counts] CKPT_W = floor(S * dense_bps / bps_ragged(W)):")
    print(f"{'S':>4} {'budget_GiB':>11} |" + "".join(f" W={W:<5}" for W in W_SWEEP[1:]))
    slot_table = {}
    for S in S_BUDGETS:
        budget = S * dense_bps
        budget_gib = budget / (1 << 30)
        row = []
        for W in W_SWEEP[1:]:  # skip W=0 (dense)
            bps = table[W]["bytes_per_slot_ragged"]
            ckpt_w = budget // bps  # FLOOR (never round -> never exceed budget)
            row.append(ckpt_w)
        slot_table[S] = {
            "budget_bytes": budget,
            "slots_by_W": dict(zip(W_SWEEP[1:], row)),
        }
        print(f"{S:>4} {budget_gib:>10.2f}GiB |" + "".join(f" {n:<7}" for n in row))

    max_slots = max(
        row for st in slot_table.values() for row in st["slots_by_W"].values()
    )
    print(
        f"\n[max-slots] largest CKPT_W in the table = {max_slots}"
        + (
            "  (WARN >10000: slot metadata may be non-negligible)"
            if max_slots > 10000
            else ""
        )
    )

    # saturation hint: at S slots, DASC ragged W=16 slots vs distinct=184.
    # If floor(S * dense_bps / bps_ragged(16)) >= 184 -> W=16 already saturated.
    bps16 = table[16]["bytes_per_slot_ragged"]
    for S in S_BUDGETS:
        slots16 = (S * dense_bps) // bps16
        sat = "SATURATED" if slots16 >= 184 else "unsaturated"
        print(
            f"[saturation] S={S}: W=16 ragged slots={slots16} vs distinct=184 -> {sat}"
        )

    out_dir = os.path.join(_BUNDLE_ROOT, "experiments", "results", "wmax_sweep")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "bps_wmax_table.json")
    payload = {
        "model": os.path.basename(args.model),
        "L": L,
        "HV": HV,
        "d_k": args.d_k,
        "d_v": args.d_v,
        "dense_bps": dense_bps,
        "dense_total_units": dense_total_units,
        "elem_bytes": BF16_ELEM_BYTES,
        "crosscheck_W0_dense": ok0,
        "info_W16_vs_prior_live": prior_w16,
        "info_W16_drift": drift16,
        "table": {str(k): v for k, v in table.items()},
        "slot_floor_table": {str(s): v for s, v in slot_table.items()},
        "max_slots_seen": max_slots,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
