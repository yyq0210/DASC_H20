#!/usr/bin/env bash
# INT8 equal-HBM serving experiment (KDA only).
#
# Adds INT8-quantized arms to the equal-HBM table (Table 8). At each S budget,
# INT8 gets floor(S * dense_bps / int8_bps) slots — more slots from smaller
# bytes/slot at the SAME per-rank HBM. This isolates "uniform INT8 quantization"
# vs "DASC decay-aware drop" at matched HBM.
#
# INT8 pool stores 16 heads/rank (TP-split) vs head-aware's 32 heads/rank
# (replicated), so the slot gain (~3.9x) exceeds the 1.97x pure-quantization
# ratio. The table reports actual measured HBM so the comparison is honest.
#
# Usage:  bash experiments/scripts/run_equal_hbm_int8.sh [--dry-run]
#         CHAIN_PID=3707262 bash experiments/scripts/run_equal_hbm_int8.sh  # wait for chain
set -uo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

NGROUPS="${NGROUPS:-40}"
S_LIST="${S_LIST:-48 96 144}"
SEEDS="${SEEDS:-42,123,7}"
FREE_GB="${FREE_GB:-40}"
DENSE_BPS=20971520   # KDA dense head-aware bytes/slot (20 layers, 32 heads, BF16)
LOGDIR="${LOGDIR:-$PWD/experiments/results/equal_hbm}"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ORCH="$LOGDIR/orchestrator_int8_${STAMP}.log"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }

wait_gpu_free(){
  local need=$(( FREE_GB * 1024 ))
  for _ in $(seq 1 120); do
    local minfree
    minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
    [ -z "$minfree" ] && { sleep 5; continue; }
    if [ "$minfree" -ge "$need" ]; then
      log "  gpu free ok (min ${minfree}MiB >= ${need}MiB)"; return 0
    fi
    sleep 10
  done
  log "  WARN: gpu never freed to ${need}MiB; proceeding anyway"; return 0
}

# Phase 0: calibrate INT8 bps (1-seed launch at CKPT=48)
calibrate_int8_bps(){
  local cal_ckpt=48
  local cal_label="kda_int8_N${NGROUPS}_w0_ckpt${cal_ckpt}_recon0"
  local cal_log="$LOGDIR/cal_${cal_label}.log"
  log "=== Phase 0: calibrate INT8 bps (1 seed, ckpt=$cal_ckpt) ==="
  wait_gpu_free
  CKPT_OVERRIDE="$cal_ckpt" MEASURE_SEEDS="${SEEDS%%,*}" NGROUPS="$NGROUPS" \
    LOGDIR="$LOGDIR" EVICT="${EVICT:-lru}" MAX_CONC="${MAX_CONC:-16}" \
    OUTPUT_LEN="${OUTPUT_LEN:-64}" \
    bash experiments/scripts/bench_arena_shared_prefix.sh kda int8 >"$cal_log" 2>&1
  local rc=$?
  local cal_json
  cal_json=$(ls "$LOGDIR"/result_${cal_label}_seed*.json 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || [ -z "$cal_json" ]; then
    log "  cal FAILED (rc=$rc); tail:"; tail -30 "$cal_log" | tee -a "$ORCH"
    return 1
  fi
  INT8_BPS=$(python3 -c "import json; print(json.load(open('$cal_json')).get('_bytes_per_slot', 0))")
  if [ "${INT8_BPS:-0}" -gt 0 ] 2>/dev/null; then
    log "  cal: int8 bps=$INT8_BPS (dense=$DENSE_BPS, ratio=$(python3 -c "print(f'{$DENSE_BPS/$INT8_BPS:.2f}')"))"
  else
    log "  cal: bps extraction FAILED"; return 1
  fi
}

# compute slots: floor(S * dense_bps / int8_bps)
calc_int8_slots(){  # $1=S
  python3 - "$DENSE_BPS" "$INT8_BPS" "$1" <<'PY'
import sys, math
bd, bi, S = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
print(math.floor(S * bd / bi))
PY
}

# ---- dry-run: print slot/HBM table ----
if [ "$DRY_RUN" -eq 1 ]; then
  # Use offline estimate: INT8 bps ≈ 5,324,800 (10 layers * (16*128*128*1 + 16*128*2) per rank)
  INT8_BPS=5324800
  echo "=== INT8 equal-HBM dry-run (bps=OFFLINE EST, will live-calibrate) ==="
  printf "%-4s %-7s %-4s %8s %10s %10s\n" backend mode S slots HBM_GiB budget_GiB
  for S in $S_LIST; do
    B=$(( S * DENSE_BPS ))
    slots=$(calc_int8_slots "$S")
    hbm=$(python3 -c "print(f'{$slots * $INT8_BPS / 2**30:.2f}')")
    budget=$(python3 -c "print(f'{$B / 2**30:.2f}')")
    printf "%-4s %-7s %-4d %8d %10s %10s\n" kda int8 "$S" "$slots" "$hbm" "$budget"
  done
  echo
  echo "NOTE: live-calibrated bps will replace this estimate; slots may differ."
  exit 0
fi

# ---- full sweep ----
export NGROUPS LOGDIR

# Ensure dataset exists
DS="/tmp/arena_shared_prefix_N${NGROUPS}.jsonl"
if [ ! -f "$DS" ]; then
  log "building dataset N=$NGROUPS ..."
  no_proxy='*' python experiments/scripts/build_arena_shared_prefix.py --n-groups "$NGROUPS" --out "$DS" \
    >>"$ORCH" 2>&1 || { log "FATAL dataset build failed"; exit 1; }
fi

log "=== INT8 EQUAL-HBM sweep START  N=$NGROUPS S=[$S_LIST] seeds=$SEEDS ==="

# Phase 0: calibrate
calibrate_int8_bps || { log "FATAL: calibration failed"; exit 1; }

# Phase 1: run 3 seeds at each S
i=0; TOTAL=$(echo $S_LIST | wc -w)
for S in $S_LIST; do
  i=$((i+1))
  B=$(( S * DENSE_BPS ))
  slots=$(calc_int8_slots "$S")
  hbm=$(python3 -c "print(f'{$slots * $INT8_BPS / 2**30:.2f}')")
  budget=$(python3 -c "print(f'{$B / 2**30:.2f}')")
  log "--- launch $i/$TOTAL: kda int8 S=$S -> ckpt=$slots HBM=${hbm}GiB (budget=${budget}GiB) ---"
  wait_gpu_free
  ARMLOG="$LOGDIR/arm_kda_int8_N${NGROUPS}_S${S}.log"
  CKPT_OVERRIDE="$slots" MEASURE_SEEDS="$SEEDS" LOGDIR="$LOGDIR" NGROUPS="$NGROUPS" \
    EVICT="${EVICT:-lru}" MAX_CONC="${MAX_CONC:-16}" OUTPUT_LEN="${OUTPUT_LEN:-64}" \
    bash experiments/scripts/bench_arena_shared_prefix.sh kda int8 >"$ARMLOG" 2>&1
  rc=$?
  RES=$(ls -t "$LOGDIR"/result_kda_int8_N${NGROUPS}_*_ckpt${slots}_*_seed*.json 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || [ -z "$RES" ]; then
    log "  launch kda/int8 S=$S FAILED (rc=$rc); tail:"; tail -30 "$ARMLOG" | tee -a "$ORCH"
    log "  continuing to next launch."
  else
    njsons=$(ls "$LOGDIR"/result_kda_int8_N${NGROUPS}_*_ckpt${slots}_*_seed*.json 2>/dev/null | wc -l)
    log "  launch kda/int8 S=$S OK -> $njsons seed JSONs"
    grep -E "^\[result\]" "$ARMLOG" | tail -3 | tee -a "$ORCH" || true
    # xcheck: live bps * slots <= budget
    python3 - "$RES" "$B" "$S" "$slots" <<'PY'
import json, sys
res, budget, S, slots = sys.argv[1:5]
budget = int(budget); S = int(S); slots = int(slots)
d = json.load(open(res))
live_bps = d.get("_bytes_per_slot", 0)
used = live_bps * slots
slack = budget - used
rel = abs(used - budget) / budget if budget else 0
status = "OK" if rel < 0.05 else f"DRIFT {rel*100:.1f}%"
print(f"  [xcheck] kda/int8 S={S}: live_bps={live_bps} slots={slots} used={used} budget={budget} slack={slack} -> {status}")
PY
  fi
done

log "=== INT8 sweep DONE ==="
log "=== Now running plot_equal_hbm.py to aggregate (includes int8) ==="
python experiments/scripts/plot_equal_hbm.py --logdir "$LOGDIR" 2>&1 | tee -a "$ORCH" || \
  log "  plot failed (run manually: python experiments/scripts/plot_equal_hbm.py)"
log "=== ALL DONE ==="
