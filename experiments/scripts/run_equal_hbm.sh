#!/usr/bin/env bash
# EQUAL-HBM serving control experiment (generalizes run_arena_ab_all.sh).
#
# The prior 6-arm serving A/B compared dense (small pool) vs DASC (big pool) at
# DIFFERENT HBM -> a reviewer can dismiss the DASC win as "just spent more HBM".
# This sweep instead FIXES the checkpoint-pool memory budget (equal bytes) and
# lets dense/padded/ragged use the SAME byte count, so any hit-rate/TTFT/thr gain
# is purely from compression (smaller bytes/slot -> more prefixes fit the SAME HBM
# -> higher hit rate) and NOT from extra memory.
#
# Design (see docs plan): budgets are parameterized by a dense-equivalent slot
# count S in {48,96,144}; budget B = S * bytes_per_slot[dense]; each arm's slots =
# round(B / bytes_per_slot[arm]). S < distinct(=184) so dense STARVES at every
# point while DASC holds more (padded/ragged compress bytes/slot). Each launch does
# 1 warm + 3 measure passes (seeds 42,123,7) on the SAME workload (num-prompts=184
# = all requests -> every pass has identical input tokens; seed only reshuffles
# order -> clean steady-state scheduling/eviction variance).
#
#   Usage:  bash experiments/scripts/run_equal_hbm.sh [--dry-run]
#   Env:    S_LIST="48 96 144"  SEEDS="42,123,7"  NGROUPS=40  FREE_GB=40
#           BACKENDS="kda gdn"
#
# 2 backend x 3 budget x 3 mode = 18 launches, SEQUENTIAL (2x H20 TP2 hosts one
# 48B/80B at a time). Any failure is logged and the sweep continues. Background
# with nohup; progress in orchestrator_equal_hbm_<STAMP>.log.
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

NGROUPS="${NGROUPS:-40}"
S_LIST="${S_LIST:-48 96 144}"
SEEDS="${SEEDS:-42,123,7}"
BACKENDS="${BACKENDS:-kda gdn}"
FREE_GB="${FREE_GB:-40}"
LOGDIR="${LOGDIR:-$PWD/experiments/results/equal_hbm}"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ORCH="$LOGDIR/orchestrator_equal_hbm_${STAMP}.log"

# bytes/slot per (backend,mode). Dense is STRUCTURAL (always correct); padded/
# ragged are LIVE-CALIBRATED in Phase 0 below (hardcoded values caused a stale-bps
# bug where padded HBM drifted to 2.04 GiB instead of 1.88 — see iclr_paper notes).
# After calibration, BPS[backend:mode] is overwritten with the live-measured value.
declare -A BPS=(
  [kda:dense]=20971520  [kda:padded]=0  [kda:ragged]=0
  [gdn:dense]=37748736  [gdn:padded]=0  [gdn:ragged]=0
)
# First seed for calibration (quick 1-seed pass to measure live bytes/slot)
CAL_SEED="${SEEDS%%,*}"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }

# compute slots for one (backend,mode,S): floor(S * bps[dense] / bps[mode])
# FLOOR guarantees bps*slots <= budget (never over-budget). Slack is reported.
calc_slots(){  # $1=backend $2=mode $3=S -> echoes slot count
  python3 - "${BPS[$1:dense]}" "${BPS[$1:$2]}" "$3" <<'PY'
import sys, math
bd,bm,S=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
print(math.floor(S*bd/bm))
PY
}

# Phase 0 helper: run one quick launch to measure live bytes_per_slot for a
# (backend, mode) pair. Uses CKPT=S (dense-equivalent) + 1 seed. Reads
# _bytes_per_slot from the result JSON and overwrites BPS[backend:mode].
calibrate_bps(){  # $1=backend $2=mode $3=S
  local bk="$1" md="$2" S="$3"
  local cal_slots="$S"
  [ "$md" = "dense" ] && { log "  cal $bk/$md: structural bps=${BPS[$bk:dense]} (skip)"; return 0; }
  local cal_label="${bk}_${md}_N${NGROUPS}_w$([ "$bk" = kda ] && echo 16 || echo 64)_ckpt${cal_slots}_recon0"
  local cal_log="$LOGDIR/cal_${cal_label}.log"
  log "  cal $bk/$md: launching 1-seed calibration (ckpt=$cal_slots)..."
  wait_gpu_free
  WMAX_DASC="$([ "$bk" = kda ] && echo 16 || echo 64)" RECON=0 CKPT_OVERRIDE="$cal_slots" \
    MEASURE_SEEDS="$CAL_SEED" NGROUPS="$NGROUPS" LOGDIR="$LOGDIR" EVICT="${EVICT:-lru}" \
    MAX_CONC="${MAX_CONC:-16}" OUTPUT_LEN="${OUTPUT_LEN:-64}" \
    bash experiments/scripts/bench_arena_shared_prefix.sh "$bk" "$md" >"$cal_log" 2>&1
  local rc=$?
  local cal_json
  cal_json=$(ls "$LOGDIR"/result_${cal_label}_seed*.json 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || [ -z "$cal_json" ]; then
    log "  cal $bk/$md FAILED (rc=$rc); using offline fallback"
    return 1
  fi
  local live_bps
  live_bps=$(python3 -c "import json; print(json.load(open('$cal_json')).get('_bytes_per_slot', 0))")
  if [ "$live_bps" -gt 0 ] 2>/dev/null; then
    BPS[$bk:$md]=$live_bps
    log "  cal $bk/$md: live bps=$live_bps (was 0/placeholder)"
  else
    log "  cal $bk/$md: bps extraction FAILED; using 0"
  fi
}

# ---- dry-run: print the full 18-row slot/HBM table and self-check invariants ----
if [ "$DRY_RUN" -eq 1 ]; then
  # dry-run uses OFFLINE bps estimates (live calibration not available without GPU)
  BPS[kda:padded]=11709440; BPS[kda:ragged]=7523584
  BPS[gdn:padded]=37748736; BPS[gdn:ragged]=27983872
  echo "=== EQ-GATE-0 dry-run: budget/slot table (distinct=184; bps=OFFLINE EST) ==="
  echo "  NOTE: padded/ragged bps are OFFLINE estimates; live calibration may differ."
  printf "%-4s %-7s %-4s %8s %10s %10s\n" backend mode S slots HBM_GiB budget_GB
  fail=0
  for bk in $BACKENDS; do
    for S in $S_LIST; do
      B=$(( S * ${BPS[$bk:dense]} ))
      prev=999999
      for md in dense padded ragged; do
        slots=$(calc_slots "$bk" "$md" "$S")
        hbm=$(python3 -c "print(f'{$B/2**30:.2f}')")
        gb=$(python3 -c "print(f'{$B/1e9:.3f}')")
        printf "%-4s %-7s %-4d %8d %10s %10s\n" "$bk" "$md" "$S" "$slots" "$hbm" "$gb"
        # invariant: slots non-decreasing dense<=padded<=ragged (compression only shrinks bytes/slot)
        [ "$slots" -lt "$prev" ] && [ "$md" != dense ] && { log "  WARN $bk S=$S $md slots=$slots < prev=$prev (non-monotone)"; fail=1; }
        prev=$slots
        # invariant: dense starves (< distinct 184)
        [ "$md" = dense ] && [ "$slots" -ge 184 ] && { log "  WARN $bk S=$S dense slots=$slots >= distinct 184 (not starving)"; fail=1; }
      done
    done
  done
  echo
  [ "$fail" -eq 0 ] && echo "EQ-GATE-0: PASS (dense<distinct, slots monotone, budget consistent per S)" \
                     || echo "EQ-GATE-0: WARN (see above)"
  exit 0
fi

# ---- full sweep ----
export NGROUPS LOGDIR

wait_gpu_free(){  # poll until every visible GPU has >= FREE_GB free
  local need=$(( FREE_GB * 1024 ))
  for _ in $(seq 1 120); do   # up to ~20 min
    local minfree
    minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
              | sort -n | head -1)
    [ -z "$minfree" ] && { sleep 5; continue; }
    if [ "$minfree" -ge "$need" ]; then
      log "  gpu free ok (min ${minfree}MiB >= ${need}MiB)"; return 0
    fi
    sleep 10
  done
  log "  WARN: gpu never freed to ${need}MiB; proceeding anyway"; return 0
}

log "=== EQUAL-HBM sweep START  N=$NGROUPS backends=[$BACKENDS] S=[$S_LIST] seeds=$SEEDS free_gb=$FREE_GB ==="

# ensure dataset exists (build if missing)
DS="/tmp/arena_shared_prefix_N${NGROUPS}.jsonl"
if [ ! -f "$DS" ]; then
  log "building dataset N=$NGROUPS ..."
  no_proxy='*' python experiments/scripts/build_arena_shared_prefix.py --n-groups "$NGROUPS" --out "$DS" \
    >>"$ORCH" 2>&1 || { log "FATAL dataset build failed"; exit 1; }
fi

i=0; TOTAL=0

# ---- Phase 0: live BPS calibration (1 seed per non-dense mode) ----
FIRST_S="${S_LIST%% *}"   # calibrate using the first S value
log "=== Phase 0: live BPS calibration (1 seed per mode, S=$FIRST_S) ==="
for bk in $BACKENDS; do
  for md in dense padded ragged; do
    calibrate_bps "$bk" "$md" "$FIRST_S"
  done
done
log "  calibration done. BPS table:"
for bk in $BACKENDS; do
  for md in dense padded ragged; do
    log "    $bk:$md bps=${BPS[$bk:$md]}"
  done
done
for bk in $BACKENDS; do for S in $S_LIST; do for md in dense padded ragged; do TOTAL=$((TOTAL+1)); done; done; done

for bk in $BACKENDS; do
  for S in $S_LIST; do
    B=$(( S * ${BPS[$bk:dense]} ))
    for md in dense padded ragged; do
      i=$((i+1))
      slots=$(calc_slots "$bk" "$md" "$S")
      hbm=$(python3 -c "print(f'{$B/2**30:.2f}')")
      log "--- launch $i/$TOTAL: $bk $md S=$S -> ckpt=$slots budget=${hbm}GiB ---"
      wait_gpu_free
      ARMLOG="$LOGDIR/arm_${bk}_${md}_N${NGROUPS}_S${S}.log"
      CKPT_OVERRIDE="$slots" MEASURE_SEEDS="$SEEDS" LOGDIR="$LOGDIR" NGROUPS="$NGROUPS" \
        bash experiments/scripts/bench_arena_shared_prefix.sh "$bk" "$md" >"$ARMLOG" 2>&1
      rc=$?
      # per-pass result files: result_<bk>_<md>_N<ng>_w<w>_ckpt<slots>_seed<seed>.json
      RES=$(ls -t "$LOGDIR"/result_${bk}_${md}_N${NGROUPS}_*_ckpt${slots}_*_seed*.json 2>/dev/null | head -1)
      if [ $rc -ne 0 ] || [ -z "$RES" ]; then
        log "  launch $bk/$md S=$S FAILED (rc=$rc, result=${RES:-none}); tail:"
        tail -30 "$ARMLOG" | tee -a "$ORCH"
        if [ $i -eq 1 ]; then
          log "  FIRST launch (GATE-1 smoke) failed -> ABORT sweep."; exit 1
        fi
        log "  continuing to next launch."
      else
        log "  launch $bk/$md S=$S OK -> $(ls "$LOGDIR"/result_${bk}_${md}_N${NGROUPS}_*_ckpt${slots}_*_seed*.json 2>/dev/null | wc -l) seed JSONs"
        grep -E "^\[result\]" "$ARMLOG" | tail -3 | tee -a "$ORCH" || true
        # P0 cross-check: live bps * slots <= budget (floor guarantee)
        python3 - "$RES" "$B" "$bk" "$md" "$S" "$slots" <<'PY'
import json, sys
res, budget, bk, md, S, slots = sys.argv[1:7]
budget = int(budget); S = int(S); slots = int(slots)
d = json.load(open(res))
live_bps = d.get("_bytes_per_slot", 0)
used = live_bps * slots
slack = budget - used
rel = abs(used - budget) / budget if budget else 0
status = "OK" if rel < 0.02 else f"DRIFT {rel*100:.1f}%"
print(f"  [xcheck] {bk}/{md} S={S}: live_bps={live_bps} slots={slots} used={used} budget={budget} slack={slack} -> {status}")
PY
      fi
    done
  done
done

log "=== sweep DONE; running plot_equal_hbm.py ==="
python experiments/scripts/plot_equal_hbm.py --logdir "$LOGDIR" 2>&1 | tee -a "$ORCH" || \
  log "  plot failed (run manually: python experiments/scripts/plot_equal_hbm.py)"
log "=== ALL DONE ==="
