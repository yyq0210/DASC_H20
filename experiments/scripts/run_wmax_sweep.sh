#!/usr/bin/env bash
# W_MAX sweep serving experiment (KDA, ShareGPT-V3 shared-prefix).
#
# Decomposes the DASC benefit into two orthogonal axes:
#   Group 1 (fixed slots):   paired ΔTTFT(W) = TTFT_WR(W) − TTFT_NR(W).
#                            Same slot count across W; HBM varies (isolates recon
#                            cost, NOT compression benefit).
#   Group 2 (NR, fixed HBM): more slots from smaller bytes/slot -> higher hit
#                            rate -> better TTFT/throughput. NO recon (NR).
#   Group 3 (WR, fixed HBM): same as Group 2 but WITH recon. Paired with Group 2
#                            (same W, slots, HBM, request order; only NR->WR).
#                            net = WR(W)−Dense; recon overhead = WR(W)−NR(W).
#
# P0 fixes vs the original plan:
#   - PILOT first (P0-1): N40 saturates at W=16 for S>=66 (KDA compresses 2.77x).
#     Default pilot tests N=200 at S in {24,32,48} to find a 3-zone rising curve
#     (W=16/64/128 all unsaturated). --pilot prints a recommendation.
#   - FLOOR not round (P0-2): CKPT_W = floor(budget / bps_ragged(W)); never
#     exceeds the HBM budget. dry-run warns if max slots > 10000.
#   - reset_peak_memory (P0-3): harness calls /reset_peak_memory after warm before
#     each measure seed so peak_allocated = measure-window peak (rank-0 lower
#     bound on recon scratch; NOT matched total HBM — that claim is slots*bps).
#   - replay counters (P0-4): harness injects _recon_n_hits / _replay_tokens_*;
#     actual replay = min(W_max, P) per hit (counter records the real value).
#
# Usage:
#   bash experiments/scripts/run_wmax_sweep.sh --dry-run        # 21-row table + invariants
#   bash experiments/scripts/run_wmax_sweep.sh --pilot          # choose N, S
#   bash experiments/scripts/run_wmax_sweep.sh                  # full 21-launch sweep
# Env:
#   NGROUPS=200  SLOTS=32  W_LIST="16 64 128 512 1024"
#   SEEDS="42,123,7,555,31337"  FREE_GB=40  EVICT=lru  MAX_CONC=16  OUTPUT_LEN=64
#   PILOT_N_LIST PILOT_S_LIST PILOT_W_LIST PILOT_SEEDS  (pilot overrides)
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

MODE="full"
DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1;;
  --pilot)   MODE="pilot";;
  --full)    MODE="full";;
  "")        ;;
  *) echo "usage: $0 [--dry-run|--pilot|--full]"; exit 2;;
esac

NGROUPS="${NGROUPS:-200}"
SLOTS="${SLOTS:-32}"                 # dense-equivalent slot count (pilot-chosen)
W_LIST="${W_LIST:-16 64 128 512 1024}"
SEEDS="${SEEDS:-42,123,7,555,31337}"
FREE_GB="${FREE_GB:-40}"
EVICT="${EVICT:-lru}"
MAX_CONC="${MAX_CONC:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
LOGDIR="${LOGDIR:-$PWD/experiments/results/wmax_sweep}"
mkdir -p "$LOGDIR"
export LOGDIR
STAMP="$(date +%Y%m%d_%H%M%S)"
ORCH="$LOGDIR/orchestrator_wmax_${MODE}_${STAMP}.log"
BPS_JSON="$LOGDIR/bps_wmax_table.json"
MANIFEST="$LOGDIR/manifest_${STAMP}.json"

# ---- Phase 0: BPS table (compute_bps_wmax.py, GPU-free) ----
if [ ! -f "$BPS_JSON" ]; then
  echo "[phase0] computing BPS table (GPU-free) ..."
  python experiments/scripts/compute_bps_wmax.py --out "$BPS_JSON" >"$(mktemp)" 2>&1 || {
    echo "[fatal] compute_bps_wmax.py failed; run it manually"; exit 1; }
fi
[ -f "$BPS_JSON" ] || { echo "[fatal] $BPS_JSON missing"; exit 1; }

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }

# bps_ragged(W): read from the offline table. echoes bytes_per_slot_ragged.
bps_ragged(){ python3 -c "
import json,sys
d=json.load(open('$BPS_JSON'))
print(d['table'][str($1)]['bytes_per_slot_ragged'])
"; }

dense_bps(){ python3 -c "
import json
d=json.load(open('$BPS_JSON'))
print(d['dense_bps'])
"; }

# ---- dry-run: 21-row table + invariants (no server launch) ----
if [ "$DRY_RUN" -eq 1 ]; then
  DB=$(dense_bps)
  BUDGET=$(( SLOTS * DB ))
  echo "=== W_MAX sweep dry-run: N=$NGROUPS S=$SLOTS dense_bps=$DB budget=$BUDGET B ==="
  echo "=== Group 1 (fixed slots=$SLOTS; HBM varies — isolates recon cost) ==="
  printf "%-6s %-5s %-7s %10s %12s\n" group W recon slots ckpt_HBM_GiB
  fail=0
  for W in $W_LIST; do
    BPS=$(bps_ragged "$W")
    HBM=$(( SLOTS * BPS ))
    HBM_GIB=$(python3 -c "print(f'{$HBM/2**30:.2f}')")
    for RECON in 0 1; do
      printf "%-6s %-5s %-7s %10s %12s\n" G1 "$W" "recon$RECON" "$SLOTS" "${HBM_GIB}GiB"
    done
  done
  echo
  echo "=== Group 2+3 (fixed HBM = $SLOTS dense slots = $BUDGET B = $(python3 -c "print(f'{$BUDGET/2**30:.2f}')")GiB) ==="
  printf "%-6s %-5s %-7s %10s %12s %10s\n" group W recon slots ckpt_HBM_GiB slack_B
  # dense baseline (shared by Groups 2+3):
  DHBM=$(( SLOTS * DB ))
  printf "%-6s %-5s %-7s %10s %12s %10s\n" G23 dense - "$SLOTS" "$(python3 -c "print(f'{$DHBM/2**30:.2f}')")GiB" 0
  max_slots=0
  for W in $W_LIST; do
    BPS=$(bps_ragged "$W")
    CKPT_W=$(( BUDGET / BPS ))   # FLOOR (never exceeds budget)
    SLACK=$(( BUDGET - CKPT_W * BPS ))
    HBM_GIB=$(python3 -c "print(f'{$CKPT_W*$BPS/2**30:.2f}')")
    for RECON in 0 1; do
      printf "%-6s %-5s %-7s %10s %12s %10s\n" G23 "$W" "recon$RECON" "$CKPT_W" "${HBM_GIB}GiB" "$SLACK"
    done
    [ "$CKPT_W" -gt "$max_slots" ] && max_slots=$CKPT_W
  done
  echo
  # invariants
  [ "$max_slots" -gt 10000 ] && { echo "[WARN] max CKPT_W=$max_slots > 10000 (slot metadata may be non-negligible; W=512/1024 produce tens of thousands)"; fail=1; }
  python3 -c "
import json
d=json.load(open('$BPS_JSON'))
bps16=d['table']['16']['bytes_per_slot_ragged']
db=d['dense_bps']
distinct=$NGROUPS*46//10  # ~4.6 distinct prefixes per N40 group (estimate)
s16=($SLOTS * db)//bps16
print(f'[invariant] W=16 ragged slots at S=$SLOTS: {s16} vs distinct~{distinct} -> '+('SATURATED' if s16>=distinct else 'unsaturated'))
" || true
  [ "$fail" -eq 0 ] && echo "GATE-0 dry-run: PASS (floor slots <= budget, slack reported)" || echo "GATE-0 dry-run: WARN (see above)"
  exit 0
fi

export NGROUPS LOGDIR EVICT MAX_CONC OUTPUT_LEN

wait_gpu_free(){
  local need=$(( FREE_GB * 1024 ))
  for _ in $(seq 1 120); do
    local minfree
    minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
    [ -z "$minfree" ] && { sleep 5; continue; }
    [ "$minfree" -ge "$need" ] && { log "  gpu free ok (min ${minfree}MiB >= ${need}MiB)"; return 0; }
    sleep 10
  done
  log "  WARN: gpu never freed to ${need}MiB; proceeding anyway"; return 0
}

# ensure dataset exists (build if missing)
ensure_dataset(){
  local n="$1"
  local ds="/tmp/arena_shared_prefix_N${n}.jsonl"
  if [ ! -f "$ds" ]; then
    log "building dataset N=$n ..."
    no_proxy='*' python experiments/scripts/build_arena_shared_prefix.py --n-groups "$n" --out "$ds" >>"$ORCH" 2>&1 \
      || { log "FATAL dataset build failed for N=$n"; return 1; }
  fi
}

# run one arm; returns 0 on success. $1=group $2=W $3=recon $4=ckpt $5=mode(dense/ragged)
launch_arm(){
  local grp="$1" W="$2" recon="$3" ckpt="$4" md="$5"
  local label="N${NGROUPS}_w${W}_ckpt${ckpt}_recon${recon}"
  local armlog="$LOGDIR/arm_${grp}_${label}.log"
  log "--- $grp: W=$W recon=$recon ckpt=$ckpt mode=$md ---"
  # clean stale result files (serving benchmark APPENDS to --output-file, so
  # leftover files from a prior run get concatenated -> JSONDecodeError)
  rm -f "$LOGDIR"/result_kda_${md}_${label}_seed*.json 2>/dev/null
  wait_gpu_free
  WMAX_DASC="$W" RECON="$recon" CKPT_OVERRIDE="$ckpt" MEASURE_SEEDS="$SEEDS" \
    NGROUPS="$NGROUPS" LOGDIR="$LOGDIR" EVICT="$EVICT" MAX_CONC="$MAX_CONC" \
    OUTPUT_LEN="$OUTPUT_LEN" \
    bash experiments/scripts/bench_arena_shared_prefix.sh kda "$md" >"$armlog" 2>&1
  local rc=$?
  # collect seed JSONs
  local njson
  njson=$(ls "$LOGDIR"/result_kda_${md}_${label}_seed*.json 2>/dev/null | wc -l)
  if [ $rc -ne 0 ] || [ "$njson" -eq 0 ]; then
    log "  $grp W=$W recon=$recon FAILED (rc=$rc, seed_jsons=$njson); tail:"
    tail -25 "$armlog" | tee -a "$ORCH"
    python3 -c "import json;m=json.load(open('$MANIFEST')) if __import__('os').path.exists('$MANIFEST') else {'arms':[]};m.setdefault('arms',[]).append({'group':'$grp','W':$W,'recon':$recon,'ckpt':$ckpt,'mode':'$md','status':'FAIL','seed_jsons':$njson});json.dump(m,open('$MANIFEST','w'),indent=2)" || true
    return 1
  fi
  # P0-2: verify floor budget (live-measured bps * ckpt <= budget) for G23
  local budget_slack="n/a"
  if [ "$grp" = "G23" ] && [ "$md" != "dense" ]; then
    budget_slack=$(python3 -c "
import json,glob
fs=sorted(glob.glob('$LOGDIR/result_kda_${md}_${label}_seed*.json'))
d=json.load(open(fs[0]))
bps=d.get('_bytes_per_slot') or 0
ckpt=d.get('_ckpt_slots') or 0
budget=$SLOTS*$(dense_bps)
slack=budget-bps*ckpt
print(f'slots={ckpt} bps={bps} used={bps*ckpt} budget={budget} slack={slack} ({\"OK\" if slack>=0 else \"OVER-BUDGET\"})')
" 2>/dev/null || echo "n/a")
  fi
  log "  $grp W=$W recon=$recon OK ($njson seed JSONs); $budget_slack"
  grep -E "^\[result\]" "$armlog" | tail -1 | tee -a "$ORCH" || true
  python3 -c "import json,os;m=json.load(open('$MANIFEST')) if os.path.exists('$MANIFEST') else {'arms':[]};m.setdefault('arms',[]).append({'group':'$grp','W':$W,'recon':$recon,'ckpt':$ckpt,'mode':'$md','status':'OK','seed_jsons':$njson});json.dump(m,open('$MANIFEST','w'),indent=2)" || true
  return 0
}

# ===================== PILOT =====================
if [ "$MODE" = "pilot" ]; then
  PILOT_N_LIST="${PILOT_N_LIST:-200}"
  PILOT_S_LIST="${PILOT_S_LIST:-24 32 48}"
  PILOT_W_LIST="${PILOT_W_LIST:-16 64 128}"
  PILOT_SEEDS="${PILOT_SEEDS:-42}"
  log "=== PILOT: N=[$PILOT_N_LIST] S=[$PILOT_S_LIST] W=[$PILOT_W_LIST] seeds=$PILOT_SEEDS ==="
  for n in $PILOT_N_LIST; do ensure_dataset "$n" || exit 1; done
  i=0
  for S in $PILOT_S_LIST; do
    for n in $PILOT_N_LIST; do
      DB=$(dense_bps); BUDGET=$(( S * DB ))
      NGROUPS="$n" SLOTS="$S"  # for launch_arm label/manifest
      export NGROUPS
      # dense baseline
      i=$((i+1)); log "--- pilot $i: N=$n S=$S dense ---"
      WMAX_DASC=0 RECON=0 CKPT_OVERRIDE="$S" MEASURE_SEEDS="$PILOT_SEEDS" \
        NGROUPS="$n" LOGDIR="$LOGDIR" bash experiments/scripts/bench_arena_shared_prefix.sh kda dense \
        >"$LOGDIR/pilot_N${n}_S${S}_dense.log" 2>&1 || log "  pilot dense FAILED"
      for W in $PILOT_W_LIST; do
        i=$((i+1)); log "--- pilot $i: N=$n S=$S W=$W ---"
        BPS=$(bps_ragged "$W"); CKPT_W=$(( BUDGET / BPS ))
        WMAX_DASC="$W" RECON=0 CKPT_OVERRIDE="$CKPT_W" MEASURE_SEEDS="$PILOT_SEEDS" \
          NGROUPS="$n" LOGDIR="$LOGDIR" bash experiments/scripts/bench_arena_shared_prefix.sh kda ragged \
          >"$LOGDIR/pilot_N${n}_S${S}_w${W}.log" 2>&1 || log "  pilot W=$W FAILED"
      done
    done
  done
  log "=== pilot DONE; hit-rate table: ==="
  python3 - <<'PY'
import json,glob,os
print(f"{'N':>4}{'S':>4}{'arm':>10}{'hit_rate':>10}{'ttft':>8}{'slots':>7}")
for f in sorted(glob.glob(os.path.join(os.environ["LOGDIR"], "result_kda_*_seed*.json"))):
    try: d=json.load(open(f))
    except: continue
    n=d.get("_ngroups"); w=d.get("_wmax"); md=d.get("_dasc_mode"); s=d.get("_ckpt_slots")
    if n is None: continue
    hit=d.get("_token_hit_rate"); ttft=d.get("mean_ttft_ms")
    arm = "dense" if md=="dense" else f"w{w}"
    print(f"{n:>4}{s:>4}{arm:>10}{(f'{hit:.3f}' if hit is not None else 'NA'):>10}{(f'{ttft:.0f}' if ttft else 'NA'):>8}{s:>7}")
PY
  log "=== pick the largest S where W=128 hit < 0.95 (unsaturated) AND W=16 hit < 0.9; rerun: NGROUPS=<N> SLOTS=<S> bash $0 --full ==="
  exit 0
fi

# ===================== FULL SWEEP =====================
log "=== W_MAX sweep START  N=$NGROUPS S=$SLOTS W=[$W_LIST] seeds=$SEEDS free_gb=$FREE_GB ==="
ensure_dataset "$NGROUPS" || exit 1
DB=$(dense_bps); BUDGET=$(( SLOTS * DB ))
log "  dense_bps=$DB budget(G23)=$BUDGET B = $(python3 -c "print(f'{$BUDGET/2**30:.2f}')")GiB"
log "  claim: matched persistent checkpoint-pool HBM (slots*bps); peak_allocated is rank-0 post-warm lower bound on recon scratch"

i=0; TOTAL=0
for W in $W_LIST; do for r in 0 1; do TOTAL=$((TOTAL+1)); done; done   # G1: 10
TOTAL=$((TOTAL+1))   # G23 dense
for W in $W_LIST; do for r in 0 1; do TOTAL=$((TOTAL+1)); done; done  # G23 DASC: 10

# ---- Phase 1: Group 1 (fixed slots=S; HBM varies) ----
log "=== Phase 1: Group 1 (fixed slots=$SLOTS; isolates recon cost) ==="
for W in $W_LIST; do
  for RECON in 0 1; do
    i=$((i+1))
    log "--- launch $i/$TOTAL: G1 W=$W recon=$RECON ckpt=$SLOTS (fixed slots) ---"
    launch_arm G1 "$W" "$RECON" "$SLOTS" ragged
    [ $i -eq 1 ] && { ! ls "$LOGDIR"/result_kda_ragged_N${NGROUPS}_w${W}_ckpt${SLOTS}_recon0_seed*.json >/dev/null 2>&1 && { log "  FIRST launch (GATE-1 smoke) failed -> ABORT"; exit 1; }; }
  done
done

# ---- Phase 2: Groups 2+3 (fixed HBM = S*dense_bps) ----
log "=== Phase 2: Groups 2+3 (fixed HBM = $BUDGET B) ==="
# dense baseline (shared)
i=$((i+1)); log "--- launch $i/$TOTAL: G23 dense ckpt=$SLOTS ---"
launch_arm G23 0 0 "$SLOTS" dense
for W in $W_LIST; do
  BPS=$(bps_ragged "$W")
  CKPT_W=$(( BUDGET / BPS ))   # FLOOR
  SLACK=$(( BUDGET - CKPT_W * BPS ))
  log "  W=$W: bps_ragged=$BPS -> CKPT_W=$CKPT_W (slack=$SLACK B, $(python3 -c "print(f'{$SLACK/2**30:.3f}')")GiB)"
  for RECON in 0 1; do
    i=$((i+1))
    log "--- launch $i/$TOTAL: G$([ $RECON -eq 0 ] && echo 2 || echo 3) W=$W recon=$RECON ckpt=$CKPT_W ---"
    launch_arm G23 "$W" "$RECON" "$CKPT_W" ragged
  done
done

log "=== sweep DONE; running plot_wmax_sweep.py ==="
python experiments/scripts/plot_wmax_sweep.py --logdir "$LOGDIR" 2>&1 | tee -a "$ORCH" || \
  log "  plot failed (run manually: python experiments/scripts/plot_wmax_sweep.py)"
log "=== ALL DONE ==="
