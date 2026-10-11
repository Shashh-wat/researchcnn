#!/usr/bin/env bash
# Dental membership-inference run, one command.
#   bash scripts/run_dental.sh check   preflight only: PASS/FAIL per item, changes nothing
#   bash scripts/run_dental.sh         full run: pools -> detector -> evaluate -> calibrate
# Finished stages are skipped on re-run. Progress: cat "$OUT/STATUS".
set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-$HOME/pjt_ak/researchcnn}"
DATA="${DATA:-$WORK/data_dental}"
RUNS="${RUNS:-$WORK/runs_dental}"
OUT="${OUT:-$WORK/pipeline_out}"
PY="${PY:-/root/myenv/bin/python3}"
EPOCHS="${EPOCHS:-30}"
SG="${SG:-$(dirname "$(find "$HOME" /data -maxdepth 4 -name generate.py -path '*stylegan2-ada*' 2>/dev/null | head -1)")}"
PKL="$(ls -t "$RUNS"/*/network-snapshot-*.pkl 2>/dev/null | head -1)"

# Containers often have a tiny /dev/shm, which crashes multi-worker data loading.
SHM_MB="$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $4}')"
if [ "${SHM_MB:-0}" -ge 2000 ]; then WORKERS=2; else WORKERS=0; fi

mkdir -p "$OUT"
STATUS="$OUT/STATUS"
say() { echo "[$(date +%H:%M:%S)] $*"; }
set_status() { echo "$(date '+%F %H:%M:%S') $*" > "$STATUS"; say "$*"; }
fail() { set_status "FAILED: $*"; exit 1; }

PASS=1
chk() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "PASS  $desc"; else echo "FAIL  $desc"; PASS=0; fi
}
weights_ok() {
  "$PY" -c "import torchvision, timm; torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2); timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)"
}

preflight() {
  chk "python ($PY) has torch, torchvision, timm, sklearn and sees a GPU" \
    "$PY" -c "import sys, torch, torchvision, timm, sklearn; sys.exit(0 if torch.cuda.is_available() else 1)"
  chk "manifests present in $DATA (real_train/val/test.csv)" \
    test -f "$DATA/real_train.csv" -a -f "$DATA/real_val.csv" -a -f "$DATA/real_test.csv"
  chk "a GAN snapshot exists under $RUNS" test -n "$PKL"
  chk "StyleGAN generate.py found in $SG" test -f "$SG/generate.py"
  chk "scripts/train.py in $REPO is current (has --warmup-steps)" grep -q "warmup-steps" "$REPO/scripts/train.py"
  if [ "${NO_PRETRAINED:-0}" != 1 ]; then
    chk "ImageNet weights download (resnet50 + ViT)" weights_ok
  fi
  echo "INFO  /dev/shm free: ${SHM_MB:-unknown} MB -> data loader workers: $WORKERS"
  echo "INFO  snapshot: ${PKL:-none}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/INFO  gpu /'
}

preflight
if [ "${1:-}" = "check" ]; then
  if [ "$PASS" = 1 ]; then
    echo "ALL CHECKS PASSED. Start the run with:"
    echo "  nohup bash $REPO/scripts/run_dental.sh > \"$OUT/run.log\" 2>&1 &"
  else
    echo "Some checks FAILED (see FAIL lines). If the weights check failed, set HF_TOKEN or rerun with NO_PRETRAINED=1."
  fi
  exit 0
fi
[ "$PASS" = 1 ] || fail "preflight failed; run: bash scripts/run_dental.sh check"

make_pool() { # name seeds wanted
  local dir="$DATA/$1" have
  have="$(ls "$dir" 2>/dev/null | wc -l)"
  if [ "$have" -ge "$3" ]; then say "pool $1 already complete ($have images), skipping"; return 0; fi
  set_status "generating pool $1 ($3 images)"
  case "$dir" in */synth_*) rm -rf "$dir" ;; esac
  # StyleGAN prints long, harmless fallback warnings; keep them in a log, not on screen.
  ( cd "$SG" && "$PY" generate.py --outdir="$dir" --seeds="$2" --trunc=1 --network="$PKL" \
      > "$OUT/generate_$1.log" 2>&1 ) \
    || { tail -n 15 "$OUT/generate_$1.log"; fail "generate.py failed for pool $1 (log: $OUT/generate_$1.log)"; }
}
make_pool synth_train 0-1999 2000
make_pool synth_val 10000-10499 500
make_pool synth_eval 20000-20999 1000

# The GPU is shared, so find the largest batch / draw count that actually fits right now.
# Exit code 3 from the probe means "does not fit"; any other failure is a real error.
pick_fit() {
  if [ -n "${BATCH:-}" ] && [ -n "${KD:-}" ]; then return 0; fi
  set_status "probing GPU memory for the largest batch that fits"
  local pair b k rc
  for pair in "8 8" "6 8" "4 8" "4 6" "4 4"; do
    read -r b k <<< "$pair"
    "$PY" "$REPO/scripts/probe_fit.py" "$b" "$k" > "$OUT/probe_${b}_${k}.log" 2>&1
    rc=$?
    say "probe batch=$b K=$k: $(grep -v -i warn "$OUT/probe_${b}_${k}.log" | tail -1)"
    if [ "$rc" -eq 0 ]; then BATCH="$b"; KD="$k"; return 0; fi
    if [ "$rc" -ne 3 ]; then tail -n 15 "$OUT/probe_${b}_${k}.log"; fail "GPU probe crashed (not a memory problem), see $OUT/probe_${b}_${k}.log"; fi
  done
  return 1
}

# On a busy shared GPU, keep retrying until WAIT_HOURS have passed (default 0 = try once).
fit_with_wait() {
  local deadline=$(( $(date +%s) + ${WAIT_HOURS:-0} * 3600 ))
  while true; do
    pick_fit && return 0
    [ "$(date +%s)" -ge "$deadline" ] && return 1
    set_status "waiting for free GPU memory (retrying every ${WAIT_SECS:-300}s)"
    sleep "${WAIT_SECS:-300}"
  done
}

if [ -f "$OUT/train.done" ]; then
  say "detector already trained, skipping (delete $OUT/train.done to retrain)"
else
  fit_with_wait || fail "not even batch=4 K=4 fits in free GPU memory right now; the card is too busy (rerun with WAIT_HOURS=12 to wait for a gap)"
  set_status "training detector ($EPOCHS epochs, batch=$BATCH, K=$KD)"
  PRE=""; [ "${NO_PRETRAINED:-0}" = 1 ] && PRE="--no-pretrained"
  ( cd "$REPO" && "$PY" scripts/train.py \
      --train-manifest "$DATA/real_train.csv" --train-synth-dir "$DATA/synth_train" \
      --val-manifest "$DATA/real_val.csv" --val-synth-dir "$DATA/synth_val" \
      --batch-size "$BATCH" --k-draws "$KD" --epochs "$EPOCHS" --warmup-steps 30 \
      --no-amp --grad-checkpointing --num-workers "$WORKERS" \
      --ckpt-dir "$OUT/checkpoints" $PRE ) || fail "detector training failed"
  touch "$OUT/train.done"
fi

set_status "evaluating on calibration and test splits"
cd "$REPO" || fail "cannot cd to $REPO"
"$PY" scripts/evaluate.py --checkpoint "$OUT/checkpoints/best.pt" \
  --manifest "$DATA/real_val.csv" --synth-dir "$DATA/synth_eval" \
  --out-npz "$OUT/cal.npz" --out-metrics "$OUT/cal_metrics.json" --num-workers "$WORKERS" \
  || fail "evaluation on the calibration split failed"
"$PY" scripts/evaluate.py --checkpoint "$OUT/checkpoints/best.pt" \
  --manifest "$DATA/real_test.csv" --synth-dir "$DATA/synth_eval" \
  --out-npz "$OUT/test.npz" --out-metrics "$OUT/test_metrics.json" --num-workers "$WORKERS" \
  || fail "evaluation on the test split failed"
"$PY" scripts/calibrate.py --cal-npz "$OUT/cal.npz" --test-npz "$OUT/test.npz" \
  --alpha 0.1 --out-json "$OUT/conformal.json" || fail "conformal calibration failed"

"$PY" - "$OUT" <<'EOF' | tee "$OUT/SUMMARY.txt"
import json, sys
o = sys.argv[1]
t = json.load(open(f"{o}/test_metrics.json"))["metrics"]
c = json.load(open(f"{o}/conformal.json"))
print("TEST (real_test.csv against synthetic pool synth_eval)")
for k in ("accuracy", "precision", "recall", "f1", "auc"):
    print(f"  {k}: {t[k]:.3f}")
print("  confusion:", t["confusion_matrix"])
print(f"CONFORMAL: coverage {c['empirical_coverage']:.3f} (target {c['target_coverage']:.2f}), abstain {c['abstain_rate']:.3f}")
print("Raw outputs are in", o, "and were not edited by hand.")
EOF
set_status "DONE"
