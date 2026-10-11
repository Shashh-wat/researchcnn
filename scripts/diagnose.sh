#!/usr/bin/env bash
# One-shot diagnosis of the dental run. Read-only apart from a short GPU probe.
#   bash scripts/diagnose.sh        (output is also saved to $OUT/diagnose.txt)
# Env overrides: WORK, DATA, OUT, PY, PROBE_B, PROBE_K (probe batch / draws, default 8 / 8).
set -u
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-$HOME/pjt_ak/researchcnn}"
export DATA="${DATA:-$WORK/data_dental}"
OUT="${OUT:-$WORK/pipeline_out}"
PY="${PY:-/root/myenv/bin/python3}"
mkdir -p "$OUT"

section() { echo; echo "=================== $*"; }

{
section "1. RUN STATE"
echo "STATUS file: $(cat "$OUT/STATUS" 2>/dev/null || echo '(none)')"
echo "train.py processes alive: $(pgrep -f 'scripts/train.py' | wc -l)"
echo "--- error lines in run.log"
grep -a -n "Traceback\|Error\|Killed\|out of memory\|OutOfMemory" "$OUT/run.log" 2>/dev/null | tail -8 | cut -c1-220
echo "--- last 30 lines of run.log"
tail -n 30 "$OUT/run.log" 2>/dev/null | cut -c1-220
echo "--- repo version"
git -C "$REPO" log --oneline -3 2>&1 | head -3
git -C "$REPO" status --short 2>&1 | head -5

section "2. MACHINE: GPU, memory, shared memory, disk"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | head -4
echo "--- shared memory and disk"; df -h /dev/shm / 2>&1 | cat
echo "--- RAM"; free -g 2>&1 | head -2

section "3. DATA"
"$PY" - <<'EOF'
import csv, os
from PIL import Image
D = os.environ["DATA"]
first = {}
for split in ("train", "val", "test"):
    f = f"{D}/real_{split}.csv"
    if not os.path.exists(f):
        print(f"real_{split}.csv: MISSING"); continue
    rows = list(csv.DictReader(open(f)))
    labs = [int(r["label"]) for r in rows]
    missing = [r["path"] for r in rows if not os.path.exists(r["path"])]
    first[split] = rows[0]["path"] if rows else None
    print(f"real_{split}.csv: {len(rows)} rows, label1={sum(labs)}, label0={len(labs)-sum(labs)}, missing files={len(missing)}")
for p in ("synth_train", "synth_val", "synth_eval"):
    d = f"{D}/{p}"
    names = sorted(os.listdir(d)) if os.path.isdir(d) else None
    print(f"{p}: {'MISSING' if names is None else str(len(names)) + ' files'}")
    if names:
        im = Image.open(f"{d}/{names[0]}"); print(f"   sample {names[0]}: size={im.size} mode={im.mode}")
if first.get("train"):
    im = Image.open(first["train"]); print(f"real sample: size={im.size} mode={im.mode}")
EOF

section "4. PYTHON ENVIRONMENT AND GPU PROBE (runs the real model: 2 training steps)"
echo "python: $PY"
timeout 900 "$PY" - <<'EOF'
import os, sys, time, platform
sys.path.insert(0, os.environ["REPO"])
import torch, torchvision, timm
print("python", platform.python_version(), "| torch", torch.__version__, "| torchvision", torchvision.__version__, "| timm", timm.__version__)
print("cuda available:", torch.cuda.is_available(), "| visible devices:", torch.cuda.device_count())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    free, total = torch.cuda.mem_get_info()
    print(f"gpu: {torch.cuda.get_device_name(0)} | free {free/1e9:.1f} GB of {total/1e9:.1f} GB")
from src.config import Config
from src.losses import CombinedLoss
from src.model import MembershipFingerprintNet
B = int(os.environ.get("PROBE_B", 8)); K = int(os.environ.get("PROBE_K", 8))
cfg = Config(pretrained_backbones=False, use_grad_checkpointing=True, k_draws=K)
model = MembershipFingerprintNet(cfg).to(dev).train()
loss_fn = CombinedLoss(cfg.lambda_bce, cfg.lambda_arcface, cfg.lambda_supcon, cfg.supcon_temperature)
labels = torch.tensor([0, 1] * (B // 2), device=dev)
x = torch.randn(B, 3, 224, 224, device=dev)
y = torch.randn(B, K, 3, 224, 224, device=dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
times = []
for _ in range(2):
    if dev.type == "cuda": torch.cuda.synchronize()
    t = time.time()
    out = model(x, y, labels)
    losses = loss_fn(out, labels)
    opt.zero_grad(); losses["total"].backward(); opt.step()
    if dev.type == "cuda": torch.cuda.synchronize()
    times.append(time.time() - t)
print(f"probe (batch={B}, K={K}, fp32, no pretrained weights): step times {[round(t, 1) for t in times]} s")
if dev.type == "cuda":
    print(f"peak GPU memory during probe: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("loss finite:", bool(torch.isfinite(losses['total'])))
steps_per_epoch = 408 // B
print(f"ROUGH ESTIMATE: ~{steps_per_epoch} steps/epoch x {times[-1]:.1f}s = {steps_per_epoch*times[-1]/60:.0f} min/epoch (compute only; the data loader adds more)")
EOF
echo "probe exit code: $?  (124 = timed out after 15 min)"
} 2>&1 | cut -c1-230 | tee "$OUT/diagnose.txt"

echo
echo "Saved to $OUT/diagnose.txt"
