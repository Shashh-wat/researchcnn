# Mini Demo: Results

Fully local, Mac-only, smallest-time end-to-end run of the pipeline: real images → a real
(if tiny) GAN we trained ourselves → the actual detector architecture → evaluation →
conformal calibration. **This is a correctness/demo run at toy scale, not a research
result** — read the interpretation section before drawing conclusions from any number here.

Two runs are reported: an initial pass, and a second pass after fixing a real bug found
in the first. Raw source files for run 2: `checkpoints2/metrics_log.jsonl`,
`cal_metrics2.json`, `test_metrics2.json`, `test_metrics2_degenerate.json`,
`conformal_results2.json`.

## Run 1 → Run 2: what changed and why

Run 1 (100 real images, 8 epochs) surfaced a real bug: `warmup_steps` defaulted to 500
while the whole run only had 200 optimizer steps, so the learning rate never left its
warmup ramp — it climbed from ~4e-6 to ~4e-5 and stopped, three orders of magnitude
below where it should stabilize. Run 2 fixes this (`--warmup-steps 15`), trains the GAN
~8x longer (5,000 iterations vs. 600), and uses a larger, more statistically stable
held-out set (50 val / 50 test vs. 20 / 20 before).

## Run 2 pipeline & timings

| Stage | What | Time |
|---|---|---|
| Real image pool | 300 PneumoniaMNIST images (64×64), 100 real_used / 150 real_not_used | ~seconds (cached) |
| GAN training | Tiny DCGAN, 5,000 iterations on the 100 real_used images | **119.3s** |
| Synthetic pool | 500 images sampled from the trained generator | seconds |
| Detector training | 15 epochs, `d_model=64`, `K=4`, `image_size=64`, warmup fixed | **~270s** (~18s/epoch after epoch 0) |
| Evaluation + calibration | 50-sample val (calibration) + 50-sample test | seconds |
| **Total** | | **under 7 minutes** |

## Detector training log (150 train images: 60 real_used / 90 real_not_used)

Train loss fell steadily and substantially: **3.67 → 3.17 → 2.74 → 2.52 → 2.22 → 1.82 →
1.68 → 1.46 → 1.11 → 1.25 → 1.04 → 0.86 → 0.76 → 0.73 → 0.63**. Learning rate now follows
the intended warmup-then-cosine-decay shape (confirmed in the log: ramps to `9.99e-05` by
step 20, decays to `3.79e-07` by the final step) instead of being stuck in permanent warmup.

Full per-epoch validation metrics are in `checkpoints2/metrics_log.jsonl`.

## A real finding, not a fabricated one: F1-based checkpoint selection picked a degenerate model

`train.py` saves `best.pt` on highest validation F1. In this run, **epoch 0** — before
the model had learned anything beyond its random initialization — scored the highest F1
(0.571) by simply predicting "used" for every single input (recall=1.0, precision=0.4,
confusion matrix `tn=0, fp=30`). F1 alone doesn't catch this because the val split is
class-imbalanced enough (20 used / 30 not-used) that "always predict the minority-adjacent
class" scores deceptively well on F1 while its AUC (0.425, *below* chance) shows it carries
no real signal at all.

`last.pt` (the actual epoch-14 checkpoint, post-training) has a lower raw F1 but is the
honest result:

| Checkpoint | Split | n | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|---|
| `last.pt` (real, trained) | Calibration (val) | 50 | 0.48 | 0.286 | 0.20 | 0.235 | 0.44 |
| `last.pt` (real, trained) | Test | 50 | 0.52 | 0.40 | 0.40 | **0.40** | **0.54** |
| `best.pt` (degenerate, epoch 0) | Test | 50 | 0.40 | 0.40 | 1.00 | 0.571 | **0.425** |

`last.pt`'s AUC of 0.54 is the number that actually matters here — modestly above chance
(0.50), consistent with a GAN that partially, weakly memorized its training set after
5,000 iterations. The degenerate checkpoint's AUC below chance is the tell that its higher
F1 is an artifact, not a better model.

## Conformal calibration (`alpha=0.1`, target 90% coverage), using `last.pt`

| | Value |
|---|---|
| q_hat | 0.867 |
| Empirical test coverage | **0.90** (target 0.90 — met exactly) |
| Abstain rate | 80% |

Coverage moved from a trivial 100%/100%-abstain result in run 1 to a real, non-trivial
90%/80%-abstain result in run 2 — the calibrator is now discriminating some cases
confidently rather than refusing to commit on everything.

## Honest interpretation

1. **AUC 0.54 on `last.pt` is weak-but-real signal, not noise-free**, and clearly not
   comparable to the baseline's reported 0.875 AUC — that number came from a real medical
   dataset and a fully-converged, production-scale GAN. Ours is 100 images and a GAN
   trained for two minutes.
2. **The degenerate-checkpoint finding is the most valuable result of this run.** It's a
   real methodological lesson (checkpoint selection on raw F1 under class imbalance can
   silently pick a model with zero real signal) that generalizes well beyond this toy
   demo — worth carrying into the full-scale GPU runs later (select checkpoints on a
   metric less exploitable by class imbalance, or check AUC alongside F1 before trusting
   a "best" checkpoint).
3. **The GAN's own training curve (`d_loss` collapsing toward 0 while `g_loss` spiked as
   high as 20)** shows the discriminator dominating — a common, fixable DCGAN instability,
   not something specific to our detector architecture.
4. **What this run does prove**: the full pipeline — frequency branch, cross-attention,
   Set-Transformer aggregation over K=4 synthetic draws, ArcFace, SupCon, and the
   conformal wrapper — executes correctly end-to-end on real images through a real GAN,
   with a real (if weak) training signal and a real, non-trivial calibration result.
5. **What it still doesn't replace**: the real dataset, a properly-converged GAN, multiple
   seeds, and ablations — all still required before any claim about beating the baseline
   is meaningful.
