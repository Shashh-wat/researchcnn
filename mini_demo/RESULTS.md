# Mini Demo: Results

Fully local, Mac-only, smallest-time end-to-end run of the pipeline: real images → a real
(if tiny) GAN we trained ourselves → the actual detector architecture → evaluation →
conformal calibration. This is a **correctness/demo run, not a research result** — see
the interpretation section at the bottom before reading anything into the numbers.

Raw source files: `checkpoints/metrics_log.jsonl` (per-epoch training/val), `cal_metrics.json`,
`test_metrics.json`, `conformal_results.json`.

## Pipeline & timings

| Stage | What | Time |
|---|---|---|
| Real image pool | 200 PneumoniaMNIST images (64×64, chest X-ray, labels discarded) | ~90s (download) |
| Real_used / real_not_used split | 70 / 70, via `prepare_manifest.py` | seconds |
| GAN training | Tiny DCGAN, 600 iterations on the 70 real_used images | **18.2s** |
| Synthetic pool | 300 images sampled from the trained generator | seconds |
| Detector training | 8 epochs, `d_model=64`, `K=4`, `image_size=64`, no pretrained weights | **~163s** (~20s/epoch) |
| Evaluation + calibration | val (as calibration) + test, `evaluate.py` + `calibrate.py` | seconds |
| **Total** | | **under 5 minutes**, far inside the 30–90 min estimate |

## Detector training log (`real_train.csv`, 100 images, 50/50 split)

| Epoch | Train loss | Val Acc | Val Precision | Val Recall | Val F1 | Val AUC |
|---|---|---|---|---|---|---|
| 0 | 3.461 | 0.50 | 0.00 | 0.00 | 0.00 | 0.67 |
| 1 | 3.252 | 0.50 | 0.50 | 0.10 | 0.17 | 0.43 |
| 2 | 2.985 | 0.55 | 0.53 | 0.80 | 0.64 | 0.40 |
| 3 | 2.723 | 0.45 | 0.40 | 0.20 | 0.27 | 0.44 |
| 4 | 2.629 | 0.60 | 0.56 | 1.00 | 0.71 | 0.55 |
| 5 | 2.623 | 0.50 | 0.50 | 0.40 | 0.44 | 0.52 |
| 6 | 2.306 | 0.65 | 0.64 | 0.70 | 0.67 | 0.54 |
| 7 (best by F1) | 2.393 | 0.50 | 0.00 | 0.00 | 0.00 | 0.40 |

`best.pt` was saved at **epoch 4** (highest val F1 = 0.714, before epoch 6/7 both regressed).

## Final evaluation (using `best.pt`, checkpoint from epoch 4)

| Split | n | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Calibration (= val split, reused) | 20 | 0.60 | 0.556 | 1.00 | 0.714 | 0.53 |
| Test | 20 | 0.30 | 0.375 | 0.60 | 0.462 | 0.19 |

## Conformal calibration (`alpha=0.1`, target 90% coverage)

| | Value |
|---|---|
| q_hat | 0.557 |
| Empirical test coverage | **1.00** (target 0.90 — met, trivially) |
| Abstain rate | **100%** — every single test prediction set had size ≠ 1 |

## Honest interpretation — read this before drawing any conclusion

1. **This is not a result, it's a wiring test.** 100 training images, 20 val, 20 test, a
   DCGAN trained for 18 seconds — there is no statistical basis to claim the detector
   learned anything about GAN fingerprints specifically. Val/test AUC bouncing between
   0.19 and 0.67 across epochs/splits is consistent with noise, not signal.
2. **The conformal calibrator did exactly what it's supposed to do given a weak model**:
   faced with a score that carries no real information at this scale, it abstained on
   100% of test points rather than issue a falsely confident prediction — that's the
   coverage guarantee (math.md §9) working as designed, not a bug. A model with real
   signal would abstain far less.
3. **Train loss did trend down** (3.46 → ~2.3-2.4, noisily) — confirms gradients are
   flowing and the combined BCE+ArcFace+SupCon loss is optimizable, which is the one
   thing this run can actually claim to have verified beyond pure smoke-testing.
4. **Every architectural piece ran for real, on real images, for the first time in this
   project**: the frequency branch's FFT, the cross-attention, the Set-Transformer
   aggregation over K=4 synthetic draws, ArcFace, SupCon, and the conformal wrapper all
   executed against actual GAN output rather than random tensors.
5. **What this does NOT replace**: the real dataset (ImageCLEF or a properly-sized real
   pool), a real GAN trained for real convergence, multiple seeds, and ablations — all
   still needed before any claim about beating the baseline is meaningful.
