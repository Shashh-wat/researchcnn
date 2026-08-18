# Architecture Specification

Companion to `math.md` (formulas) and `novelty.md` (motivation/positioning). This document maps the system to code modules and describes data flow end to end. Formal symbols ($T_I$, $h_r^{(k)}$, $s(r)$, etc.) match `math.md` exactly.

```
src/
  config.py         Config dataclass — all hyperparameters in one place
  backbone.py        SpatialCNNTokenizer, SpatialViTTokenizer, FrequencyTokenizer, TokenFusion
  attention.py        CrossAttentionStack (bidirectional, L layers), PMAPool (set-pooling)
  heads.py             RelationHead, ArcFaceHead
  losses.py             SupConLoss, CombinedLoss
  model.py               MembershipFingerprintNet — assembles everything, single forward() entry point
  conformal.py             SplitConformalCalibrator
  dataset.py                RealSyntheticSetDataset — (real image, K synthetic draws, label)

scripts/
  smoke_test.py       shape/gradient sanity check with random tensors, no data or downloads required
  train.py              training loop entry point (run on real data + GPU)
  calibrate.py           post-hoc conformal calibration + prediction-set demo
```

## 1. End-to-end forward pass

Input to the model per batch: a real image $I_r$ and $K$ synthetic images $\{I_{g_1},\dots,I_{g_K}\}$ drawn from the same generator's output pool, plus the membership label $y$.

```
I_r  (B,3,224,224)  ──┐
                       ├─► tokenize() [shared weights] ─► T_r  (B,441,d)
I_g  (B,K,3,224,224) ─┘        │
                                └─► tokenize() ─► T_g  (B·K,441,d)
```

`tokenize()` (in `backbone.py`, orchestrated by `TokenFusion`) runs the three parallel streams and concatenates:

```
image ─► SpatialCNNTokenizer  ─► T^res  (49,d)   ─┐
      ─► SpatialViTTokenizer  ─► T^vit  (196,d)  ─┼─► concat + modality/pos embed ─► T_I (441,d)
      ─► FrequencyTokenizer   ─► T^freq (196,d)  ─┘
```

The real token bag is broadcast across the $K$ draws, and cross-attention runs once per (real, draw) pair:

```
T_r (repeated K times), T_g[k]  ──► CrossAttentionStack (L=2, bidirectional)
                                        │
                     ┌──────────────────┴──────────────────┐
              attended_r (B·K,441,d)            attended_g (B·K,441,d)
                     │                                      │
              PMAPool_real                            PMAPool_synth
                     │                                      │
              h_r^(k)  (B·K,d)                        h_g^(k)  (B·K,d)
```

Reshape both to `(B,K,d)`. Per-draw relation vectors:

```
h_r, h_g (B,K,d) ──► RelationHead ──► r_k (B,K,d) [for aggregation], z_k (B,K,1) [auxiliary, inspection only]
```

Set aggregation over the $K$ draws (permutation-invariant — order of synthetic draws must not matter, this is checked in the smoke test):

```
r_1..r_K (B,K,d) ──► PMAPool_set ──► ū(r) (B,d) ──► MLP_score ──► sigmoid ──► s(r) (B,) ∈ [0,1]
```

Identity embedding for the metric-learning losses (independent of any single draw):

```
e(r) = mean_k h_r^(k)   (B,d)
```

```
e(r) ──► ArcFaceHead(e(r), y) ──► L_ArcFace
e(r) ──► SupConLoss(batch of e(r), y) ──► L_SupCon
s(r) ──► BCE(s(r), y) ──► L_BCE
```

`model.forward()` returns `{"score": s(r), "identity_embedding": e(r), "arcface_logits": ..., "per_draw_logit": z_k}` — `losses.CombinedLoss` consumes this dict directly.

## 2. Why each design choice maps to a specific baseline weakness

| Baseline weakness (see `novelty.md` §2) | Fix in this architecture | Where in code |
|---|---|---|
| Global-average-pooled features before fusion destroys localized artifacts | Token-level fusion (441 tokens), spatial structure preserved through cross-attention | `backbone.py`, `attention.py` |
| No frequency-domain signal despite literature consensus that GAN upsampling artifacts are spectral | Dedicated FFT log-magnitude branch, patch-embedded and fused identically to the other two streams | `backbone.py: FrequencyTokenizer` |
| Single arbitrary real–synthetic pair per training example, despite the paper's own claim that fingerprints are distributional | $K$ synthetic draws aggregated via a permutation-invariant Set-Transformer pooling into one score | `attention.py: PMAPool`, `model.py` |
| Post-hoc cosine hinge loss doesn't enforce embedding geometry | ArcFace additive angular margin on the identity embedding | `heads.py: ArcFaceHead` |
| Only $O(B)$ supervisory pairs per batch at N=100/class | SupCon gives $O(B^2)$ same-class contrastive pairs per batch | `losses.py: SupConLoss` |
| No calibrated confidence despite the paper stating this is needed for clinical/forensic use | Split conformal wrapper with coverage guarantee, applied post-hoc | `conformal.py` |
| Single dataset / single generator / single modality, no generalization evidence | `dataset.py` is modality-agnostic (manifest-driven); intended to be run against ≥2 domains (e.g. lung CT + dental radiographs) per the plan discussed | `dataset.py`, `scripts/train.py` |

## 3. Tensor shape reference (defaults: $d=256$, $h=8$ heads, $L=2$, $K=8$)

| Tensor | Shape | Produced by |
|---|---|---|
| `T^res` | (49, 256) | `SpatialCNNTokenizer` |
| `T^vit` | (196, 256) | `SpatialViTTokenizer` |
| `T^freq` | (196, 256) | `FrequencyTokenizer` |
| `T_I` | (441, 256) | `TokenFusion` |
| `attended_r`, `attended_g` | (441, 256) each | `CrossAttentionStack` |
| `h_r^{(k)}`, `h_g^{(k)}` | (256,) each, per draw | `PMAPool` |
| `r_k` | (256,), per draw | `RelationHead` |
| `ū(r)` | (256,) | `PMAPool` (set aggregation) |
| `s(r)` | scalar in [0,1] | `model.py` score head |
| `e(r)` | (256,) | `model.py` (mean over K) |

## 4. Compute note

Cross-attention runs $B \times K$ times per forward pass (441×441 attention, twice per layer per direction, $L{=}2$ layers). With $K{=}8$ this is the dominant cost. Keep $K \in [4,8]$ and batch size modest (8–16) on a single consumer/workstation GPU; scale $K$ up only if hardware allows — $K$ does not need to be large for the permutation-invariance property to hold, only large enough that the aggregated statistic is stable (ablate this empirically once data is available).

## 5. Interpretability payoff

The 441×441 cross-attention weights (§`attention.py`) can be reshaped back onto the three source grids (7×7 spatial, 14×14 spatial, 14×14 frequency) to produce a fingerprint localization heat-map per (real, synthetic) pair — this is new relative to the baseline, whose single-global-vector attention has no spatial structure to visualize. Not yet wired into a plotting utility; flagged here as the natural next addition once real data is available to visualize against.
