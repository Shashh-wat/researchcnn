# Novelty Document: Spatial–Frequency Set-Transformer for Calibrated GAN Training-Data Membership Inference

Baseline: Kalinathan et al., *"Hybrid Siamese Architecture for Detecting Genuine Training Data Fingerprints in Medical Generative Adversarial Networks"*, Sci Rep (2026). Baseline reports Accuracy 0.883±0.03, F1 0.540±0.06, Precision 0.461±0.09 on lung-CT GAN membership detection (N=100 real_used / 100 real_not_used / 5000 synthetic).

---

## 1. Problem Restatement

Given a real medical image $I_r$, decide $y \in \{0,1\}$: was $I_r$ a member of the GAN's training set (`real_used`, $y=1$) or held out (`real_not_used`, $y=0$)? This is **membership inference (MI)** against a generative model, not ordinary real/fake classification. The only usable evidence is indirect: the statistical fingerprint $I_r$ may have left on the generator's *output distribution*, observed through a pool of synthetic samples $\{I_{g_1}, \dots, I_{g_M}\}$.

The baseline's own "Theoretical Rationale" (§3.4.3) correctly identifies that the signal is **distributional**, not instance-level — then implements it as a single arbitrary real–synthetic pair per training example. The redesign below closes that gap directly.

---

## 2. Token Flow: From Pixels to Membership Score

### 2.1 Three parallel token streams per image

Every image (real or synthetic) is encoded into three token sets instead of one pooled vector, so spatial/frequency structure survives into the fusion stage.

| Branch | Source | Grid | Token dim (pre-proj) | Purpose |
|---|---|---|---|---|
| **Spatial-CNN** | ResNet-50 Layer4 feature map, *not* GAP'd | 7×7 = 49 tokens | 2048 | local texture / anatomical structure |
| **Spatial-Transformer** | ViT-B/16 patch tokens (CLS dropped) | 14×14 = 196 tokens | 768 | long-range structural context |
| **Frequency** | log-magnitude 2D-FFT of the image, patchified 16×16 | 14×14 = 196 tokens | 256 (small conv embed) | periodic upsampling artifacts GANs leave (Zhang et al. 2019; Frank et al. 2020) |

Each stream is linearly projected to a common width $d$ and given its own **learned modality-type embedding** (segment-embedding style) plus a **2D sinusoidal or learned positional encoding** matched to its grid:

$$T^{\text{res}} = F^{\text{res}} W_{\text{res}} + E_{\text{res}} + P^{7\times7}, \quad T^{\text{vit}} = F^{\text{vit}} W_{\text{vit}} + E_{\text{vit}} + P^{14\times14}, \quad T^{\text{freq}} = F^{\text{freq}} W_{\text{freq}} + E_{\text{freq}} + P^{14\times14}$$

Concatenating gives one token bag per image:

$$T_I = \big[\,T^{\text{res}} \,;\, T^{\text{vit}} \,;\, T^{\text{freq}}\,\big] \in \mathbb{R}^{441 \times d}$$

This replaces the baseline's premature global-average-pool (2048-d vector before any fusion happens) with a representation that still carries *where in the image* and *at what spatial frequency* the evidence lives.

### 2.2 Bidirectional token-level cross-attention (real ↔ synthetic)

For a real image $r$ and one synthetic draw $g_k$, run $L{=}2$ standard transformer cross-attention blocks, real-as-query/synthetic-as-key-value and the reverse, each followed by FFN + residual + LayerNorm:

$$Q = T_r W_Q,\quad K = T_{g_k} W_K,\quad V = T_{g_k} W_V, \qquad \text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V$$

Unlike the baseline's single global-vector cross-attention (Eq. 6–7 in the paper — one query, one key, no spatial structure), this is a full $441\times441$ attention matrix. Reshaping the attention weights back onto the 7×7, 14×14, and 14×14 grids yields a **fingerprint localization heat-map**: which anatomical regions and which spatial frequencies the model actually used to flag membership. This is the interpretability payoff the baseline lacks entirely.

### 2.3 Learned set-pooling (not GAP)

A learned aggregator query $q \in \mathbb{R}^d$ attends over the 441 attended tokens to produce one fixed vector per (real, synthetic-draw) pair:

$$h_r^{(k)} = \text{softmax}\!\left(\frac{qW_Q(\hat T_r^{(k)}W_K)^\top}{\sqrt{d}}\right)\hat T_r^{(k)} W_V \in \mathbb{R}^d$$

and symmetrically $h_{g_k}^{(r)}$ for the synthetic side.

### 2.4 Pairwise relation vector

$$u_k = \big[\,h_r^{(k)} - h_{g_k}^{(r)}\,,\; h_r^{(k)} \odot h_{g_k}^{(r)}\,,\; h_r^{(k)}\,,\; h_{g_k}^{(r)}\,\big] \quad\rightarrow\quad z_k = \text{MLP}(u_k) \in \mathbb{R}$$

This generalizes the baseline's Eq. 8 (abs-diff + dot-product only) with a richer, still-cheap relation representation.

### 2.5 Set aggregation over K synthetic draws — the actual fix for "distribution-level" evidence

Instead of one arbitrary pair, sample $K$ synthetic images per real image and treat $\{u_1,\dots,u_K\}$ as an **unordered set**. Aggregate with Pooling by Multihead Attention (Set Transformer, Lee et al. 2019) using a learned seed $q'$:

$$\bar{u}(r) = \text{PMA}(q', U) = \text{softmax}\!\left(\frac{q'W_Q'(UW_K')^\top}{\sqrt{d}}\right) U W_V', \qquad U = [u_1;\dots;u_K]$$

$$s(r) = \sigma\big(\text{MLP}(\bar u(r))\big) \in [0,1]$$

This is a **permutation-invariant statistic over the synthetic pool** — mathematically the object the baseline's own rationale calls for but never builds (see §5 for the formal interpretation of this as a two-sample-test witness function).

---

## 3. ArcFace Angular Margin Penalty — why it replaces the baseline's angular loss

The baseline's angular term (Eq. 10–11) is a **post-hoc hinge on raw cosine similarity**:

$$L_{\text{ang}} = \frac{1}{N}\sum_i \Big[y_i(1-\cos\theta_i) + (1-y_i)\max(0, \cos\theta_i - m)\Big]$$

This only penalizes cosine similarity *after* it's computed — it does not reshape the angular geometry of the embedding space itself, and margin enforcement is asymmetric/soft (hinge only fires past a threshold).

**ArcFace** (Deng et al. 2019) instead inserts the margin *inside the angle*, before the cosine is taken, against class prototypes $W_0$ (`real_not_used`) and $W_1$ (`real_used`):

$$\hat h = \frac{h_r}{\|h_r\|}, \qquad \hat W_j = \frac{W_j}{\|W_j\|}, \qquad \cos\theta_j = \hat W_j^\top \hat h$$

$$L_{\text{ArcFace}} = -\log \frac{e^{\,s\cos(\theta_y + m)}}{e^{\,s\cos(\theta_y + m)} + \sum_{j \ne y} e^{\,s \cos\theta_j}}$$

with scale $s$ (e.g. 30) and margin $m$ (e.g. 0.2–0.5 rad, smaller than face-recognition defaults given only 2 classes and N=100/class). Geometrically this **guarantees a minimum angular gap** between the `real_used` and `real_not_used` clusters on the hypersphere — a provably tighter decision boundary than a hinge applied after the fact, and it is the standard replacement for exactly this kind of cosine-based auxiliary loss in modern metric learning (face verification, re-ID, few-shot).

Applied here on the per-real-image embedding $h_r$ (before the pairwise relation step), it directly attacks the low-precision failure mode the baseline reports (Precision 0.461, 78 false positives) by making the two membership classes angularly well-separated regardless of which synthetic sample is drawn.

---

## 4. Supervised Contrastive (SupCon) Loss — why it matters specifically at N=100/class

Baseline supervision is essentially pairwise (one real, one synthetic, one label). With a batch of $B$ real embeddings $\{\hat h_i\}$ and labels $y_i \in \{0,1\}$, SupCon (Khosla et al. 2020) uses **every same-class pair in the batch as a positive**:

$$L_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\hat h_i \cdot \hat h_p / \tau)}{\sum_{a \in A(i)} \exp(\hat h_i \cdot \hat h_a / \tau)}$$

where $P(i) = \{p \ne i : y_p = y_i\}$, $A(i)$ = all other samples in the batch, $\tau$ = temperature.

The combinatorial payoff: a batch of size $B$ yields $O(B^2)$ supervisory contrastive pairs versus $O(B)$ pairs in a plain pairwise scheme. At N=100/class this is the difference between a network that has seen ~100 labeled examples and one that has effectively seen ~$\binom{100}{2}$ same-class relationships — directly targeting the low-data regime the baseline paper flags repeatedly as its core limitation (§4.2, §4.3).

**Total training objective:**

$$L = \lambda_1 L_{\text{BCE}}(s(r), y) + \lambda_2 L_{\text{ArcFace}} + \lambda_3 L_{\text{SupCon}}$$

$L_{\text{BCE}}$ supervises the final set-aggregated membership score; $L_{\text{ArcFace}}$ and $L_{\text{SupCon}}$ jointly regularize the *geometry* of the per-image embedding space that feeds it — one via hard class prototypes with margin, the other via dense pairwise attraction/repulsion.

---

## 5. Conformal Prediction — turning a score into a guaranteed-coverage decision

The baseline's Discussion explicitly says calibrated confidence is essential in forensic/clinical settings, then never delivers it (sigmoid outputs are not calibrated probabilities; Fig. 6d shows real miscalibration). Conformal prediction (Vovk et al.; Angelopoulos & Bates, 2023 tutorial) fixes this **without assuming the score model is correct**.

**Procedure:**

1. Hold out a calibration set $\mathcal{D}_{\text{cal}} = \{(r_i, y_i)\}_{i=1}^n$, disjoint from train/test.
2. Nonconformity score: $\alpha_i = 1 - s(r_i)[y_i]$ (one minus the model's predicted probability of the *true* class).
3. Compute the empirical $(1-\epsilon)$ quantile:

$$\hat q = \text{Quantile}\left(\{\alpha_i\}_{i=1}^n;\; \frac{\lceil (n+1)(1-\epsilon) \rceil}{n}\right)$$

4. For a new real image $r$, output the **prediction set**:

$$C(r) = \{\, y \in \{0,1\} : 1 - s(r)[y] \le \hat q \,\}$$

**Guarantee (distribution-free, finite-sample):** $P\big(y \in C(r)\big) \ge 1 - \epsilon$, regardless of whether the underlying network is well-specified.

**Operational meaning for this task:**
- $C(r) = \{1\}$ → confidently a training-set member.
- $C(r) = \{0\}$ → confidently not.
- $C(r) = \{0,1\}$ → abstain, flag for manual forensic/clinical review.

This directly operationalizes the "risk-aware decision making" language already in the baseline's Discussion section, and is a cheap, mathematically rigorous add-on (no retraining, just a calibration pass).

---

**Two concrete paths that would make mathematical novelty, not just architectural:**

1. **Formalize $s(r)$ as a learned witness function for a two-sample test.** The set-aggregated score in §2.5 is structurally identical to a kernel/neural witness function used in Maximum Mean Discrepancy (MMD) two-sample testing (Gretton et al.) between the point mass $\delta_r$ and the empirical synthetic distribution $\hat P_G$ induced by the GAN. If you derive (not just assert) that $s(r)$ converges to — or bounds — a proper MMD-style divergence between $r$'s neighborhood and the generator's output manifold, that is a real, citable theoretical contribution: it reframes "training-data fingerprint detection" as *a specific instance of generative-model two-sample testing*, connecting an applied forensics problem to an established statistical framework in a way the baseline (and most GAN-fingerprinting papers) do not.
2. **Prove a coverage/set-size bound under the set-aggregation estimator.** Standard conformal guarantees hold for i.i.d. calibration/test data; here, calibration examples are themselves built from *sets* of $K$ correlated synthetic draws per real image. Working out whether/how exchangeability survives this construction (and what it implies for conformal set size as a function of $K$, $M$, and the angular margin $m$) would be a genuine, non-trivial derivation — and directly relevant, since it tells you how many synthetic samples per real image you actually need for the coverage guarantee to hold tightly.

Either of those, worked out with proofs (not just cited), is what would let you say "novel architecture with a mathematical contribution" rather than "well-composed novel architecture." Recommend picking one — (1) is more tractable and has a cleaner narrative arc (deepfake/MI detection *is* two-sample testing) — before committing engineering time to the rest.
