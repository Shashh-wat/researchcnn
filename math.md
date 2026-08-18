# Math Specification: Spatial–Frequency Set-Transformer for GAN Training-Data Membership Inference

This document is the single source of truth for the formulas implemented in `src/`. Every symbol here maps to a named variable/module in code. Default dims used throughout: embedding width $d=256$, attention heads $h=8$, cross-attention layers $L=2$, synthetic draws per real image $K=8$.

---

## 1. Token extraction

**Spatial-CNN stream.** ResNet-50 truncated before global average pool, on input $I \in \mathbb{R}^{3\times224\times224}$:

$$F^{\text{res}} = \text{ResNet50}_{\le \text{layer4}}(I) \in \mathbb{R}^{2048\times7\times7} \;\to\; \text{flatten} \;\to\; \mathbb{R}^{49\times2048} \;\to\; T^{\text{res}} = F^{\text{res}} W_{\text{res}} \in \mathbb{R}^{49\times d}, \quad W_{\text{res}} \in \mathbb{R}^{2048\times d}$$

**Spatial-Transformer stream.** ViT-B/16 patch tokens (CLS dropped):

$$F^{\text{vit}} = \text{ViT-B/16}(I)[1:] \in \mathbb{R}^{196\times768} \;\to\; T^{\text{vit}} = F^{\text{vit}} W_{\text{vit}} \in \mathbb{R}^{196\times d}, \quad W_{\text{vit}} \in \mathbb{R}^{768\times d}$$

**Frequency stream.** Grayscale $I_g = 0.299R+0.587G+0.114B$, 2D FFT, shifted, log-magnitude:

$$M(I) = \log\big(1 + |\mathcal{F}\{I_g\}|\big), \qquad \mathcal{F}\{I_g\}\text{ shifted so DC is centered}$$

$M(I)$ is normalized (per-sample z-score) then embedded with a strided conv acting as a $16\times16$ non-overlapping patch embedding, directly producing a $14\times14$ token grid:

$$T^{\text{freq}} = \text{Conv2d}_{1\to d,\,k=16,\,s=16}\big(M(I)\big) \in \mathbb{R}^{196\times d}$$

**Fusion.** Each stream gets a learned modality embedding $E_m \in \mathbb{R}^d$ ($m \in \{\text{res}, \text{vit}, \text{freq}\}$) and a learned positional embedding $P$ matched to its grid, then all three are concatenated:

$$T_I = \big[\,T^{\text{res}}{+}E_{\text{res}}{+}P^{7\times7} \;;\; T^{\text{vit}}{+}E_{\text{vit}}{+}P^{14\times14}_1 \;;\; T^{\text{freq}}{+}E_{\text{freq}}{+}P^{14\times14}_2 \,\big] \in \mathbb{R}^{441\times d}$$

Weights are **shared** between the real-image and synthetic-image tokenizer (Siamese property preserved from the baseline).

---

## 2. Bidirectional token-level cross-attention

For real tokens $T_r \in \mathbb{R}^{441\times d}$ and one synthetic draw's tokens $T_{g_k} \in \mathbb{R}^{441\times d}$, apply $L$ stacked cross-attention blocks in both directions. Per layer, per direction (shown for real-as-query):

$$Q = XW_Q,\; K = YW_K,\; V = YW_V \qquad \text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d/h}}\right)V$$

$$X' = \text{LayerNorm}\big(X + \text{MultiHead-Attn}(X,Y)\big), \qquad X'' = \text{LayerNorm}\big(X' + \text{FFN}(X')\big)$$

applied with $X{=}T_r, Y{=}T_{g_k}$ to get $\hat T_r^{(k)}$, and symmetrically with $X{=}T_{g_k}, Y{=}T_r$ to get $\hat T_{g_k}^{(r)}$. Weights are independent between the two directions but shared across the $K$ draws and across layers within the stack.

---

## 3. Set-pooling by Multihead Attention (PMA)

Used three times: to collapse 441 attended tokens → 1 vector (twice, once per side), and later to aggregate $K$ relation vectors → 1 vector. General form with learned seed $q \in \mathbb{R}^{1\times d}$ over a token set $X \in \mathbb{R}^{N\times d}$:

$$\text{PMA}(q, X) = \text{softmax}\!\left(\frac{qW_Q'(XW_K')^\top}{\sqrt{d}}\right) XW_V' \in \mathbb{R}^{1\times d}$$

$$h_r^{(k)} = \text{PMA}_{\text{real}}\big(q_r, \hat T_r^{(k)}\big), \qquad h_{g_k}^{(r)} = \text{PMA}_{\text{synth}}\big(q_g, \hat T_{g_k}^{(r)}\big)$$

---

## 4. Pairwise relation vector

$$u_k = \big[\,h_r^{(k)} - h_{g_k}^{(r)}\;;\; h_r^{(k)} \odot h_{g_k}^{(r)}\;;\; h_r^{(k)}\;;\; h_{g_k}^{(r)}\,\big] \in \mathbb{R}^{4d}$$

$$r_k = \text{ReLU}(u_k W_{\text{rel}}) \in \mathbb{R}^{d} \qquad \text{(fed to set aggregation, §5)}$$

$$z_k = \text{MLP}_{\text{aux}}(u_k) \in \mathbb{R} \qquad \text{(auxiliary per-draw logit, for inspection/ablation only)}$$

---

## 5. Set aggregation over $K$ synthetic draws → membership score

$$\bar u(r) = \text{PMA}_{\text{set}}\big(q_{\text{set}}, \{r_1,\dots,r_K\}\big) \in \mathbb{R}^d$$

$$s(r) = \sigma\big(\text{MLP}_{\text{score}}(\bar u(r))\big) \in [0,1]$$

$s(r)$ is a **permutation-invariant statistic over the synthetic pool**, i.e. it doesn't depend on the order the $K$ draws were sampled in — this is the formal property that makes it a distribution-level statistic rather than an instance-level one, matching the baseline's own stated (but unimplemented) goal.

**Identity embedding**, used by the ArcFace and SupCon losses below, is the real-image-side pooled representation averaged over all $K$ cross-attention instances:

$$e(r) = \frac{1}{K}\sum_{k=1}^{K} h_r^{(k)} \in \mathbb{R}^d$$

---

## 6. ArcFace additive angular margin loss

Class prototypes $W_0, W_1 \in \mathbb{R}^d$ (for `real_not_used`, `real_used`), learnable, with scale $s_{\text{arc}}$ and margin $m$:

$$\hat e = \frac{e(r)}{\|e(r)\|}, \qquad \hat W_j = \frac{W_j}{\|W_j\|}, \qquad \cos\theta_j = \hat W_j^\top \hat e, \quad j\in\{0,1\}$$

$$\theta_j = \arccos(\text{clamp}(\cos\theta_j, -1+\epsilon, 1-\epsilon))$$

$$\text{logit}_j = \begin{cases} s_{\text{arc}}\cos(\theta_j + m) & j = y \\ s_{\text{arc}}\cos\theta_j & j \ne y \end{cases}$$

$$L_{\text{ArcFace}} = -\log \frac{e^{\,\text{logit}_y}}{\sum_{j} e^{\,\text{logit}_j}}$$

Geometrically, this enforces a minimum angular gap of $m$ radians between the `real_used` and `real_not_used` clusters on the unit hypersphere in $\mathbb{R}^d$ — a property the baseline's post-hoc cosine hinge (which penalizes similarity *after* the fact, not the angle itself) does not guarantee.

---

## 7. Supervised Contrastive (SupCon) loss

For a batch of $B$ identity embeddings $\{\hat e_i\}_{i=1}^B$ (L2-normalized) with labels $y_i \in \{0,1\}$, let $P(i) = \{p \ne i : y_p = y_i\}$ and $A(i) = \{a \ne i\}$:

$$L_{\text{SupCon}} = \frac{1}{B}\sum_{i=1}^{B} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\hat e_i \cdot \hat e_p / \tau)}{\sum_{a \in A(i)} \exp(\hat e_i \cdot \hat e_a / \tau)}$$

with temperature $\tau$ (default $0.1$). Anchors with $|P(i)|=0$ (no same-class partner in the batch) are excluded from the mean. Batch size must be large enough that both classes are represented (the training script uses a class-balanced sampler to guarantee this).

---

## 8. Total training objective

$$L_{\text{total}} = \lambda_1\, L_{\text{BCE}}\big(s(r), y\big) \;+\; \lambda_2\, L_{\text{ArcFace}} \;+\; \lambda_3\, L_{\text{SupCon}}$$

$$L_{\text{BCE}} = -\frac{1}{B}\sum_i \big[y_i \log s(r_i) + (1-y_i)\log(1-s(r_i))\big]$$

Default weights: $\lambda_1{=}1.0,\ \lambda_2{=}0.5,\ \lambda_3{=}0.3$ (tune empirically as in the baseline's own $\lambda$ ablation, §4.5.1 of the reference paper).

---

## 9. Split conformal prediction (post-hoc, no retraining)

Given a calibration set $\mathcal{D}_{\text{cal}} = \{(r_i,y_i)\}_{i=1}^n$ disjoint from train/test, and the trained score $s(\cdot)$, define per-class score $\hat p(r)[1] = s(r)$, $\hat p(r)[0] = 1-s(r)$. Nonconformity:

$$\alpha_i = 1 - \hat p(r_i)[y_i]$$

$(1-\epsilon)$-quantile with finite-sample correction:

$$\hat q = \text{Quantile}\left(\{\alpha_i\}_{i=1}^n \,;\, \frac{\lceil (n+1)(1-\epsilon)\rceil}{n}\right)$$

Prediction set for new real image $r$:

$$C(r) = \big\{\, y \in \{0,1\} : 1 - \hat p(r)[y] \le \hat q \,\big\}$$

**Guarantee.** Under exchangeability of $(\mathcal{D}_{\text{cal}} \cup \{r_{\text{test}}\})$, $P(y_{\text{test}} \in C(r_{\text{test}})) \ge 1-\epsilon$, marginally, distribution-free, finite-sample — independent of whether $s(\cdot)$ is well-calibrated or even close to correct. Note on validity here: because each calibration point aggregates its own fixed-size draw of $K$ synthetic images via a fixed, order-invariant aggregation function (§5), and every real image (calibration or test) is subject to the identical sampling-and-aggregation procedure, the *per-real-image* nonconformity scores $\alpha_i$ are themselves i.i.d./exchangeable across images — standard split-conformal validity applies directly at the real-image level with no additional assumptions required.

---

## 10. On mathematical novelty (cross-reference)

As established in `novelty.md` §6: every individual component above (ArcFace, SupCon, PMA/Set Transformer, split conformal prediction) is drawn from established literature (2019–2023). The composition is architectural, not a new theorem. If a genuine mathematical contribution is pursued later, the credible route is grounding $s(r)$ against the Bayes-optimal likelihood-ratio membership inference test (Sablayrolles et al. 2019; Carlini et al.'s LiRA, 2022) rather than an MMD-witness-function framing — see `novelty.md` for the full discussion of why that path was preferred over the alternatives considered and discarded.
