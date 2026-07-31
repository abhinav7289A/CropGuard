# CropGuard — Brain

The theory behind every component that **actually exists in this repo**. Maths, mechanisms,
and the reasoning you would need to defend under hard questioning from a senior ML engineer.

`explaination.md` says *what the code does*. This says *why the maths works, and where it
breaks*.

**Scope note:** only built components are covered. **DVC is deliberately absent** — it is not
in this project yet, so there is nothing here to defend. Likewise no MLflow, sweeps,
calibration, drift detection, or Render deployment. Claiming those in an interview when the
repo does not contain them is exactly the trap this document exists to avoid.

**Also absent: results.** No model has been trained. Architecture and training theory below
describe what *will* run, not what has. Say so plainly if asked — "the pipeline is verified end
to end but the baseline has not been trained yet" is a fine answer; inventing an accuracy
number is not.

---

## 1. Data acquisition

### What happens

PlantVillage ships as a 2.2GB `data.zip` inside a HuggingFace repo, laid out as
`raw/<variant>/<class>/<file>.jpg` with three variants: `color`, `grayscale`, `segmented`.
We extract only `color` into an ImageFolder tree.

### Why not `datasets.load_dataset`

The repo is *script-based* — a `plant_village.py` builder plus the archive. `datasets` v4
removed loading-script execution (a supply-chain hardening decision: `trust_remote_code`
executed arbitrary code from the Hub). So the script is ignored and its `color`/`grayscale`/
`segmented` configs no longer exist.

Reading the archive directly is also **cheaper**: the script path decodes every JPEG into a
PIL image and re-encodes it into Arrow, which is 54K decode/encode round-trips and a
generational quality loss on lossy JPEG. Byte-level extraction preserves the exact published
bytes — which matters for reproducibility, since a re-encoded dataset is not the same dataset.

**Likely question — "How do you know your data is what the paper used?"**
The archive is content-addressed by the Hub (`data.zip` has a fixed hash), we copy bytes
without transcoding, and `manifest.json` records source, variant, class counts and totals.
Our count (54,305 / 38 classes) matches the published figure.

---

## 2. Data validation

Three gates, run before anything trains, exiting non-zero so CI fails on bad data:

| Check | Rule | Why |
|---|---|---|
| Integrity | `PIL.verify()` decodes | A truncated JPEG throws mid-epoch, killing a multi-hour run |
| Resolution | ≥128px short side | Upsampling to 224 fabricates detail that isn't there |
| Class balance | ≥100 samples/class | Below this, per-class metrics have uselessly wide error bars |

That last threshold is a **statistical** argument, not a stylistic one. Per-class recall is a
binomial proportion; its standard error is `√(p(1-p)/n)`. At n=100 and p≈0.9 that is ~3%, so a
95% CI spans roughly ±6pp. Below n=100 the interval widens past the point where per-class
numbers mean anything.

Result on our data: 54,305 images, 0 corrupt, 0 low-resolution, 0 underpopulated (smallest
class = 152).

---

## 3. The split — the deepest part of this project

### 3.1 The leakage mechanism

PlantVillage holds 54,305 images of only **~7,600 distinct physical leaves** — about 7
photographs per leaf, varying angle and lighting.

A per-image stratified split treats those 7 photos as 7 independent samples. They are not.
They share the same lesion pattern, same leaf geometry, same venation. Put 5 in train and 2 in
test and the model can succeed on test by **memorising leaf identity** rather than learning
disease morphology.

Formally: the standard generalisation argument assumes train and test are i.i.d. draws from
the same distribution *and mutually independent*. Near-duplicates violate the independence of
the two samples. Test error stops estimating the population risk

  R(f) = E_(x,y)~D [ L(f(x), y) ]

and instead estimates something between population risk and training error. The bias is
**downward** — it flatters the model.

**Measured on the naive split:**

```
test: 6,045 / 8,146 images (74.2%) shared a leaf with train
restricted to resolvable leaf IDs: 6,045 / 6,154 = 98% leaked
```

This is why PlantVillage papers routinely report 99%+. That number is close to meaningless.

### 3.2 Why this poisons everything downstream

The A/B testing framework in §8 compares models on the holdout. If the holdout is 74%
contaminated, McNemar's test faithfully answers **the wrong question** — "which model
memorises better?" not "which generalises better?" The statistics would be perfectly executed
and completely misleading. Rigour downstream cannot rescue a broken split upstream.

### 3.3 The grouped split

Two empirical facts made a clean algorithm possible:

1. **Groups never span classes** — verified, 0 of 20,015. A leaf has exactly one disease.
2. **The smallest class has 37 groups** — enough to divide three ways.

Fact 1 is what matters: it means the constraint graph decomposes. Group integrity and class
stratification would normally fight each other — a group spanning classes forces a trade-off.
Because none do, each class can be packed **independently and exactly**.

**Algorithm (per class):**

```
sort groups by size, descending          # LPT ordering
for each group g:
    target ← argmax_s ( fraction[s] · N − assigned[s] )    # largest deficit
    assign all of g to target
```

This is **Longest Processing Time first**, the classic greedy for multiway number
partitioning. Sorting descending is what makes it work: large groups are placed while there is
still slack to absorb them, and small groups act as fine-grained filler at the end. Reversed,
you place a 33-image group last into bins already at target and blow the fractions.

LPT's guarantee for makespan is `4/3 − 1/(3m)`; here we want *balance to a target ratio*
rather than minimum makespan, but the same intuition holds — worst-case imbalance is bounded
by roughly the largest remaining item, so error is `O(max group size / N)`. With max 33 and
N≈1,400 per class, that predicts ~2% worst case. Measured: mean deviation **0.0012**, worst
class 0.158 vs 0.150. Theory and measurement agree.

**Complexity:** `O(G log G)` per class for the sort, `O(G·3)` for assignment — trivial.

### 3.4 Results

| Metric | Naive | Grouped |
|---|---|---|
| test images sharing a leaf with train | 74.2% | **0.0%** |
| groups spanning a boundary | — | **0 / 20,015** |
| mean per-class \|test_frac − 0.15\| | — | **0.0012** |
| sizes | 38,013 / 8,146 / 8,146 | 38,008 / 8,172 / 8,125 |

Sizes are slightly uneven because groups are indivisible — you cannot hit exact fractions when
the atoms have size up to 33.

### 3.5 Honest limitations

- **~24% of images don't resolve** to a real leaf ID and become singleton groups. Those may
  still be near-duplicates of each other. So 0.0% is *measured on resolvable IDs*; true
  leakage is ≥0 but bounded well below 74%.
- **Grouping is only as good as `leaf-map.json`**, which is upstream-provided and unaudited.
- **A grouped split is harder**, so accuracy will *drop*. That drop is the honest number, and
  it is the whole point.

**Likely question — "Why not just use GroupKFold?"**
`StratifiedGroupKFold` approximates both constraints simultaneously and cannot hit arbitrary
70/15/15 fractions exactly — you get k folds of 1/k each, so 70/15/15 needs either k=20 with
fold-bundling (which risks classes vanishing from a fold when a class has few groups) or a
two-stage nested split that compounds error. Since groups never span classes here, the problem
decomposes and per-class LPT packing is exact, simpler, and lands within 0.12pp.

---

## 4. Class imbalance

The dataset is imbalanced **36.2×** — `Potato___healthy` 152, `Orange___Haunglongbing` 5,507.

### Why accuracy is the wrong metric

Accuracy is `Σ_c n_c · recall_c / Σ_c n_c` — a **support-weighted** mean of per-class recall.
The 5 largest classes hold ~40% of the data, so a model that ignores rare diseases entirely
still scores well. For crop disease that is exactly backwards: rare diseases are often the
ones worth catching.

### Macro-F1

Per class: `F1_c = 2·P_c·R_c / (P_c + R_c)` — the harmonic mean of precision and recall.
Harmonic, not arithmetic, because it is dominated by the *smaller* of the two: you cannot
score well by sacrificing one. Then

  `F1_macro = (1/K) Σ_c F1_c`

Unweighted, so `Potato___healthy` (152 images) counts exactly as much as
`Orange___Haunglongbing` (5,507). This drives checkpoint selection and early stopping.

**Note:** in single-label multi-class, micro-F1 = accuracy exactly (every error is
simultaneously one FP and one FN). So "micro-F1" would add nothing over accuracy — a detail
worth knowing if challenged.

**Not yet addressed:** class weighting, balanced sampling, per-class thresholds. Say so.

---

## 5. Augmentation

Three policies, increasing strength:

| Policy | Transforms |
|---|---|
| light | RandomResizedCrop(0.8–1.0), HFlip |
| medium | + VFlip, ColorJitter, Rotation(20°) |
| heavy | + RandAugment(n=2, m=9), RandomErasing(p=0.25) |

### The theory

Augmentation is **regularisation via an invariance prior**. You assert that `T(x)` carries the
same label as `x` for transforms `T` in some group, which is equivalent to expanding the
training set along directions you believe are label-preserving. It shrinks the hypothesis
space to functions invariant to `T`, reducing variance at the cost of bias if the invariance
is false.

**Domain justification matters here.** Vertical flip is included because a leaf photographed
on a bench has no canonical up — unlike natural-scene datasets where VFlip is harmful.
ColorJitter is bounded (brightness/contrast/saturation 0.2, hue 0.05) because **hue carries
diagnostic signal**: chlorosis is yellowing, necrosis is browning. Aggressive hue shift would
destroy the label. This is the kind of judgement an interviewer probes for.

RandomErasing/Cutout forces reliance on distributed evidence rather than a single lesion.

**Interaction with §3:** augmentation and leakage are *not* substitutes. Augmenting a leaked
split still leaks — you are augmenting images the model will see again at test time.

---

## 6. Model architecture

Backbones come from `timm` via config, so the same module trains every experiment.

### ResNet50 — baseline

**Residual learning.** A block computes `y = F(x, W) + x`. The identity shortcut means the
layer learns a *residual* rather than a full mapping. The gradient through the block is

  `∂L/∂x = ∂L/∂y · (1 + ∂F/∂x)`

The `1` guarantees gradient flow even when `∂F/∂x → 0`, which is what makes 50+ layers
trainable — it addresses degradation (deeper nets doing *worse* on training error), which is
an optimisation problem, not overfitting.

**Bottleneck blocks:** 1×1 reduce → 3×3 conv → 1×1 expand. The 1×1s cut channel dimensionality
before the expensive spatial convolution, giving depth at manageable FLOPs.

**BatchNorm** normalises pre-activations per channel over the batch, then rescales by learned
γ, β. It smooths the loss landscape and permits higher learning rates. It also matters for
deployment — see §7.2, where BN folding is the root of a real bug.

### ConvNeXt-Tiny — challenger

A ConvNet modernised with transformer-era design decisions: depthwise 7×7 convolutions (large
receptive field, cheap), inverted bottleneck, **LayerNorm instead of BatchNorm**, GELU instead
of ReLU, fewer activation/norm layers per block.

**Why it is a good challenger:** it differs along several axes at once (receptive field,
normalisation, activation), so if it wins, the interesting question is *which* change mattered
— that is what the Week 2 ablations are for. It is also a genuinely different inductive bias,
not just a bigger ResNet, so the two models make *different mistakes*. That matters for
McNemar (§8.1), which only extracts information from disagreements.

### Transfer learning

Both start from ImageNet weights. Early layers learn edges, textures, colour opponency —
largely domain-general. Leaf disease is texture-and-colour driven, so ImageNet features
transfer well. With 38K training images across 38 classes (~1,000 each), training from scratch
would badly overfit.

**Dropout** (0.2 baseline / 0.3 challenger) before the classifier. At train time it zeroes
units with probability p, approximating an ensemble over subnetworks; at inference it scales
by `1/(1-p)` (inverted dropout) so expectations match. The challenger uses more because heavy
augmentation plus a stronger backbone needs more regularisation.

---

## 7. Training

### 7.1 Objective

**Cross-entropy with label smoothing (ε = 0.1).**

Standard CE with one-hot targets is `L = −log p_y`. Its minimum requires `p_y → 1`, which
demands the correct logit diverge to +∞ — driving overconfidence and large weights.

Label smoothing replaces the target with

  `y'_k = (1−ε)·1[k=y] + ε/K`

so the loss is minimised at a *finite* logit gap. For uniform smoothing the optimal gap is

  `log( (K−1)(1−ε) / ε )`

With K=38, ε=0.1: `log(37·0.9/0.1) ≈ 5.8`. Bounded, so the model cannot chase infinite
confidence.

**The trade-off, and it is a real one:** label smoothing improves accuracy and generalisation
but *degrades* calibration in a specific way — it deliberately caps confidence, so predicted
probabilities systematically understate. Since calibration (ECE, temperature scaling) is
planned work, this interaction must be handled: **temperature scaling should be fitted after
training, on the validation set, against the smoothed model.** Anyone asking about calibration
should hear that these two choices interact.

### 7.2 Optimisation

**AdamW.** Adam maintains per-parameter first and second moment estimates:

```
m_t = β₁m_{t−1} + (1−β₁)g_t          v_t = β₂v_{t−1} + (1−β₂)g_t²
m̂_t = m_t/(1−β₁^t)                   v̂_t = v_t/(1−β₂^t)      # bias correction
θ_t = θ_{t−1} − lr · m̂_t/(√v̂_t + ε)
```

Bias correction matters because `m_0 = v_0 = 0` biases early estimates toward zero.

**Why AdamW, not Adam + L2.** In Adam, an L2 penalty enters through the gradient and is then
divided by `√v̂`. Parameters with large historical gradients get *less* effective decay — so
regularisation strength becomes an accident of gradient history. AdamW decouples it:

  `θ_t = θ_{t−1} − lr·(m̂_t/(√v̂_t+ε) + λθ_{t−1})`

The decay term is applied directly, so λ means the same thing for every parameter. This is why
the challenger can meaningfully use `weight_decay=5e-2` — that number is only interpretable
under decoupled decay.

**Cosine annealing:**

  `η_t = η_min + ½(η_max − η_min)(1 + cos(πt/T))`

Smooth decay, slow at both ends. The slow start acts like an implicit warmup; the slow tail
gives many low-LR steps to settle into a minimum. Empirically favours flatter minima, which
correlate with better generalisation.

**Mixed precision (`16-mixed`).** Forward/backward in fp16, master weights in fp32. ~2×
throughput and half the activation memory on tensor-core GPUs. Requires **gradient scaling**:
fp16 has a minimum normal ~6e-5, so small gradients flush to zero. The loss is multiplied by a
scale factor before backward and unscaled before the step, with the factor adapted dynamically
on inf/NaN detection. Lightning handles this, but you should know *why* it is needed.

### 7.3 Selection and stopping

Checkpointing and early stopping both monitor **`val_f1_macro`**, not `val_acc` — see §4.

Early stopping (patience 5) is itself regularisation: it limits effective capacity by halting
before the model fully exploits training-set idiosyncrasies.

**Subtlety worth raising before you are caught by it:** selecting the best checkpoint *on*
validation makes validation-set performance optimistically biased — you have used it for
model selection, so it is no longer a clean estimate. That is exactly why the **test split is
touched only once, at the end**, and why the hypothesis tests in §8 run on test rather than
validation.

---

## 8. Statistical comparison

The heart of the project. Three genuinely distinct questions.

### 8.1 Is the difference real? — McNemar's test

Two models on the same holdout. Build the contingency table over per-image correctness:

```
                  B correct   B wrong
    A correct        n₁₁        n₁₀
    A wrong          n₀₁        n₀₀
```

**Only the discordant cells `n₁₀` and `n₀₁` carry information.** Images both models get right
(or both wrong) say nothing about which is better; they are nuisance parameters. This is the
key insight, and it is why McNemar is far more sensitive than comparing two independent
accuracy figures — it conditions away the shared difficulty of the test set.

Under H₀ (the models are equally accurate — formally, marginal homogeneity), each discordant
pair is equally likely to fall either way:

  `n₀₁ | (n₀₁ + n₁₀) = n ~ Binomial(n, ½)`

**Large samples:**

  `χ² = (|n₀₁ − n₁₀| − 1)² / (n₀₁ + n₁₀)  ~  χ²(1)`

The `−1` is **Edwards' continuity correction** — a discrete binomial is being approximated by a
continuous χ², and without the correction the test is anti-conservative (too many false
positives).

**Small samples (<25 discordant):** the χ² approximation degrades, so we use the exact
binomial

  `p = 2 · P(X ≤ min(n₀₁, n₁₀))`,  `X ~ Bin(n, ½)`

**Why not a paired t-test on per-image correctness?** Correctness is Bernoulli, not normal.
McNemar is the exact conditional test for this design.

**Why not two-proportion z-test?** It assumes independent samples. Our samples are paired —
the same images. Ignoring pairing discards the covariance term and inflates variance
(see §8.2), losing power.

### 8.2 How big, and how uncertain? — Bootstrap

Non-parametric bootstrap for the accuracy difference. Resample n images **with replacement**
10,000 times, recompute `acc(B) − acc(A)` each time, take the 2.5th/97.5th percentiles.

Justification is the **plug-in principle**: the empirical distribution `F̂ₙ` converges to `F`
(Glivenko–Cantelli), so sampling from `F̂ₙ` approximates sampling from `F`. It requires no
normality assumption about the statistic's sampling distribution.

**Critical implementation detail: both models are resampled on the same indices.** This
preserves pairing. Because

  `Var(A − B) = Var(A) + Var(B) − 2·Cov(A, B)`

and the two models are strongly positively correlated (they find the same images hard), the
covariance term is large and the paired interval is *much* narrower. Resampling independently
would throw that away and produce a needlessly wide, less useful interval.

### 8.3 Is it spread out? — per-class paired t-test and Cohen's d

Compute per-class accuracy for both models: 38 paired observations rather than 8,000. Then

  `t = d̄ / (s_d/√n)  ~  t(n−1)`,  where `d = acc_B − acc_A` per class

This asks a **different question** from McNemar: not "are there more wins than losses?" but
"is the improvement distributed across classes, or driven by a few common ones?" A model that
improves only `Orange___Haunglongbing` (5,507 images) could win McNemar decisively while
being worthless on rare diseases.

**Cohen's d_z** (the paired variant) is `d_z = d̄ / s_d` — mean difference in units of the
standard deviation *of the differences*. Note `t = d_z·√n`, which is precisely why p-values
and effect sizes must be reported together: **t grows with sample size, d_z does not.** With
enough data any nonzero difference becomes significant. Effect size is what tells you whether
it matters.

Conventional benchmarks: <0.2 negligible, <0.5 small, <0.8 medium, ≥0.8 large. Rules of thumb,
not laws.

### 8.4 Could we even have detected it? — power

Power = `P(reject H₀ | H₁ true)` = `1 − β`. For a paired t-test with true effect `d_z`, the
statistic follows a **noncentral** t with noncentrality parameter

  `δ = d_z · √n`

and power is the mass beyond the critical values:

  `power = P(T'(ν, δ) > t_{1−α/2,ν}) + P(T'(ν, δ) < −t_{1−α/2,ν})`

We report both tails so the calculation is correct if the effect points the other way.

**Why this is reported alongside every null result:** a non-significant result from an
underpowered test means *we could not tell*, **not** *there is no difference*. Absence of
evidence is not evidence of absence. When power < 0.8 the framework says so explicitly and
computes the n that would be needed.

Worked example from the implementation: a 0.36pp difference across 38 classes gives d_z=0.19,
power 0.209 — and reports that n=217 classes would be needed. The honest conclusion is "this
experiment cannot resolve a difference this small," not "the models are equivalent."

### 8.5 The promotion rule

```
challenger_is_better  ⟺  McNemar p < 0.05
                      AND more discordant pairs favour B
                      AND bootstrap CI excludes 0
```

The **direction check is not redundant**. McNemar is two-sided: a model that is significantly
*worse* also produces p < 0.05. Gating on the p-value alone would promote it. There is a test
for exactly this case.

Requiring the CI to exclude zero adds an effect-magnitude condition on top of significance.
`compare.py` exits non-zero unless all three hold, so CI gates promotion on **evidence**
rather than a raw accuracy delta.

### 8.6 Multiple comparisons — a limitation to own

Running many model comparisons inflates family-wise error: at α=0.05, testing 20 challengers
yields ~64% chance of at least one false positive. This project does **not** currently apply
Bonferroni or Benjamini–Hochberg correction. With a planned sweep (20 runs), that becomes a
real concern. The honest answer is: correction is needed once we start selecting the winner
*from* the sweep, and the holdout must not be reused for each candidate.

---

## 9. Export and quantisation

### 9.1 Why ONNX

Training needs autograd, dynamic graphs, ~2GB of torch. Inference needs none of it. Exporting
to ONNX gives a static graph runnable by a ~50MB runtime, so the serving image stays inside
free-tier limits — and the entire `serving` package is importable without torch.

### 9.2 BatchNorm folding

At inference BN is a fixed affine map, so it can be folded into the preceding convolution:

  `BN(Wx + b) = γ·(Wx + b − μ)/√(σ² + ε) + β`

Collecting terms:

  `W' = γW/√(σ² + ε)`,  `b' = γ(b − μ)/√(σ² + ε) + β`

One fused convolution, no runtime normalisation. This is what `do_constant_folding=True` does
— and it is the root of a real bug (§9.4).

### 9.3 Dynamic INT8 quantisation

Map fp32 to 8-bit integers via an affine transform `r ≈ s·(q − z)`.

For **symmetric** per-tensor weight quantisation (z = 0):

  `s = max|r| / 127`,  `q = clip(round(r/s), −127, 127)`

**Dynamic** means weights are quantised ahead of time, but activation scales are computed at
runtime per batch — no calibration dataset needed, unlike static quantisation. The trade-off
is a small per-inference cost to compute activation ranges, in exchange for robustness to
distribution shift in inputs.

Measured: **32.2MB → 8.1MB**, ~4×, as expected from 32-bit → 8-bit weights.

Accuracy impact is small because weight distributions per layer are roughly unimodal and
centred, so 256 levels capture them well — but it is **not** zero, which is why the test suite
asserts the INT8 and fp32 probability vectors stay within 0.1 max absolute difference.

### 9.4 The two traps

**Trap 1 — the dynamo exporter's graph will not quantise.** torch ≥2.6 defaults to the
`torch.export`-based exporter. Its output fails ORT shape inference during quantisation
(`Inferred shape and existing shape differ`), with or without dynamic axes. We pin
`dynamo=False`.

**Trap 2 — constant folding defeats the quantiser.** Folded conv weights (§9.2) are emitted as
**Constant nodes**, while the dynamic quantiser only rewrites **initializers** — hence
`Expected onnx::Conv_205 to be an initializer`. Two fixes were measured:

| Approach | Float | INT8 |
|---|---|---|
| disable constant folding | 44.7MB | 11.2MB |
| **`quant_pre_process()` then quantise** ✅ | 32.2MB | **8.1MB** |

The second wins because it keeps BN folding *and* quantises — ORT's pre-processing pass
converts those constants into initializers.

### 9.5 The parity gate

After export, identical random inputs go through PyTorch and ONNX Runtime; max |logit
difference| must be < 1e-3 or the export **raises**. Currently ~2.4e-07 (fp32 round-off).

This exists because a silently-wrong graph is the worst failure mode in the pipeline: no
error, no crash, just quietly degraded predictions in production. Operator semantics genuinely
do differ between frameworks — padding conventions, resize kernels, epsilon placement in
normalisation. The gate turns a silent corruption into a loud build failure.

---

## 10. Serving

### 10.1 Numerical stability

Softmax is computed as

  `softmax(z)_i = exp(z_i − max z) / Σ_j exp(z_j − max z)`

Subtracting the max is mathematically an identity (numerator and denominator scale by the same
constant) but numerically essential: `exp(1000)` overflows fp32, `exp(1000 − 1000) = 1` does
not.

### 10.2 Training/serving skew — the real risk

Serving has no torch, so preprocessing is a **NumPy reimplementation** of torchvision's
`eval_transforms`: resize short side to `224·256/224`, centre crop 224, scale to [0,1],
normalise by ImageNet mean/std.

Duplicated logic is a classic production failure: the two implementations drift, accuracy
silently degrades, and nothing errors. `test_preprocess.py` pins them against each other on
real images. Small differences remain (PIL bilinear vs torchvision antialiasing), so the test
asserts a mean-absolute bound rather than exact equality — the honest tolerance given two
different resampling kernels.

### 10.3 Uncertainty — currently a placeholder

We report normalised predictive entropy:

  `H(p) = −Σ_k p_k log p_k`,  normalised by `log K` to give [0, 1]

**Be upfront about its limitation:** this captures only *aleatoric-looking* spread in a single
softmax output. It cannot distinguish "genuinely ambiguous input" from "input far outside the
training distribution" — a confidently wrong out-of-distribution prediction yields *low*
entropy. True epistemic uncertainty needs MC-dropout or an ensemble, which is planned, not
built. Do not oversell this.

### 10.4 Degraded-mode design

If no model file is present, `/health` reports `degraded` and `/predict` returns 503 with a
clear message — rather than the process crashing.

This is deliberate: the container must be deployable and health-checkable **before the first
model exists**. Otherwise CI cannot smoke test the image, and a platform like Render would
crash-loop on startup. Separating "the service is up" from "the model is loaded" is standard
practice for anything with a heavy artifact dependency.

### 10.5 Prometheus metrics

Three instruments, chosen by type:

- `cropguard_requests_total` — **Counter** (monotonic; rate computed by the query engine)
- `cropguard_request_duration_seconds` — **Histogram** (pre-defined buckets)
- `cropguard_prediction_confidence` — **Histogram** labelled by predicted class

Histograms rather than summaries because histogram buckets are **aggregatable across
instances** — you can sum bucket counts from N replicas and compute a global quantile.
Pre-computed quantiles in summaries cannot be averaged meaningfully.

The confidence histogram is the seed of drift detection: a shift in the distribution of
prediction confidence is an early warning that inputs have moved away from the training
distribution — observable without any ground-truth labels.

**Cardinality caution:** labelling by predicted class means 38 label values per metric. Fine
here; labelling by user or image ID would explode the time-series count and take down
Prometheus. Cardinality discipline is a real production concern.

---

## 11. CI/CD — what is actually built

Three jobs on push and PR:

| Job | Does |
|---|---|
| `lint` | `ruff check` + `ruff format --check` |
| `test` | Installs CPU-only torch, runs 81 tests |
| `docker` | Builds the serving image, runs the container, asserts `/health` responds |

**Design points worth defending:**

- **CPU-only wheels** via the PyTorch CPU index. The CUDA build is several GB and would blow
  the runner's disk quota, for zero benefit on a GPU-less runner.
- **The container smoke test asserts `degraded`**, not `healthy` — no model is baked in, so
  `degraded` is the *correct* response. Asserting the true expected state, rather than the
  convenient one, is what makes the test meaningful.
- **A placeholder `classes.json`** is generated in CI when absent, since it is a build input
  produced by the data pipeline.
- **`concurrency` with `cancel-in-progress`** so superseded pushes stop consuming free minutes.

**Not built, do not claim it:** no training job, no model registry gate, no automated
retraining, no deployment step. `compare.py`'s exit code is *designed* to be that gate, but
nothing wires it into a workflow yet.

---

## 12. The hard questions

**"Your test accuracy will be lower than published PlantVillage results. Why should I care
about your model?"**
Published results are mostly measured on leaked splits. I measured the leakage: 74.2% of test
images shared a physical leaf with training. My number is lower because it is measured on a
holdout with zero leaf overlap. The comparison is not like-for-like — and I can produce both
numbers, since `--strategy stratified` is retained precisely to quantify that gap.

**"How do you know your grouped split didn't just make the task artificially hard?"**
It cannot. It removes images from train, but each test image is still a real image of a real
diseased leaf drawn from the same distribution. What changes is only that the model has not
seen *that specific leaf*. That is the definition of generalisation.

**"Your McNemar test says p < 0.001. Is the model better?"**
Significantly different, in a direction. Whether it is *better* also requires the direction
check and an effect size — which is why promotion requires significance in the right
direction plus a CI excluding zero. And with n=8,000 a trivial difference reaches
significance, so I read Cohen's d before deciding it matters.

**"Why is your per-class t-test valid? Per-class accuracies aren't independent."**
A fair objection. They share a model and a training run, so they are not strictly independent
draws. The t-test is a reasonable approximation because the classes are disjoint image sets,
but the assumption is imperfect — which is one reason McNemar (which makes no normality or
independence-across-classes assumption) is the primary test, and the per-class test is
supporting evidence about *distribution* of gains.

**"What happens if leaf-map.json is wrong?"**
Grouping degrades toward the naive split, and leakage returns. I do not have an independent
audit of it. What I can say is that it produced 20,015 groups with zero class-spanning
violations, which is a consistency signal — a random or corrupted mapping would not exhibit
that structure.

**"Your INT8 model differs from fp32. How do you know it's safe to deploy?"**
The test suite bounds max absolute probability difference at 0.1, and the export gates fp32
against PyTorch at 1e-3. But I have **not** yet measured INT8 accuracy on the full holdout —
that is the correct check and it is not done. That is the honest gap.

**"You monitor macro-F1 but train on plain cross-entropy. Isn't that a mismatch?"**
Yes, and it is a real one. The loss is support-weighted while the selection metric is not, so
optimisation is pulled toward common classes while selection rewards balance. Class-weighted
loss or balanced sampling would align them; neither is implemented yet. Currently macro-F1
acts only as a *selection* criterion, not a *training* signal.

**"What's the weakest part of this project?"**
That no model is trained yet, so every number about model quality is hypothetical. After that:
uncertainty is entropy rather than genuine epistemic uncertainty; there is no correction for
multiple comparisons once sweeps begin; and nothing enforces that the serving package stays
torch-free, which is the invariant the whole deployment rests on.
