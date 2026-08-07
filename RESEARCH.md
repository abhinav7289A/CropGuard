# CropGuard — Results

What was measured, what it means, and where it breaks. Every number here is reproducible from
this repo; nothing is estimated or aspirational.

`README.md` says how to run it. `brain.md` is the full derivation. This is the five-minute
version for someone deciding whether the rest is worth reading.

---

## Problem

Identify 38 crop diseases from a leaf photograph, well enough to be trusted, on infrastructure
that costs nothing. The modelling is the easy part — PlantVillage is a saturated benchmark
where ~99% is routine. The work is in the parts that decide whether a 99% number means
anything: an uncontaminated holdout, calibrated confidence, a promotion rule fixed in advance,
and a latency figure measured on the machine that actually serves.

**Dataset:** PlantVillage, 54,305 images, 38 classes, 36× imbalance between largest and
smallest class. Leaves photographed against uniform backgrounds under controlled lighting.

---

## 1. The holdout is the result everything else rests on

PlantVillage contains ~7,600 *distinct physical leaves* photographed ~7 times each. A standard
per-image stratified split scatters those near-duplicates across train and test, so a model can
score well by recognising a leaf it has already seen rather than the disease on it.

**Measured: a naive stratified split leaves 74.2% of test images sharing a leaf with training.**

Rebuilding the split grouped by leaf identity (20,015 groups, zero class-spanning violations)
brings that to 0.0%. The project was built on the expectation that removing the leakage would
lower accuracy, and that the drop would be the honest number.

| Split | Test leakage | Best val macro-F1 |
|---|---|---|
| Naive stratified | 74.2% | 0.9921 |
| **Leaf-grouped** | **0.0%** | **0.9926** |

**It did not drop.** The contaminated split scored marginally *lower*. These are different
validation sets so it is not a clean paired comparison, but the direction contradicts
leakage-inflation and the gap is noise.

The honest reading: PlantVillage is genuinely easy — uniform backgrounds give a model ample
non-leaf signal, so it never needs to memorise leaf identity. Published ~99% figures appear to
be largely real. **The hypothesis was falsified, and that is the finding.** It is reported
rather than quietly dropped, because "the standard split's leakage does not materially inflate
results on this dataset" is a claim nobody had checked, and checking it required building the
grouped split regardless.

---

## 2. Baseline

ResNet50, ImageNet-pretrained, 12 epochs, medium augmentation, label smoothing 0.1, selected on
validation macro-F1. Evaluated once on the 8,125-image leak-free holdout.

| Metric | Value |
|---|---|
| Test accuracy | 0.9911 |
| **Test macro-F1** | **0.9865** |

Read macro-F1. At 36× imbalance, accuracy is dominated by the largest classes.

**Reproducibility, measured rather than asserted.** Training ran on a Colab T4; the checkpoint
was exported and evaluated on a Windows CPU. The two agree to **16 decimal places** — same
accuracy, same macro-F1, across two operating systems and two processor architectures. That
holds because the split is *regenerated* from a seed and a sorted file listing then SHA-256
verified, label order is pinned in `classes.json`, the ONNX export is gated against PyTorch at
<1e-3, and the serving preprocessor is pinned against torchvision by a test. Break any one and
the numbers drift silently.

---

## 3. The A/B test, and the null it produced

A ConvNeXt-Tiny challenger was trained and compared against the deployed baseline on the
identical holdout, image by image.

| | Baseline | Challenger |
|---|---|---|
| Test accuracy | **0.9911** | 0.9908 |
| Test macro-F1 | 0.9865 | **0.9890** |

```
McNemar:            49 discordant for A vs 46 for B, p = 0.8374   -> not significant
Bootstrap accuracy: -0.0004, 95% CI [-0.0027, +0.0020]            -> includes 0
Bootstrap macro-F1: +0.0025, 95% CI [-0.0015, +0.0067]            -> includes 0
VERDICT: no significant improvement demonstrated                   (exit 1)
```

**The gate declined and the challenger was not deployed.** Three things make this more
informative than a bare null:

**The two headline metrics moved in opposite directions.** Accuracy fell 0.0004 while macro-F1
rose 0.0025 — the challenger traded majority-class accuracy for rare-class balance. That
exposed a real inconsistency: the documentation tells readers to trust macro-F1, while the
promotion gate is keyed on accuracy. Nothing surfaced that until a model made the two disagree.

**The per-class test could never have settled it.** Its sample size is the *class count*: at
d = 0.200 it achieves power 0.225 and would need n = 198 classes for 0.8. PlantVillage has 38.
No amount of additional labelling moves that number — it is a structural ceiling, and a test
that cannot reach adequate power on its own problem cannot produce an informative null.

**So a macro-F1 bootstrap over images was added**, resampling the 8,125 holdout images and
recomputing the macro statistic on each resample. It is powered by the holdout rather than by
the class count, and it returned an interval spanning zero. The challenger is not better on the
metric that appeared to favour it, and the null went from "we could not tell" to a tight bound.

**The gate was deliberately not widened** to accept a macro-F1 win. The rule was fixed before
any challenger existed; loosening it after seeing which metric moved would be choosing the test
that returns the desired answer. Macro-F1 is measured, printed, and allowed to contradict the
verdict in public. A unit test pins that behaviour so a later refactor cannot soften it.

**What the comparison actually rests on:** 8,004 images both models got right and 26 both got
wrong carry no information about which is better. Every claim here rests on the 95 discordant
images. Stating that is more useful than letting a reader see "n = 8,125" and assume otherwise.

**What it cannot say:** seven settings differ between the two configs — architecture,
augmentation, epochs, batch size, learning rate, weight decay, dropout. The answer is about
*configurations*, not architectures. "ConvNeXt is no better than ResNet50" is not supported.

---

## 4. Calibration

A 99%-accurate model can still be badly calibrated, and this one was — in the *under*-confident
direction, which is the less common one.

| | ECE | MCE | Brier | Mean confidence |
|---|---|---|---|---|
| Raw softmax | 0.0895 | 0.278 | 0.0217 | 0.9020 |
| **Temperature-scaled (T = 0.591)** | **0.0036** | 0.338 | **0.0138** | 0.9946 |

**96% ECE reduction, accuracy provably unchanged** — dividing logits by a positive scalar cannot
reorder them. T < 1 means the model needed *sharpening*, exactly as label smoothing predicts:
ε = 0.1 over K = 38 caps achievable softmax output at (1−ε) + ε/K = 0.9026, and the model sat at
0.9020 while being 99.11% accurate.

**MCE got worse, and that is reported rather than hidden.** MCE is the worst single bin.
Calibration moved 8,042 of 8,125 predictions into one high-confidence bin, leaving sparse bins
of 20–30 predictions where two or three errors produce a large gap. ECE is population-weighted
and improves; MCE is a max over bins including the tiny ones. They measure different things.

**A confirmation worth noting:** the ConvNeXt challenger, a different architecture fitted
independently on its own validation split, landed on T = 0.5914 against the baseline's 0.591.
The correction is a property of the loss function, not of the backbone.

---

## 5. Where the errors are

~72 misclassifications in 8,125, concentrated in two biologically plausible pairs:

- **Corn: Cercospora / Gray leaf spot ↔ Northern Leaf Blight** (F1 0.940 / 0.970)
- **Tomato: Early blight ↔ Late blight** (F1 0.944 / 0.964)

Plus potato→tomato Late blight, which is literally the same pathogen on a different host. Both
pairs are hard for human agronomists. The confident errors *cluster* on these distinctions
rather than scattering, which is what a model that learned disease morphology should look like.

**A caveat before anyone else spots it:** `Potato___healthy` shows recall 0.833 — but its
support is 24 images, so that is four mistakes with a Wilson interval of [0.641, 0.933]. Ten
classes have fewer than 100 test images. Their per-class metrics carry intervals too wide to act
on, and are reported with those intervals rather than as point estimates.

---

## 6. Serving: two measurements that overturned assumptions

**Dynamic INT8 quantisation was 75× *slower*, not faster.**

| | fp32 | dynamic INT8 | static INT8 |
|---|---|---|---|
| Size | 94.3 MB | 23.8 MB | 23.8 MB |
| Latency (i5-12450H) | **24.4 ms** | 1569 ms | 78.0 ms |
| Conv operator | `Conv` | `ConvInteger` | `QLinearConv` |

`quantize_dynamic` rewrites every convolution into `ConvInteger`, for which ONNX Runtime's CPU
backend has no optimised kernel. Dynamic quantisation suits MatMul-dominated architectures —
Transformers, RNNs — not CNNs. Static quantisation emits `QLinearConv` and is 20× faster than
dynamic and accuracy-neutral, but still 3.2× slower than fp32 on a CPU without VNNI
instructions. **INT8 is not universally faster; it is faster on hardware with INT8
instructions.** fp32 is what serves.

The lesson is not about quantisation. Size reduction had been verified and accuracy delta had
been verified; latency had been *assumed* to follow. It was the one unmeasured claim, and the
one that mattered.

**Production latency is 146× the laptop figure.**

| Environment | Median inference |
|---|---|
| Laptop (i5-12450H, full cores) | 24 ms |
| **Render free tier (0.1 vCPU)** | **3,504 ms** |

Nothing is broken — Render's free tier allocates a tenth of a core, and ResNet50 at 224×224 is
~4 GFLOPs. A latency figure without its hardware is meaningless, and the only benchmark that
counts is the one taken on the deployment target. This redirected optimisation away from
quantisation and toward backbone size: MobileNetV3 is ~18× fewer FLOPs, and serving a
25.6M-parameter network to classify leaves a 5M-parameter one could handle is the real
inefficiency.

---

## 7. Monitoring without labels

Accuracy cannot be measured in production — nobody labels the leaves. What is observable is
distribution: **PSI** on the confidence distribution, **total variation distance** on the
predicted class mix.

**The finding that shaped the design:** split the test set into two random halves and compare
them. Same model, same data, nothing to find by construction.

```
PSI  0.0028  (stable)        TVD  0.0576  (no shift)
chi-squared  p = 8.35e-09    <- "wildly significant"
```

The chi-squared p-value reports p ≈ 1e-8 on **two random halves of identical data**. It is not
broken; it answers "could this difference be zero?", and at n = 4,000 the answer is essentially
always no. Significance measures *detectability*, which grows with sample size regardless of
whether the effect matters. So the verdict keys on **effect sizes**, which are invariant to
sample count. A monitor whose sensitivity depends on traffic volume will page at 3am for
nothing, get muted, and then miss the real event.

Two signals rather than one, because they mean different things: confidence shifting means the
model is hedging on off-distribution inputs; class mix shifting means the population served
changed. Neither proves degradation — they are reasons to investigate, not verdicts.

---

## 8. Limitations

Stated here rather than left for a reader to find.

1. **The model has only ever seen lab photographs.** 99.11% on PlantVillage says nothing about
   a phone picture taken in a field, and the literature reports large drops there. **This is
   the weakest part of the project.** Until it is measured, the accuracy figure describes a
   benchmark, not the problem the project claims to address.
2. **`uncertainty` is predictive entropy, not epistemic uncertainty.** It measures spread in one
   softmax, so it cannot flag an input unlike anything in training. An out-of-distribution image
   can produce a confidently wrong answer with low entropy.
3. **No multiple-comparisons correction.** With one challenger this is fine. It becomes a real
   problem the moment several are tested against the same holdout.
4. **The gate is keyed on accuracy while the documentation says to read macro-F1.** The honest
   fix is to pre-register macro-F1 as primary *before* the next challenger trains — a decision
   about the next experiment, not a reinterpretation of this one.
5. **Ten classes have fewer than 100 test images.** The validation gate checks total samples per
   class; for per-class test metrics with usable error bars it should check split size.
6. **Macro-F1 is the selection metric but the loss is plain cross-entropy**, which is
   support-weighted. Optimisation pulls toward common classes while selection rewards balance.
   Class-weighted loss would align them; it is not implemented.
7. **Degradation cannot be confirmed in production.** Drift detection sees inputs move, not
   accuracy fall. Confirming decay needs labels, which needs a `/feedback` endpoint, which is
   not built.

---

## 9. Deliberately not built

Absent by decision, not oversight — each would have added surface without adding evidence.

| | Why not |
|---|---|
| **DVC** | The dataset is one immutable HuggingFace release and the split is regenerated from a seed then hash-verified. Versioning 2.2 GB of unchanging images adds a remote to configure and nothing not already reproducible. |
| **MLflow registry** | Two models. A registry solves discovery and lineage across many; `configs/models.json` plus the Hub does the same job here without a server. |
| **Sweeps / Optuna** | Free GPU hours are the binding constraint, and a 20-run sweep on a benchmark at 99.11% would spend them chasing 0.3 points. The informative use of that compute is a controlled ablation isolating one of the seven config differences. |
| **Automated retraining** | It needs a signal that the model degraded. Input drift is not that signal. Wiring a retrain trigger to it would automate a decision the evidence cannot support. |
| **Grafana / AlertManager** | `/metrics` exports Prometheus format and the drift module computes the numbers. Dashboards over one free-tier instance serving demo traffic would be decoration. |

---

## 10. Next

In order of what each would actually add:

1. **Field-photograph evaluation.** The only measurement that would change what this project can
   claim. Everything else is refinement.
2. **`/feedback` endpoint**, closing the loop from prediction to label to confirmed degradation.
3. **A controlled ablation** isolating one variable, so an architecture claim becomes possible.
4. **MobileNetV3**, addressing the real serving cost rather than the one quantisation targets.

---

## Summary — the four claims worth making

Each is measured, reproducible from this repo, and survives being probed.

**1. A leakage audit that falsified its own hypothesis.** 74.2% of test images in the standard
PlantVillage split share a physical leaf with training. Rebuilt grouped by leaf identity;
accuracy did not fall. Reported as a negative result rather than dropped (§1).

**2. A promotion gate that declined its only challenger.** McNemar p = 0.837, accuracy CI
[-0.0027, +0.0020], macro-F1 CI [-0.0015, +0.0067]. When the two metrics disagreed, the
per-class test turned out to be *structurally* underpowered — its n is the class count, so 0.8
power needs 198 classes against the dataset's 38 — so an image-level macro-F1 bootstrap was
added rather than the gate being widened to admit the result it wanted (§3).

**3. Calibration with the failure mode reported.** ECE 0.0895 → 0.0036 via temperature scaling,
accuracy provably unchanged. MCE got *worse*, and both are reported because they measure
different things (§4).

**4. Two performance assumptions overturned by measurement.** Dynamic INT8 quantisation ran 75×
slower than fp32; production latency was 146× the laptop benchmark. In both cases the
unmeasured claim was the one that mattered (§6).

The honest framing of the whole project: the modelling is unremarkable, because the benchmark
is saturated. What is defensible is the measurement discipline around it — a holdout you can
trust, a promotion rule fixed before the data was seen, calibration applied in the serving path
rather than only in evaluation, and limitations stated in §8 before anyone has to find them.
