# CropGuard — Explanation Log

A living record of **what each file does, why it works the way it does, and what changed**.
The README tells a visitor how to run the project; this file tells *you* why it is built this
way, including the decisions that are not obvious from reading the code.

Every entry is dated. When behaviour changes, the old explanation is not deleted — it is
marked superseded, so the reasoning trail stays intact.

**Last updated:** 2026-08-06

---

## 1. Where the project stands

| | |
|---|---|
| **Critical path** | Baseline trained, deployed, calibrated; challenger trained and A/B tested |
| **Baseline** | ResNet50, **99.11% test accuracy / 0.9865 macro-F1** on the leak-free holdout |
| **Challenger** | ConvNeXt-Tiny, 99.08% / **0.9890 macro-F1** — gate declined, not promoted |
| **Tests** | 181 passing, ruff clean |
| **Dataset** | 54,305 images / 38 classes at `C:\cropguard-data` (outside OneDrive) |
| **Repo** | github.com/abhinav7289A/CropGuard |
| **Live API** | https://cropguard-api-w9ch.onrender.com |
| **Models** | `XiElonMAsk/cropguard-models` on HF Hub |

**Not yet built:** EDA notebook, duplicate detection, DVC, MLflow registry, hyperparameter
sweeps, augmentation/transfer-learning ablations, `/feedback` endpoint, Grafana, load testing,
field-photo (lab vs. real-world) evaluation.

Most of those are now *deliberate* omissions with stated reasons — see "Deliberately not
built" in the README. The one that is a genuine gap rather than a choice is field-photo
evaluation.

### The headline finding so far

The leaf-grouped split removed **74.2% test leakage** — and accuracy did **not** drop. Best
validation macro-F1 was **0.9926 on the grouped split** versus **0.9921 on the naive one**, so
the contaminated split scored marginally *lower*. Different validation sets, so not a clean
comparison, but the direction is the opposite of what leakage-inflation predicts and the gap
is noise-level.

The honest reading: **PlantVillage is genuinely an easy dataset**, not one whose published
~99% figures are mostly a leakage artifact. The grouping work was still correct — you cannot
know that without measuring, and every downstream statistic depends on a trustworthy holdout —
but the expected inflation is not there. Recorded because the original hypothesis in §3.4 was
that accuracy would fall, and it did not.

---

## 2. Repo map

### Configuration

| File | What it does |
|---|---|
| `pyproject.toml` | Package metadata and **five** dependency groups: `data`, `train`, `serve`, `eval`, `dev`. The split is deliberate — see §3.1. |
| `configs/base.yaml` | Shared defaults: seed, image size, split fractions, training hyperparameters, W&B project. |
| `configs/resnet50_baseline.yaml` | Baseline experiment. ResNet50, medium augmentation, 12 epochs. |
| `configs/convnext_tiny.yaml` | Challenger. ConvNeXt-Tiny, heavy augmentation, 30 epochs, higher weight decay. |
| `configs/classes.json` | **Generated** by the split. The single source of truth for label order. |
| `src/cropguard/config.py` | YAML loader with `extends` inheritance and env overrides. See §3.2. |

### Data pipeline — `src/cropguard/data/`

| File | What it does |
|---|---|
| `download.py` | Fetches `data.zip` from HuggingFace and extracts it into an ImageFolder tree. See §3.3. |
| `validate.py` | Integrity, resolution and class-balance gates. Exits non-zero so CI can fail on bad data. |
| `groups.py` | Derives a **leaf group** per image. The basis of the leakage-free split. See §3.4. |
| `split.py` | Train/val/test assignment, grouped or naive-stratified. See §3.4. |
| `dataset.py` | `torch.utils.data.Dataset` reading images via the persisted `splits.json`. |
| `transforms.py` | Augmentation policies: `light`, `medium`, `heavy`. |

### Training — `src/cropguard/training/`

| File | What it does |
|---|---|
| `module.py` | LightningModule wrapping a timm backbone. Tracks accuracy and macro-F1. |
| `datamodule.py` | Builds the three dataloaders from the persisted split. |
| `train.py` | CLI entrypoint: checkpointing, early stopping, logger selection. See §3.6. |

### Serving — `src/cropguard/serving/`

| File | What it does |
|---|---|
| `app.py` | FastAPI app: `/predict`, `/health`, `/metrics`. See §3.8. |
| `model_loader.py` | ONNX Runtime inference + the NumPy preprocessor. **No torch.** See §3.7. |
| `onnx_export.py` | Checkpoint → ONNX, gated by a PyTorch parity check. See §3.7. |
| `quantize.py` | Static INT8 quantisation with calibration on the validation split. |

### Evaluation — `src/cropguard/evaluation/`

| File | What it does |
|---|---|
| `hypothesis.py` | McNemar, bootstrap CI, paired t-test, Cohen's d, power analysis. See §3.9. |
| `predict.py` | Runs a model over a split, saves per-image predictions to `.npz`. |
| `compare.py` | A/B comparison CLI. Exits non-zero unless the challenger genuinely wins. |
| `calibration.py` | ECE, MCE, Brier, reliability curves, temperature scaling. |
| `calibrate.py` | CLI: fit T on validation, report calibration on test. |
| `errors.py` | Per-class metrics with Wilson intervals, confusion pairs, confident mistakes. |

### Infrastructure

| File | What it does |
|---|---|
| `Dockerfile` | CPU-only serving image. Installs `.[serve]` only, so no torch. |
| `.dockerignore` | Keeps data, checkpoints and tests out of the build context. |
| `.gitignore` | Ignore rules — note the anchoring bug described in §4.3. |
| `.github/workflows/ci.yml` | Three jobs: lint, test, and a Docker build + container smoke test. |
| `scripts/fetch_model.py` | Pulls model weights from HF Hub at container start if not baked in. |
| `notebooks/01_train_baseline.ipynb` | End-to-end baseline training. Detects Lightning AI vs Colab and sets paths accordingly. |
| `notebooks/02_challenger_ab_test.ipynb` | Trains ConvNeXt-Tiny and runs the paired A/B against the deployed baseline. |
| `notebooks/upload_to_hf.py` | Publishes the ONNX graphs and a generated model card to HF Hub. |
| `render.yaml` | Render Blueprint for the deployed API. |
| `src/cropguard/monitoring/drift.py` | PSI, KS, TVD drift detection over confidence and class mix. |
| `app/streamlit_app.py` | Demo UI — upload a leaf, see prediction, confidence, latency. |

### Tests worth knowing about

| File | Guards |
|---|---|
| `test_import_isolation.py` | `serving`/`evaluation` never import torch — the invariant the free-tier deploy rests on. |
| `test_preprocess.py` | The NumPy serving preprocessor against torchvision. Guards training/serving skew. |
| `test_data_groups.py` | Group integrity and stratification of the leaf-grouped split. |
| `test_onnx_export.py` | The exported graph stays quantizable — the trap in §3.7. |
| `test_hypothesis.py` | Statistics checked against textbook values, scipy, and published power tables. |
| `test_quantization.py` | The quantised graph emits `QLinearConv`, never `ConvInteger`. |
| `test_train_logger.py` | A W&B failure degrades to CSV instead of killing a training run. |

---

## 3. How things actually work

### 3.1 Why dependencies are split five ways

Render's free tier is memory- and disk-constrained. A torch install is ~2GB; ONNX Runtime is
~50MB and does everything inference needs. So the packages are import-isolated by environment:

- `cropguard.serving` never imports torch — it must stay installable with `.[serve]` alone
- `cropguard.evaluation` never imports torch either, so statistics can run anywhere including CI
- `cropguard.training` is the only torch-dependent subpackage

`src/cropguard/__init__.py` is kept free of heavy imports for the same reason. **If you ever
add `import torch` to `serving/` or `evaluation/`, the deployment breaks** — and it would break
only in production, since every dev machine has torch installed. `test_import_isolation.py`
enforces it by importing those modules in a subprocess with torch blocked.

### 3.2 Config inheritance

A child config declares `extends: base.yaml` and is **deep-merged** onto its parent, so it can
override `train.lr` without discarding the rest of the `train` block. A shallow merge would
silently drop `optimizer`, `scheduler`, `early_stopping_patience` and so on — a nasty failure
because training still runs, just with different settings than you think.

`CROPGUARD_DATA_DIR` overrides `data.root` from the environment. This exists because the
dataset must live outside OneDrive (see §4.4).

### 3.3 Data acquisition — and why it is not `load_dataset`

**Superseded approach (2026-07-31):** originally called
`load_dataset("mohanty/PlantVillage", "color")`. This fails with
`BuilderConfig 'color' not found. Available: ['default']`.

Two causes stacked:

1. The upstream repo is a **script-based dataset** — a `plant_village.py` loader plus a single
   2.2GB `data.zip`. The `color` config only ever existed inside that script.
2. `datasets` v4 **removed loading-script support** entirely. So the script is ignored, the
   repo is treated as a plain data-file directory, and only a `default` config is reported.

**Current approach:** skip `datasets` altogether. `huggingface_hub.hf_hub_download` fetches
`data.zip`, and we extract `raw/color/<class>/<file>` directly into the ImageFolder layout.
Fewer moving parts, no 54K-image decode/re-encode cycle, and it pins us to the exact bytes
upstream published.

The archive also contains files worth keeping, saved to `<data_root>/reference/`:

- `color_train.txt` / `color_test.txt` — the official published split, for benchmark comparison
- `leaf-map.json` — **the leaf grouping map**, which turned out to matter enormously (§3.4)

### 3.4 The split — the most important design decision in the project

**The problem.** PlantVillage contains 54,305 images but only **~7,600 distinct physical
leaves** — roughly 7 photographs of each leaf from different angles. A per-image stratified
split scatters those near-duplicates across train and test. The model memorises a leaf during
training and "recognises" it at test time.

**Measured on the naive split:**

```
val:  6,033 / 8,146 images (74.1%) shared a leaf with train
test: 6,045 / 8,146 images (74.2%) shared a leaf with train
```

Restricted to images whose leaf ID resolves (76% of the test set), **98% were leaked**. This is
why PlantVillage papers casually report ~99% accuracy. That number is close to meaningless.

It also poisons everything downstream: an A/B test on a leaked holdout measures which model
memorises better, not which generalises better. Every statistic in §3.9 would be describing
the wrong thing.

**The fix — grouped assignment.** Two facts made a clean algorithm possible:

- Leaf groups **never span classes** (verified: 0 of 20,015)
- The smallest class still has 37 groups — plenty to divide three ways

So each class is packed independently: walk its groups largest-first, and hand each whole
group to whichever split is furthest below its target count. This is the classic
longest-processing-time heuristic. It copes with group sizes ranging from 1 to 33 images.

**Result:**

| Metric | Naive | Grouped |
|---|---|---|
| test images sharing a leaf with train | 74.2% | **0.0%** |
| groups spanning a split boundary | — | **0 of 20,015** |
| mean per-class \|test_frac − 0.15\| | — | **0.0012** |
| split sizes | 38,013 / 8,146 / 8,146 | 38,008 / 8,172 / 8,125 |

Stratification is essentially unharmed — the worst class lands at 0.158 against a 0.150 target.

**Caveat worth remembering:** ~24% of images do not resolve to a real leaf ID and fall back to
a class-scoped filename identifier. Those form singleton groups. So grouping is imperfect at
the margins — but 0% measured leakage versus 74% is not a close call.

`--strategy stratified` is retained **only** so the gap can be quantified and written up. It
must never produce a headline metric.

### 3.5 Class imbalance

The dataset is imbalanced **36.2×** — `Potato___healthy` has 152 images,
`Orange___Haunglongbing` has 5,507. Consequences:

- **Accuracy is a poor metric here.** A model ignoring the 5 rarest classes still scores well.
  This is why `module.py` monitors **macro-F1**, which weights every class equally.
- Checkpoint selection and early stopping both key off `val_f1_macro`, not `val_acc`.
- Not yet addressed: class weighting, balanced sampling, or per-class thresholds.

### 3.6 Training

- **Backbone** comes from timm via config, so the same module trains every experiment.
- **Macro-F1 drives** checkpointing and early stopping (see §3.5).
- **Resumable by design.** `save_last=True` writes every epoch; `--resume` continues after a
  Colab or Lightning AI disconnect — free-tier sessions get killed routinely.
- **W&B is a soft dependency.** *Superseded (2026-07-31):* `train.py` previously imported
  `WandbLogger` at module level, so training crashed with `ModuleNotFoundError` when wandb was
  absent — despite the docstring promising it was optional. Now `build_logger()` falls back to
  `CSVLogger`, so CI and offline machines run the identical entrypoint rather than a divergent
  code path.

### 3.7 ONNX export — three separate traps

**Trap 1: the parity check exists for a reason.** After export, the same random inputs go
through PyTorch and ONNX Runtime, and max |logit difference| must be < 1e-3. A silently wrong
graph would otherwise reach production. Currently measures ~2.4e-07.

**Trap 2: torch's dynamo exporter produces an unquantizable graph.** torch ≥2.6 defaults to
the new `torch.export`-based exporter. Its output fails ONNX Runtime's shape inference during
INT8 quantization:

```
InferenceError: Inferred shape and existing shape differ in dimension 0: (512) vs (3)
```

This happens with `dynamic_axes`, with `dynamic_shapes`, and with no dynamic axes at all. We
therefore pin `dynamo=False` (the legacy TorchScript exporter). Revisit when the dynamo path
quantizes cleanly.

**Trap 3: constant folding defeats the quantizer.** With the legacy exporter and default
`do_constant_folding=True`, BatchNorm gets folded into the preceding Conv — and the folded
weights are emitted as **Constant nodes**, whereas the dynamic quantizer only rewrites
**initializers**:

```
ValueError: Expected onnx::Conv_205 to be an initializer
```

Two possible fixes were measured:

| Approach | Float | INT8 |
|---|---|---|
| `do_constant_folding=False` | 44.7MB | 11.2MB |
| **`quant_pre_process()` then quantize** ✅ | 32.2MB | **8.1MB** |

We use the second: ORT's own pre-processing pass resolves those constants into initializers,
so BN folding is retained *and* quantization works. Four times smaller, which is what keeps
the serving image inside the free tier.

**Why INT8 matters:** Render's free tier is tight. 8.1MB versus 32.2MB is the difference
between a comfortable deploy and fighting the memory limit.

### 3.8 Serving

- **Starts `degraded`, never crashes.** If no model file is present, `/health` reports
  `degraded` and `/predict` returns 503 with a clear message. This lets the container be
  deployed and health-checked *before* the first model exists — otherwise CI could not smoke
  test it, and Render would crash-loop.
- **Validation runs before inference**, so bad uploads are rejected cheaply: JPEG/PNG only,
  ≥128px short side, ≤10MB.
- **Preprocessing is a NumPy reimplementation** of torchvision's `eval_transforms`, because
  torch is not installed here. That duplication is a genuine risk — training and serving could
  drift apart silently and degrade accuracy with no error anywhere. `test_preprocess.py` pins
  the two against each other. **Do not delete that test.**
- **`uncertainty` is currently normalized predictive entropy**, a placeholder. MC-dropout and
  ensemble variance are planned.

### 3.9 Statistical comparison

The module answers "is B better than A?" properly, which means three separate questions:

1. **Is the difference real?** McNemar's test on per-image correctness. Only *discordant*
   pairs carry information — images both models get right say nothing about which is better.
   Uses the exact binomial when there are fewer than 25 discordant pairs (where the
   chi-squared approximation misbehaves), otherwise chi-squared with Edwards' continuity
   correction (without which the test is anti-conservative).
2. **How big is it, and how uncertain?** Bootstrap CI over 10,000 paired resamples. Both
   models are resampled on the *same* indices, preserving pairing and tightening the interval.
   Plus a per-class paired t-test with Cohen's d_z, asking whether gains are spread across
   classes or concentrated in the common ones.
3. **Could we even have detected it?** Power analysis via the noncentral t. Reported alongside
   every null result, because "not significant" from an underpowered test means *we could not
   tell*, not *there is no difference*.

**Promotion requires significance in the correct direction AND a CI excluding zero.** A
significant p-value alone would happily declare a *worse* model the winner — there is a test
for exactly that case. `compare.py` exits non-zero otherwise, so CI can gate promotion on
evidence rather than a raw accuracy delta.

Correctness is checked against **independent** references, not just internal consistency:
the textbook McNemar table (χ²=4.2667, p=0.0389), a hand-computed exact binomial (2·56/1024),
scipy's own `ttest_rel`, and the published n=34 for d=0.5 at 80% power.

---

## 4. Decisions and gotchas

### 4.1 `configs/classes.json` is generated, but committed

Written by `split.py`, and it defines label order for training, ONNX export **and** the API.
If these three ever disagree, predictions silently map to the wrong disease names — no error,
just wrong answers. It is committed because the Dockerfile needs it at build time.

**Consequence:** re-running the split on a different dataset rewrites this file. If a trained
model is already deployed against the old ordering, that model is now mislabelled.

### 4.2 Evaluation runs through the ONNX graph, not the torch checkpoint

`predict.py` deliberately uses the exported ONNX model and the *serving* preprocessor, so the
statistics describe what production actually does rather than a torch-side approximation.

### 4.3 The `.gitignore` anchoring bug

The original rule was `data/`. Git patterns without a leading slash match at **any depth**, so
it also matched `src/cropguard/data/` — silently excluding the entire data pipeline from
version control. Nothing errors; the files just never appear in `git status`.

Fixed by anchoring: `/data/`, `/models/`, plus `/checkpoints/` and `/logs/` which were not
ignored at all (training would have offered 268MB of checkpoints for commit).

**Lesson:** after adding an ignore rule, run `git status --short` and confirm the file count
is what you expect.

### 4.4 Keep the dataset out of OneDrive

The repo lives under OneDrive. The dataset must not: 54K small files sync appallingly and can
be locked mid-write during training, which is why `CROPGUARD_DATA_DIR` points at
`C:\cropguard-data`.

*Correction (2026-07-31):* an earlier version of this section also blamed OneDrive for the
local `.git` directory being reset to empty on 2026-07-31. That was wrong — the directory was
deleted deliberately, by hand, to re-push the repository. OneDrive was not involved and the
repo source tree is fine where it is. Recorded here because a maintenance log that
misattributes a cause is worse than no log at all.

---

## 5. Change log

### 2026-08-06 — The A/B ran, the gate said no, and the two metrics disagreed

The challenger notebook was executed on Colab. **ConvNeXt-Tiny: 0.9908 accuracy, 0.9890
macro-F1** against the baseline's 0.9911 / 0.9865. McNemar found 49 discordant pairs favouring
the baseline against 46 favouring the challenger (p = 0.837), the accuracy bootstrap CI spans
zero, and `compare.py` exited 1. **The challenger was not promoted, and the baseline still
serves traffic.**

The result is more interesting than a plain null, because the two headline metrics moved in
opposite directions: accuracy down 0.0004, macro-F1 up 0.0025. The challenger traded a little
majority-class accuracy for rare-class balance, and on a 36x-imbalanced problem macro-F1 is
the metric §0 of `brain.md` tells the reader to trust.

**What was missing to settle it.** Neither existing test could. The accuracy bootstrap targets
the wrong statistic. The per-class t-test does address balance, but its sample size is the
*class count* - 38 classes at d = 0.200 gives power 0.225, and 0.8 would need 198. That is a
structural ceiling: no amount of extra labelling fixes it, because the dataset has 38 diseases.

So `bootstrap_macro_f1_difference` was added. It resamples the 8,125 **images** and recomputes
macro-F1 on each resample, so it is powered by the holdout rather than by the class count, and
it targets exactly the quantity being claimed. Macro-F1 is not a per-image mean, so the CI has
to come from recomputation rather than arithmetic on a correctness vector - which is also why
it needs the predicted classes, not just per-image correctness: recall follows from labels and
correctness, but precision needs to know *which* wrong class was predicted.

**The gate was deliberately not widened.** It still requires a significant McNemar result in
the challenger's favour with a bootstrap CI excluding zero - on accuracy. Changing the rule to
accept a macro-F1 win *after* seeing which metric moved would be choosing the test that gives
the answer you want. Macro-F1 is measured, printed, and allowed to contradict the verdict;
`test_the_gate_stays_keyed_on_accuracy_even_when_macro_f1_prefers_the_challenger` pins that
behaviour so a later refactor cannot quietly soften it.

**Also shipped:** `.github/workflows/promotion-gate.yml` finally wires `compare.py`'s exit code
into CI - it pulls both prediction files from the Hub, runs the gate, and publishes the report
whether or not the answer is yes, because a gate that only leaves a trace when it approves is
not auditable. And `configs/models.json` plus a rewritten Streamlit panel let a visitor pick a
model, compare two side by side on one image, and toggle calibration on and off.

**Corrections made at the same time.** The README's worked example of `compare` output was
*illustrative data* invented before any challenger existed, and it read as a real win
(`VERDICT: challenger is significantly better`). It is now the measured null. The confound
count in the notebook and change-log entry below said five; the YAMLs differ in **seven**
(batch size 64->48 and dropout 0.2->0.3 were missed). `convnext_tiny.yaml` claimed its values
were "refined after the W&B sweep" - there is no sweep. Each of these was a claim that would
not have survived being checked in an interview.

### 2026-08-03 (later) — Challenger notebook, and being careful about what it can claim

`notebooks/02_challenger_ab_test.ipynb`: trains ConvNeXt-Tiny, pulls the **deployed** baseline
from HF Hub rather than retraining it, scores both on the identical holdout, and runs the
paired comparison.

**The confound, stated up front in the notebook.** The two configs differ in seven ways at
once - architecture, augmentation, epochs, batch size, learning rate, weight decay and
dropout. So the experiment answers *"which configuration should ship?"*, which is legitimate
and useful. It does **not** answer *"is ConvNeXt better than ResNet50"*, because seven
variables moved together and no single one can be credited. Isolating any one needs a
controlled ablation, which is not built.

*Correction (2026-08-06):* this entry and the notebook both originally said five. Batch size
(64 -> 48) and dropout (0.2 -> 0.3) were in the YAML and not in the table. A confound count
that undercounts is worse than none, since it is offered as the honest caveat.

This is worth being explicit about because the overclaim is the natural thing to say and
exactly what an interviewer probes.

**Why the baseline is pulled from HF Hub rather than retrained:** the comparison should be
against the artifact actually serving traffic, not a fresh model that happens to share its
config. Retraining would also introduce run-to-run variance as a sixth confound.

**Pairing requires an identical holdout**, so the notebook asserts the split hash before
training. McNemar and the bootstrap compare image by image - that is what makes them more
sensitive than comparing two accuracy numbers, and it is void if the two models saw different
test sets.

**Expect a null.** The baseline sits at 99.11% on a saturated benchmark with 0.9% of headroom.
A framework that declines to promote is doing its job, and the notebook says so rather than
implying a win is the goal.

Verified before shipping: `convnext_tiny.yaml`'s `experiment_name` matches the checkpoint path
the notebook globs, and `compare.py` runs on real-format prediction files and exits 1 on a
null - so the CI gate behaves.

### 2026-08-03 — Drift detection, and a lesson about p-values

`src/cropguard/monitoring/drift.py`. The constraint that shapes it: **you cannot measure
accuracy in production**, because nobody labels the leaves. What is observable is distribution.

**PSI** on the confidence distribution (inputs drifting off-distribution makes the model hedge)
and **total variation distance** on the predicted class mix (the population served changed).
Bin edges come from the *reference* quantiles, never pooled — pooled edges let the current
distribution redefine the yardstick it is measured against.

**The p-value finding, which is the interesting part.** Split the test set into two random
halves. Same model, same data, nothing to find by construction:

```
PSI 0.0028 (stable)     TVD 0.0576 (no shift)
chi-squared p = 8.35e-09   <- "wildly significant"
```

The chi-squared p-value reports p ~ 1e-8 on two random halves of identical data. It is not
broken - it answers "could this difference be zero?", and at n=4,000 the answer is essentially
always no. So the verdict keys on **effect sizes**, which do not move with sample count. A
monitor whose sensitivity scales with traffic volume pages at 3am for nothing, gets muted, and
then misses the real event.

**A design gap the testing exposed.** The first version keyed the verdict on PSI alone. Run
against tomato-only traffic, class mix collapsed to a single crop while confidence held
perfectly steady - and the verdict said "no drift". Confidence and class mix are now separate
signals, because they mean different things: one says the model is struggling, the other says
the world moved.

**A caveat that bit us directly.** Comparing my class-balanced validation sample against the
naturally-imbalanced test split reported TVD 0.268 and chi-squared p = 5.7e-256 - entirely an
artefact of how I sampled, not drift. **Reference and current windows must be sampled the same
way**, and the failure is silent.

**What it does not do:** none of this proves the model got worse. Input drift means the world
moved, which may or may not hurt. These are triggers to investigate, not verdicts. Confirming
real degradation needs labels, which means the /feedback endpoint, which is not built.

Tests: 143 -> 166.

### 2026-08-02 (quick wins) — Calibration reaches production, and a miscount corrected

**1. Temperature applied in the serving path.** `CROPGUARD_TEMPERATURE` now feeds
`CropGuardModel`, and responses carry a `calibrated` flag so a consumer can tell whether
`confidence` has been adjusted — an unlabelled probability is worse than a plainly raw one.

Two guards: `T <= 0` is rejected at construction (zero divides; negative reverses the ordering
and would serve the *least* likely class), and a test asserts the predicted class and top-k
ordering are identical across temperatures. That monotonicity property is precisely what lets
this be switched on for a model already serving traffic without re-validating accuracy.

Not yet enabled on Render — set `CROPGUARD_TEMPERATURE=0.5908` in `render.yaml` to turn it on.

**2. The validation gate measured the wrong quantity.** `min_samples_per_class` counted the
whole dataset, but per-class recall is measured on the *test* split, where the standard error
is `sqrt(p(1-p)/n_test)`. At a 15% test fraction, 100 images total leaves 15 in test.

`Potato___healthy`: 152 images, clears the gate, ends up with 24 in test. The gate now derives
the test share and **warns rather than fails** — refusing to train on a valid public dataset
would be the wrong call; knowing which per-class numbers cannot carry weight is the useful
output.

**3. `POST /predict/batch`.** Capped at 8 images, because each is a full forward pass and on
0.1 vCPU a large batch would hold the single worker long enough to look like an outage.
Per-image errors rather than all-or-nothing: failing an entire batch because one upload is
malformed makes the endpoint useless for its actual purpose, which is processing a folder of
field photos where a few are always broken. Results keep input order and carry filenames, so a
caller can zip them back.

**4. CORS no longer defaults to `*`.** Now `localhost:5173,localhost:8501` (Vite, Streamlit).
CORS is the only thing stopping an arbitrary page from calling the API with a visitor's
browser. Note it is a *browser* control — the Streamlit app calls server-side and is
unaffected either way.

**A correction.** Earlier entries said **eight** classes have fewer than 100 test images. It is
**ten**: Apple scab (95), Apple Black rot (90), Cedar apple rust (40), Corn Cercospora (77),
Grape healthy (63), Peach healthy (55), Potato healthy (24), Raspberry healthy (55),
Strawberry healthy (68), Tomato mosaic virus (56). I had miscounted from the classification
report. Fixed in brain.md, README.md and the module docstrings.

Tests: 127 -> 143.

### 2026-08-02 (later) — Calibration, error analysis, and a demo UI

**Calibration: the model was under-confident, by exactly the predicted amount.**

| | ECE | MCE | Brier | mean conf | accuracy |
|---|---|---|---|---|---|
| Raw softmax | 0.0895 | 0.2777 | 0.0217 | 0.9020 | 0.9911 |
| Temperature-scaled | **0.0036** | 0.3377 | **0.0138** | 0.9946 | 0.9911 |

T = 0.5908, fitted on 2,377 validation images, applied to the 8,125-image test set. **T < 1
means sharpening** - the model was under-confident, which is the less common direction and
precisely what label smoothing predicts. Mean confidence 0.9020 against a theoretical ceiling
of 0.9026 and 99.11% accuracy: theory, observed ceiling and fitted temperature all agree.

Accuracy is identical before and after, and the CLI asserts that rather than trusting it -
dividing logits by a positive scalar is monotonic, so the argmax cannot move.

**MCE rose, and that is not a bug.** Calibration concentrated 8,042 of 8,125 predictions into
one high-confidence bin, leaving sparse bins (n=27, n=18) where a few errors make a large gap.
ECE is population-weighted; MCE is a max over bins including the tiny ones. Both are reported
rather than only the flattering one.

**Error analysis with honest intervals.** Per-class recall is a binomial proportion, so every
figure carries a **Wilson** interval - not the normal approximation, which returns a zero-width
interval at p=1.0 and so claims certainty from a handful of samples. `Potato___healthy`:
recall 0.833 on n=24 -> CI [0.641, 0.933]. Four mistakes. Quoting 0.833 as a weakness without
that interval would be overclaiming, and ten classes fall under 100 test images.

The confusion structure is more informative than any single number:

| True -> Predicted | Count | |
|---|---|---|
| Tomato Early blight -> Late blight | 8 | bidirectional |
| Corn Cercospora <-> Northern Leaf Blight | 6 + 3 | bidirectional |
| Tomato Spider mites -> Target Spot | 6 | |
| Potato Late blight -> Tomato Late blight | 3 | same pathogen, different host |

The last is *Phytophthora infestans*, which genuinely causes late blight in both crops with
near-identical lesions. The four most-confident mistakes all fall in these pairs rather than
scattering - the shape you want, since scattered confident errors would suggest the model had
latched onto something spurious.

**Streamlit demo** (`app/streamlit_app.py`): upload a leaf, get prediction, top-3, confidence,
uncertainty and both latencies, against either the live API or a local ONNX model. The chart
uses the *emphasis* form - rank 1 in the accent hue, ranks 2-3 recessive - because the winner
is the subject, not the class identities; every bar is directly labelled so nothing depends on
colour alone. It surfaces the confidence ceiling explicitly rather than presenting ~0.90 as if
it were a calibrated probability.

**`predict.py` now stores logits** alongside probabilities. Temperature scaling operates on
logits; recovering them from probabilities works (log(p) = z - logsumexp(z), and softmax is
shift-invariant) but storing them removes the need for that argument.

Tests: 101 -> 127.

**Known gap (closed the same day — see the entry above):** the fitted temperature was used
in evaluation but not applied in the serving path.

### 2026-08-02 — Deployed, and production latency is 146x the laptop benchmark

**Live: https://cropguard-api-w9ch.onrender.com** (Render free tier, Docker from
`render.yaml`, weights baked into the image at build time). `/health` healthy, 6/6 correct on
test-set images across six classes, all three Prometheus instruments recording.

| Environment | CPU | Median inference |
|---|---|---|
| Local | i5-12450H, all cores | **24 ms** |
| Render free tier | **0.1 vCPU** | **3504 ms** |

Nothing is broken - ResNet50 at 224x224 is ~4 GFLOPs, and a tenth of a core takes that long.
But it makes a point worth internalising: **every performance claim in this project has now
changed on different hardware.** fp32 latency, INT8-versus-fp32, whether model size matters at
all. This is exactly why the INT8 decision was left open on a laptop benchmark rather than
settled by one.

**Do not quote sub-second latency.** The honest figure is ~3.5 s/image on the free tier. The
first fix worth trying is not quantisation but a smaller backbone - MobileNetV3 is ~18x fewer
FLOPs, and on a dataset this easy the accuracy cost is likely small.

**A theoretical prediction confirmed in production telemetry.** Live confidences: 0.9049,
0.9022, 0.9026, 0.9139, 0.8972 - median 0.9025. Label smoothing at eps=0.1 over K=38 caps the
achievable softmax output at exactly `(1-eps) + eps/K = 0.9026`. The model is *structurally
incapable* of reporting more than ~90% confidence, by design of the loss.

Two consequences: the confidence value must not be shown to a user as a probability until it
is recalibrated (temperature scaling, not yet built), and the ceiling matching the closed form
to three decimals is strong evidence the loss behaved as the maths says. See brain.md 7.1.

**brain.md restructured** for this: new 11 on deployment, hard questions added on production
latency and the confidence ceiling, sections renumbered.

### 2026-08-01 (phase B) — Static quantisation: built, measured, not deployed

`src/cropguard/serving/quantize.py` implements static INT8 quantisation with calibration on
the **validation** split (never test — the holdout decides whether a model ships, so letting
it shape how the model is built would contaminate that decision). The calibration sample is
spread evenly across classes, since activation ranges are set by the extremes a layer sees and
a random draw from a 36x-imbalanced split would barely touch the rare classes.

| | fp32 | dynamic INT8 | static INT8 |
|---|---|---|---|
| Size | 94.3 MB | 23.8 MB | 23.8 MB |
| Latency (batched) | **24.4 ms** | 1569 ms | 78.0 ms |
| Conv op | `Conv` | `ConvInteger` | `QLinearConv` |
| Accuracy (n=3000) | 0.9893 | - | 0.9897 (+0.0003, n.s.) |

**20x faster than dynamic, still 3.2x slower than fp32** - because this machine (i5-12450H)
has no VNNI instructions. INT8 convolution needs a hardware dot-product instruction to beat
fp32; without it ONNX Runtime emulates each INT8 MAC in several AVX2 instructions while fp32
runs on tuned FMA kernels.

**So the deployment decision is still open**, and deliberately so: the benchmark was taken on
the wrong hardware. Render runs server CPUs, which usually *do* have AVX-512 VNNI, and static
INT8 would likely win there on both axes. That measurement has to happen on the target.
fp32 stays deployed until it does.

**New tests (5).** Latency is too machine-dependent to assert in CI; op types are not. The
tests pin that static quantisation emits `QLinearConv` and never `ConvInteger`, that it does
not insert a `DynamicQuantizeLinear` per layer, and - deliberately - that `quantize_dynamic`
*does* still produce `ConvInteger`. That last one turns a future ONNX Runtime fix into a
visible test failure rather than a silently outdated decision.

### 2026-08-01 — Baseline trained, and the pipeline proved reproducible across machines

**Baseline results.** ResNet50, 12 epochs, leaf-grouped split, 8,125-image holdout:
**0.9911 accuracy / 0.9865 macro-F1**, best val macro-F1 0.9926 at epoch 11.

**The leakage hypothesis did not survive contact with data.** See §1 — grouping removed 74.2%
leakage and accuracy did not fall. Left recorded rather than quietly dropped: an experiment
that fails to confirm its hypothesis is still a result, and this one says something real about
the dataset.

**Reproducibility, measured rather than asserted.** Training ran on a Colab T4; the checkpoint
was then exported and evaluated on a Windows CPU box. The two runs agree **to 16 decimal
places**:

```
Colab  : 0.9911384615384615 accuracy / 0.9865412130517314 macro-F1
Windows: 0.9911384615384615 accuracy / 0.9865412130517314 macro-F1
```

That is not a coincidence, and it is worth understanding *why* it holds, because each piece is
a deliberate design decision:

- the split is regenerated from `seed: 42` and hash-verified, not copied
- `classes.json` fixes label order across training, export and serving
- ONNX inference is deterministic, and the parity gate (3.58e-06 here) keeps the exported
  graph faithful to PyTorch
- preprocessing is pinned against torchvision by `test_preprocess.py`

Break any one of those and the numbers drift silently. This is the payoff for the
training/serving skew work in §3.8.

**Export figures.** fp32 94.3MB → INT8 23.8MB (4.0×), parity 3.58e-06 against PyTorch.

### 2026-08-01 — Lightning AI support, HF publishing

**Training moved to Lightning AI** (persistent filesystem, free credits). Rather than fork the
notebook, `01_train_baseline.ipynb` now **detects its environment** and sets paths from it:

| | Lightning AI | Colab |
|---|---|---|
| base path | `/teamspace/studios/this_studio` | `/content` |
| filesystem | persistent | reclaimed without warning |
| checkpoints | plain directory | symlinked to Google Drive |

Hardcoding either would have broken the other, and `google.colab` does not exist on Lightning,
so the Drive-mount cell had to become conditional rather than unconditional. `num_workers` is
now derived from `os.cpu_count()` instead of pinned at 2 — Colab gives ~2 vCPUs, a Lightning
studio usually more, and pinning wasted the difference.

The data-download cell is now idempotent: it checks `manifest.json` and skips a 2.2GB
re-download. That does nothing on Colab but matters on a persistent filesystem, where later
sessions would otherwise re-fetch it every time.

Renamed from `01_train_baseline_colab.ipynb` — the name was now wrong.

**Added `notebooks/upload_to_hf.py`.** Publishes both ONNX graphs, `classes.json`, and a model
card. The card is **generated from the prediction files**, not hand-written, so its numbers
cannot drift from what the model actually produced — the standard way model cards rot. It
records the leaf-grouped split, and states plainly that accuracy is *not* much below published
PlantVillage figures: the dataset is easy, rather than leakage having inflated everything.

**Namespaces differ across services** and this has already caused one wrong instruction:

| Service | Handle |
|---|---|
| GitHub | `abhinav7289A` |
| Weights & Biases | `abhinavbhatia7289` |
| HuggingFace | `XiElonMAsk` |

`CROPGUARD_HF_REPO` takes the **HuggingFace** one.

### 2026-07-31 (later still) — Docker fix and deployment config

**Fixed — the Docker build was broken outright**

`Dockerfile` ended with `COPY models*/ models/`. `models/` is gitignored, so it does not exist
in a clean checkout, and **a `COPY` whose glob matches nothing fails the build** rather than
being skipped. Every `docker build` from a fresh clone failed, and the CI `docker` job with it.

The intent — "bake the model in if it happens to be there" — is not expressible with `COPY`.
Replaced with an explicit build-time fetch from HF Hub behind `ARG CROPGUARD_HF_REPO`, falling
back to the existing startup fetch. A comment now marks the trap so it does not come back.

Also hardened while in there: non-root user, `HEALTHCHECK`, `PYTHONDONTWRITEBYTECODE`, and
dependency layers ordered before application code so edits do not trigger a full reinstall.

**Fixed — CI was quieter than it looked**

The test job installed `.[data,serve,dev]` but not `timm` or `pytorch-lightning`, so
`test_onnx_export.py`'s `importorskip("timm")` silently skipped **all 6 tests** — including the
regression test for the INT8 quantization bug — while CI still reported green. Skips are not
failures, so nothing surfaced.

Now installs the train-only deps, runs pytest with `-rs` so skip reasons appear in the log,
and fails the job outright if the ONNX tests skip. A test that silently stops running is worse
than no test, because it still reads as coverage.

**Added**

- `render.yaml` — Render Blueprint: free plan, Docker runtime, `/health` health check, and a
  `buildFilter` so notebook and docs edits do not trigger redeploys.

**Note on Render specifically.** Render does not forward Blueprint env vars into the Docker
build, so `CROPGUARD_HF_REPO` reaches the container at runtime, not build time — the weights
are fetched at startup there, and the `--build-arg` bake applies to local builds only. On a
tier that sleeps after 15 minutes that means ~26MB re-downloaded per cold start.

### 2026-07-31 (later) — Phase 0: Colab notebook and guard rails

**Added**

- `notebooks/01_train_baseline.ipynb` — end-to-end T4 training: clone → download →
  validate → split → train → ONNX/INT8 → holdout eval. Includes an optional **leakage
  ablation** section that retrains on the naive split to quantify the accuracy inflation.
- `tests/test_import_isolation.py` — enforces that `cropguard.serving` and
  `cropguard.evaluation` import **without torch**, in a subprocess with torch blocked. This
  invariant is what keeps the serving image inside the free tier, and nothing previously
  checked it — every dev machine has torch, so a stray import would only fail in production.
  Includes a "guard the guard" test: if the import blocker ever stopped working, the other
  seven tests would pass vacuously.
- `test_extends_chains_through_multiple_levels` — the Colab configs use a 3-level chain
  (`colab_resnet50` → `resnet50_baseline` → `base`), so that behaviour is now pinned.

**Fixed**

- `config.py` documented "single-level `extends`" while the implementation recurses. The
  notebook depends on the recursion, so the docstring was corrected rather than the code.
- `src/cropguard/data/*` had **never been linted**. Ruff respects `.gitignore`, and the
  unanchored `data/` rule (§4.3) hid the whole package from it. Fixing the ignore rule
  exposed a latent `UP035`. A reminder that an over-broad ignore rule silently disables
  tooling, not just version control.
- `notebooks/` excluded from ruff — cells legitimately import mid-file.

**Split reproducibility.** The notebook asserts `splits.json` hashes to
`9764d8f2eb2046…` on Colab, matching the local split. This is what allows training on Colab
without transferring the 2.2GB dataset: the split is regenerated from `seed: 42` rather than
copied. If that assert fires, results are not comparable and training should stop.

Tests: 81 → **90**.

### 2026-07-31 — Data pipeline, evaluation framework, first commit

**Fixed**

- `train.py` hard-required `wandb` despite documenting it as optional → `CSVLogger` fallback
- `download.py` targeted a config that no longer resolves → reads `data.zip` directly (§3.3)
- ONNX INT8 quantization was impossible → `dynamo=False` + `quant_pre_process` (§3.7)
- `.gitignore` silently excluded `src/cropguard/data/` (§4.3)
- Missing `README.md` broke the Docker build (both it and pyproject referenced the file)
- pydantic v2 `model_*` protected-namespace conflict in serving settings
- Windows cp1252 crash on torch's emoji progress output during export

**Added**

- Leaf-grouped split eliminating 74.2% test leakage (§3.4)
- `evaluation/` — hypothesis testing, prediction dumping, comparison CLI (§3.9)
- 81 tests, CI workflow (lint / test / Docker smoke test), README

**Verified**

- Dataset: 54,305 images, 38 classes, 0 corrupt, 0 low-resolution, 0 underpopulated
- Split: exhaustive, disjoint, 0 leaked groups, mean per-class drift 0.0012
- Training: real 1-epoch CPU run end to end
- ONNX: parity 2.4e-07, INT8 32.2MB → 8.1MB

**Known open issues**

- No trained model — the blocker for everything downstream
- `uncertainty` is entropy, not true uncertainty quantification
- No test enforces that `serving/` and `evaluation/` stay torch-free
- ~24% of images fall back to singleton leaf groups (§3.4)

---

## How to maintain this file

When behaviour changes: add a dated entry to §5, and update the relevant §3 section. If an
explanation is replaced, mark the old one *Superseded (date)* rather than deleting it — the
reasoning trail is the point. New file → add it to §2.
