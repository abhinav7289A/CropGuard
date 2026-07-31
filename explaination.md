# CropGuard — Explanation Log

A living record of **what each file does, why it works the way it does, and what changed**.
The README tells a visitor how to run the project; this file tells *you* why it is built this
way, including the decisions that are not obvious from reading the code.

Every entry is dated. When behaviour changes, the old explanation is not deleted — it is
marked superseded, so the reasoning trail stays intact.

**Last updated:** 2026-07-31

---

## 1. Where the project stands

| | |
|---|---|
| **Critical path** | Week 1, end of Day 3 — data acquired, validated, split |
| **Blocker** | Day 6–7 baseline training. Nothing downstream is real until a model exists |
| **Built ahead of schedule** | Week 3 A/B testing, most of Week 4 serving |
| **Tests** | 81 passing, ruff clean |
| **Dataset** | 54,305 images / 38 classes at `C:\cropguard-data` (outside OneDrive) |
| **Repo** | github.com/abhinav7289A/CropGuard, commit `0b12e1d` |

**Not yet built:** EDA notebook, duplicate detection, DVC, MLflow registry, hyperparameter
sweeps, augmentation/transfer-learning ablations, calibration, bias analysis, `/feedback`
endpoint, drift detection, Grafana, Render deployment, load testing.

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
| `onnx_export.py` | Checkpoint → ONNX → INT8, gated by a PyTorch parity check. See §3.7. |

### Evaluation — `src/cropguard/evaluation/`

| File | What it does |
|---|---|
| `hypothesis.py` | McNemar, bootstrap CI, paired t-test, Cohen's d, power analysis. See §3.9. |
| `predict.py` | Runs a model over a split, saves per-image predictions to `.npz`. |
| `compare.py` | A/B comparison CLI. Exits non-zero unless the challenger genuinely wins. |

### Infrastructure

| File | What it does |
|---|---|
| `Dockerfile` | CPU-only serving image. Installs `.[serve]` only, so no torch. |
| `.dockerignore` | Keeps data, checkpoints and tests out of the build context. |
| `.gitignore` | Ignore rules — note the anchoring bug described in §4.3. |
| `.github/workflows/ci.yml` | Three jobs: lint, test, and a Docker build + container smoke test. |
| `scripts/fetch_model.py` | Pulls model weights from HF Hub at container start if not baked in. |
| `notebooks/01_train_baseline_colab.ipynb` | End-to-end baseline training on a Colab T4, plus the optional leakage ablation. |

### Tests worth knowing about

| File | Guards |
|---|---|
| `test_import_isolation.py` | `serving`/`evaluation` never import torch — the invariant the free-tier deploy rests on. |
| `test_preprocess.py` | The NumPy serving preprocessor against torchvision. Guards training/serving skew. |
| `test_data_groups.py` | Group integrity and stratification of the leaf-grouped split. |
| `test_onnx_export.py` | The exported graph stays quantizable — the trap in §3.7. |
| `test_hypothesis.py` | Statistics checked against textbook values, scipy, and published power tables. |

---

## 3. How things actually work

### 3.1 Why dependencies are split five ways

Render's free tier is memory- and disk-constrained. A torch install is ~2GB; ONNX Runtime is
~50MB and does everything inference needs. So the packages are import-isolated by environment:

- `cropguard.serving` never imports torch — it must stay installable with `.[serve]` alone
- `cropguard.evaluation` never imports torch either, so statistics can run anywhere including CI
- `cropguard.training` is the only torch-dependent subpackage

`src/cropguard/__init__.py` is kept free of heavy imports for the same reason. **If you ever
add `import torch` to `serving/` or `evaluation/`, the deployment breaks.** That constraint is
not enforced by a test yet — worth adding.

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

### 2026-07-31 (later) — Phase 0: Colab notebook and guard rails

**Added**

- `notebooks/01_train_baseline_colab.ipynb` — end-to-end T4 training: clone → download →
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
