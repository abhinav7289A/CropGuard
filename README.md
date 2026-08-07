# CropGuard

Production MLOps pipeline for crop disease detection — 38-class leaf classifier trained on
PlantVillage, served as a CPU-only ONNX API with experiment tracking, statistical A/B testing,
and drift monitoring. Runs entirely on free tiers.

> **Live demo: https://huggingface.co/spaces/XiElonMAsk/cropguard** — pick a model, compare
> fp32 against static INT8 side by side, toggle calibration
>
> **Live API: https://cropguard-api-w9ch.onrender.com** — `/health` · `/predict` · `/metrics`
>
> **Status: baseline trained, evaluated, calibrated and deployed.** The ConvNeXt-Tiny
> challenger has been trained and A/B tested against it — the gate declined to promote it, and
> [that result](#statistical-model-comparison) is reported as measured. Field-photo evaluation
> and the `/feedback` loop are the open items — see [Roadmap](#roadmap).

## Results

ResNet50, 12 epochs, evaluated on a **leak-free** 8,125-image holdout:

| Metric | Value |
|---|---|
| Test accuracy | **0.9911** |
| Test macro-F1 | **0.9865** |
| Best val macro-F1 | 0.9926 |
| Test leaf leakage | **0.0%** |

Macro-F1 is the number to read: the dataset is imbalanced ~36×, so accuracy is dominated by
the largest classes.

**The leakage result.** The standard PlantVillage split leaves 74.2% of test images sharing a
physical leaf with training. Removing that entirely did **not** reduce accuracy — best
validation macro-F1 was 0.9926 grouped against 0.9921 naive, i.e. the contaminated split
scored marginally *lower*. Different validation sets, so not a paired comparison, but the
direction is opposite to what leakage-inflation predicts.

The honest conclusion is that PlantVillage is genuinely easy rather than that its published
figures are artifacts. The grouped split still matters — a 74%-contaminated holdout would make
every downstream statistical test answer the wrong question — but the inflation everyone
assumes is not there. See [`brain.md`](brain.md) §0.

**Where the errors are.** ~72 misclassifications, concentrated in two biologically plausible
pairs: corn *Cercospora ↔ Northern Leaf Blight*, and tomato *Early ↔ Late blight*. Note that
`Potato___healthy` has only 24 test images, so its 0.833 recall is four mistakes with a ±15pp
confidence interval — ten classes fall below 100 test images and their per-class numbers
should not be read as precise.

### Production latency — measured, not estimated

| Environment | CPU | Median inference |
|---|---|---|
| Local benchmark | i5-12450H, all cores | **24 ms** |
| **Render free tier** | **0.1 vCPU** | **3504 ms** |

146×. Nothing is broken — ResNet50 at 224×224 is ~4 GFLOPs and the free tier allocates a tenth
of a core. It is stated here rather than buried because a latency number without its hardware
is not a result, and every performance claim in this project changed between the two machines.

The first fix worth trying is a **smaller backbone**, not quantization: MobileNetV3 is ~18×
fewer FLOPs, and on a dataset this easy the accuracy cost is likely small. Serving a
25.6M-parameter ResNet50 to classify leaves is the actual inefficiency.

## Architecture

```mermaid
flowchart TD
    A[PlantVillage · HuggingFace] --> B[download → ImageFolder]
    B --> C[validate: integrity, resolution, class balance]
    C --> D[leaf-grouped split 70/15/15 → splits.json]
    D --> E[Lightning training · timm backbone]
    E -->|W&B: metrics, artifacts| F[checkpoint]
    F --> G[ONNX export + parity check]
    G --> H[FastAPI + ONNX Runtime]
    H --> I[/predict · /health · /metrics/]
    I --> J[Prometheus → Grafana]
```

The repo is split by runtime environment so the serving image never pulls in torch:

| Package | Install | Runs on |
|---|---|---|
| `cropguard.data` | `pip install ".[data]"` | anywhere (dataset prep) |
| `cropguard.training` | `pip install ".[train]"` | Colab / Lightning AI GPU |
| `cropguard.serving` | `pip install ".[serve]"` | Render free tier (CPU, no torch) |

`configs/classes.json` is the single source of truth for label order and is shared by training,
ONNX export, and the API — the one file that must never drift between the three.

## Quickstart

```bash
python -m pip install -e ".[data,serve,dev]"

# 1. Data — omit --limit-per-class for the full 54K-image dataset
export CROPGUARD_DATA_DIR=/path/outside/onedrive     # Windows: set CROPGUARD_DATA_DIR=C:\cropguard-data
python -m cropguard.data.download --config configs/base.yaml --limit-per-class 40
python -m cropguard.data.validate --config configs/base.yaml --min-samples-per-class 40
python -m cropguard.data.split    --config configs/base.yaml

# 2. Train (needs .[train]; GPU strongly recommended)
python -m cropguard.training.train --config configs/resnet50_baseline.yaml --fast-dev-run
python -m cropguard.training.train --config configs/resnet50_baseline.yaml

# 3. Export for serving
python -m cropguard.serving.onnx_export \
    --ckpt checkpoints/resnet50-baseline/best-....ckpt \
    --out models/cropguard.onnx --quantize

# 4. Serve
uvicorn cropguard.serving.app:app --port 8000
curl -F "file=@leaf.jpg" http://localhost:8000/predict
```

`--limit-per-class` builds a small subset for smoke tests. Training is resumable — checkpoints
are written every epoch and `--resume <ckpt>` picks up after a Colab/Lightning disconnect.

### The split is grouped by leaf, on purpose

PlantVillage contains 54,305 images of only ~7,600 distinct physical leaves — roughly 7 shots
of each leaf from different angles. A plain per-image stratified split scatters those
near-duplicates across train and test, letting a model memorize a leaf and "recognize" it
again at test time. Measured on this dataset:

| Split strategy | test images sharing a leaf with train |
|---|---|
| `--strategy stratified` (naive) | **74.2%** |
| `--strategy grouped` (default) | **0.0%** |

`grouped` packs whole leaf groups per class, largest-first, into whichever split is furthest
below target. It holds every class to within ~0.1pp of the requested fractions (mean
`|test_frac - 0.15|` = 0.0012) while keeping group integrity exact — 0 of 20,015 groups span
a split boundary.

This matters beyond tidiness: accuracy on the naive split is inflated, and any A/B test built
on it measures memorization rather than generalization. `--strategy stratified` is retained
only so that gap can be quantified and reported. `<data_root>/split_report.json` records the
strategy and the leakage measurement for whichever split you ran.

### Data location on Windows

Point `CROPGUARD_DATA_DIR` at a path **outside OneDrive** (e.g. `C:\cropguard-data`). OneDrive
sync on 54K small files is slow and can lock files mid-write during training.

## API

| Endpoint | Description |
|---|---|
| `POST /predict` | multipart image → `predicted_class`, `confidence`, `top_k`, `uncertainty`, `calibrated`, `model_version`, `latency_ms` |
| `POST /predict/batch` | up to 8 images; per-image errors rather than all-or-nothing |
| `GET /health` | `healthy` / `degraded` (model not loaded), model version, uptime |
| `GET /metrics` | Prometheus: request counts, latency histogram, confidence histogram |

Uploads are validated before inference: JPEG/PNG only, ≥128px on the short side, ≤10MB.
The API starts `degraded` rather than crashing when no model is present, so the container is
deployable before the first model exists.

`uncertainty` is currently normalized predictive entropy in [0, 1]; MC-dropout and ensemble
variance land with the calibration work.

**Confidence is capped at ~0.90 in the raw softmax by design.** Label smoothing (ε=0.1, K=38) bounds the
achievable softmax output at `(1−ε) + ε/K = 0.9026`, and live predictions sit exactly there
(median 0.9025). That is the loss function working as intended, not model uncertainty — but it
means the raw number should not be read as a probability. Set `CROPGUARD_TEMPERATURE=0.5908`
to serve calibrated confidence; responses carry a `calibrated` flag either way. See
[`brain.md`](brain.md) §7.1.

### Configuration

All settings are env vars with the `CROPGUARD_` prefix:

| Variable | Default | Purpose |
|---|---|---|
| `CROPGUARD_MODEL_PATH` | `models/cropguard.onnx` | ONNX weights |
| `CROPGUARD_CLASSES_PATH` | `configs/classes.json` | label order |
| `CROPGUARD_MODEL_VERSION` | `v0.1.0-baseline` | reported in responses + metrics |
| `CROPGUARD_CORS_ORIGINS` | `localhost:5173,localhost:8501` | comma-separated allowlist, not `*` |
| `CROPGUARD_TEMPERATURE` | `1.0` | calibration; `0.5908` for this model |
| `CROPGUARD_MAX_BATCH_SIZE` | `8` | images per `/predict/batch` |
| `CROPGUARD_DATA_DIR` | `data` | dataset root (overrides config) |
| `CROPGUARD_HF_REPO` | unset | HF Hub repo to pull weights from at startup |

## Docker

```bash
# Weights fetched from HF Hub on startup
docker build -t cropguard .
docker run -p 8000:8000 -e CROPGUARD_HF_REPO=<user>/cropguard-models cropguard

# Or bake them into the image (faster cold starts)
docker build --build-arg CROPGUARD_HF_REPO=<user>/cropguard-models -t cropguard .
docker run -p 8000:8000 cropguard
```

CPU-only ONNX Runtime, no torch — the image stays well under 500MB. It runs as a non-root
user and ships a `HEALTHCHECK`. With no model available the API starts *degraded* rather than
crashing, so the container is deployable and health-checkable before a model exists.

### Why fp32 is served, not INT8

| | fp32 | dynamic INT8 | static INT8 |
|---|---|---|---|
| Size | 94.3 MB | 23.8 MB | 23.8 MB |
| Latency (batched, i5-12450H) | **24.4 ms** | 1569 ms | 78.0 ms |
| Conv op | `Conv` | `ConvInteger` | `QLinearConv` |
| Accuracy (n=3000) | 0.9893 | — | 0.9897 (n.s.) |

`quantize_dynamic` rewrites every `Conv` into `ConvInteger`, which ONNX Runtime's CPU backend
has no optimized kernel for — a **75× regression**. Dynamic quantization suits
MatMul-dominated models (Transformers, RNNs), not CNNs. `cropguard.serving.quantize` does it
properly with calibration and emits `QLinearConv`, which is 20× faster.

It is still 3.2× slower than fp32 **on this hardware**, because an i5-12450H has no VNNI
instructions and INT8 convolution needs one to beat tuned fp32 FMA kernels. Server CPUs
generally do have AVX-512 VNNI, so this benchmark does not settle the question — **the
measurement that decides is the one taken on the deployment target**, and it hasn't been taken
yet. fp32 ships until then. See [`brain.md`](brain.md) §9.

## Deployment (Render free tier)

`render.yaml` is a Render Blueprint — **New → Blueprint** in the dashboard, pointed at this
repo, builds the Dockerfile and wires up the service. Model weights come from HuggingFace Hub
via `CROPGUARD_HF_REPO`, so the ONNX file never has to live in git.

Free-tier realities worth knowing before relying on it:

| Limit | Consequence |
|---|---|
| 512 MB RAM | ONNX Runtime + ResNet50 fp32 (94MB) fits, without much headroom |
| 0.1 vCPU | Inference is seconds per image, not milliseconds |
| Sleeps after 15 min idle | ~50s cold start, plus a ~26MB weight re-download |

The Dockerfile's `CROPGUARD_HF_REPO` build ARG carries a default, which is what bakes the
weights into the image — Render never forwards Blueprint env vars into a Docker build, so
anything needing a passed `--build-arg` would silently fall back to downloading 94MB on every
cold start.

If 0.1 vCPU proves too slow for a demo, HuggingFace Spaces (2 vCPU / 16GB, free) is better
hardware, though a weaker "production deployment" story.

## Experiments

| Config | Backbone | Augmentation | Test accuracy | Test macro-F1 | Role |
|---|---|---|---|---|---|
| `configs/resnet50_baseline.yaml` | ResNet50 | medium | **0.9911** | 0.9865 | deployed |
| `configs/convnext_tiny.yaml` | ConvNeXt-Tiny | heavy | 0.9908 | **0.9890** | challenger, not promoted |

The two configs differ in **seven** ways at once — architecture, augmentation, epochs, batch
size, learning rate, weight decay and dropout — so the comparison answers "which configuration
should ship", not "is ConvNeXt better than ResNet50". No single variable can be credited, and
isolating one needs a controlled ablation that is not built.

Configs use single-level `extends: base.yaml` inheritance with a deep merge, so a child can
override `train.lr` without losing the rest of the `train` block.

Set `WANDB_MODE=offline` (or leave `WANDB_API_KEY` unset) to train without Weights & Biases.

## Calibration

A 99% accurate model can still be badly calibrated, and this one was — in the *under*-confident
direction, which is the less common one:

| | ECE | MCE | Brier | mean confidence | accuracy |
|---|---|---|---|---|---|
| Raw softmax | 0.0895 | 0.2777 | 0.0217 | 0.9020 | 0.9911 |
| **Temperature-scaled (T = 0.591)** | **0.0036** | 0.3377 | **0.0138** | 0.9946 | 0.9911 |

```bash
python -m cropguard.evaluation.calibrate     --val artifacts/preds_val.npz --test artifacts/preds_fp32.npz     --out artifacts/calibration.json
```

**ECE fell 96%**, and accuracy is provably unchanged — dividing logits by a positive scalar is
monotonic, so no prediction can move. T is fitted on **validation only**; fitting on test would
tune the calibration to the set used to report it.

`T = 0.591 < 1` means *sharpening*: the model was under-confident. That is exactly what label
smoothing predicts (ε=0.1, K=38 ⇒ ceiling 0.9026), and mean confidence before calibration was
0.9020. Theory and measurement agree.

**MCE rose**, which is expected rather than a defect: calibration concentrated 8,042 of 8,125
predictions into one high-confidence bin, leaving sparse bins where a few errors make a large
gap. ECE is population-weighted; MCE is a max over bins including tiny ones. See
[`brain.md`](brain.md) §8.8.

## Error analysis

Per-class metrics carry **Wilson score intervals**, because per-class recall is a binomial
proportion and ten classes have fewer than 100 test images:

```
Potato___healthy      n=24   R=0.833 [0.641, 0.933]   <- four mistakes; do not read as 0.833
Corn___Cercospora     n=77   R=0.922 [0.840, 0.964]
Tomato___Early_blight n=149  R=0.913 [0.856, 0.948]
```

The confusion structure says more than any single number:

| True → Predicted | Count | |
|---|---|---|
| Tomato Early blight → Late blight | 8 | bidirectional |
| Corn Cercospora ↔ Northern Leaf Blight | 6 + 3 | bidirectional |
| Tomato Spider mites → Target Spot | 6 | |
| Potato Late blight → Tomato Late blight | 3 | *same pathogen, different host* |

The four most-confident mistakes all fall in these pairs rather than scattering — which is the
reassuring shape, since scattered confident errors would suggest something spurious had been
learned.

## Drift detection

You cannot measure accuracy in production — nobody labels the leaves. `cropguard.monitoring`
watches distributions instead: **PSI** on the confidence distribution (inputs drifting
off-distribution make the model hedge) and **total variation distance** on the predicted class
mix (the population served changed).

Both are **effect sizes, not p-values**, and the reason is measurable. Split the test set into
two random halves — same model, same data, nothing to find:

| | Value | Reads as |
|---|---|---|
| PSI | 0.0028 | stable |
| TVD | 0.0576 | no shift |
| chi² p-value | **8.35e-09** | "wildly significant" |

The p-value fires on two random halves of identical data, because significance measures
*detectability* and detectability grows with n. A monitor whose sensitivity depends on traffic
volume pages for nothing, gets muted, and then misses the real event. See
[`brain.md`](brain.md) §12.

## Demo UI

```bash
pip install -e ".[demo]"          # live API backend only
pip install -e ".[demo,serve]"    # adds onnxruntime, needed to run models locally
streamlit run app/streamlit_app.py
```

Upload a leaf and get the prediction, top-3, confidence, uncertainty and both latencies. The
backend list is built from [`configs/models.json`](configs/models.json) plus the deployed API,
and only models whose ONNX file is actually on disk are offered — the panel never lists
something it cannot run.

Two things it does that a plain demo does not:

- **Compare mode** runs the same image through several models side by side and says whether
  they agree. On roughly 98.8% of holdout images they do; watching the other 1.2% is a better
  feel for what "no significant improvement" means than reading a p-value.
- **A calibration toggle**, because calibration is a serving-time decision rather than a
  property of the weights. Turning it off shows the raw softmax sitting at the label-smoothing
  ceiling — the same image goes from 0.76 to 0.99 confidence with the predicted class
  unchanged, since dividing logits by a positive scalar cannot reorder them.

If `artifacts/ab_comparison.json` is present the panel also surfaces the A/B verdict, so the
demo reports the comparison rather than only the winner.

### Deploying the panel to a Space

**Live at https://huggingface.co/spaces/XiElonMAsk/cropguard.** To redeploy after a change:

```bash
export HF_TOKEN=hf_...                                   # write token
python scripts/deploy_space.py --repo <user>/cropguard    # --dry-run to preview
```

The script uploads the app, both configs, `spaces/Dockerfile` and the torch-free
`cropguard.serving` package — 34 KB in total. It builds as a **Docker** Space, not a Streamlit
one: the Hub no longer accepts `streamlit` as an SDK for new Spaces, so the Dockerfile runs
Streamlit itself on port 7860. **Weights are not uploaded.** The panel pulls them from
[`XiElonMAsk/cropguard-models`](https://huggingface.co/XiElonMAsk/cropguard-models) on first
use, so the demo and the API load the same artifact and a new model never has to be copied to
two places. `spaces/README.md` becomes the Space card; its YAML front matter is what tells
Spaces which SDK to run.

Running models in-process on a Space (2 vCPU) rather than calling the Render API (0.1 vCPU) is
roughly a hundredfold difference in latency for the same ONNX graph — which is the §11 lesson
made clickable rather than a table in a document. Keeping both backends selectable is the point:
the panel can show the same model at ~3.5 s and at ~50 ms, and the only thing that changed is
how much CPU somebody allocated.

## Statistical model comparison

Accuracy going up is not evidence that a model is better. `cropguard.evaluation` runs the
paired tests that decide whether a difference survives resampling, and whether it is large
enough to care about:

```bash
python -m cropguard.evaluation.predict --model models/baseline.onnx   --split test --out artifacts/preds_baseline.npz
python -m cropguard.evaluation.predict --model models/challenger.onnx --split test --out artifacts/preds_challenger.npz
python -m cropguard.evaluation.compare \
    --baseline artifacts/preds_baseline.npz \
    --challenger artifacts/preds_challenger.npz \
    --out artifacts/comparison.json
```

The first real comparison — ResNet50 baseline against the ConvNeXt-Tiny challenger, both
scored on the identical 8,125-image holdout:

```
n=8125 holdout images
accuracy: A=0.9911  B=0.9908  (diff -0.0004)
McNemar (chi-squared (continuity corrected)): discordant pairs 49 vs 46 (A > B), stat=0.0421, p=0.8374 -> not significant at alpha=0.05
Bootstrap (10000 resamples): diff=-0.0004, 95% CI [-0.0027, +0.0020] -> includes 0
Paired t-test: mean diff=+0.0037 95% CI [-0.0024, +0.0098], t=1.235, p=0.2248 -> not significant; d=+0.200 (small)
Power=0.225 at n=38 -> UNDERPOWERED; n=198 needed for 0.8 power at d=0.200
NOTE: the per-class test is underpowered — a null result here is inconclusive, not evidence of equivalence.
VERDICT: no significant improvement demonstrated
```

**The gate declined to promote, and the challenger was not deployed.** Two things are worth
reading carefully before treating that as a failure:

- **The two headline metrics disagree.** Accuracy fell by 0.0004 while macro-F1 rose from
  0.9865 to 0.9890 — and on a 36×-imbalanced problem macro-F1 has the stronger claim to being
  primary. The challenger traded a little majority-class accuracy for rare-class balance.
- **The per-class test cannot settle it.** Its sample size is the *class count*: 38 classes at
  d = 0.200 gives power 0.225, and 0.8 would need 198 classes. No amount of extra labelling
  fixes that, because the dataset has 38 diseases and always will.

The macro-F1 bootstrap (`bootstrap_macro_f1_difference`) answers the question those two points
raise: it resamples the 8,125 images and recomputes the macro statistic, so it is powered by
the holdout rather than by the class count. Run on these same prediction files:

```
macro-F1 Bootstrap (10000 resamples): diff=+0.0025, 95% CI [-0.0015, +0.0067] -> includes 0
```

**The macro-F1 gain does not survive resampling either**, so the challenger is not better on
the metric that appeared to favour it. That turns a "we could not tell" into a properly
powered null: the true difference sits between -0.0015 and +0.0067, tighter than the variation
a different random seed would produce.

The gate itself stays keyed on accuracy on purpose. The rule was fixed before any challenger
was trained, and widening it to accept a macro-F1 win *after* seeing which metric moved would
be picking the test that gives the desired answer. Macro-F1 is measured, printed, and allowed
to contradict the verdict in public.

- **McNemar's test** — the standard paired test for two classifiers on one holdout. Only
  discordant pairs carry information; it switches to an exact binomial when they are scarce.
- **Bootstrap CI** — 10,000 paired resamples. Both models are resampled on the same indices,
  which preserves the pairing and tightens the interval. Available for accuracy and, when the
  prediction files carry predicted classes, for macro-F1.
- **Per-class paired t-test + Cohen's d_z** — asks whether the gain is spread across classes
  or concentrated in the common ones.
- **Power analysis** — reported alongside every null result, because "not significant" from an
  underpowered test means *we could not tell*, not *there is no difference*.

`compare` exits non-zero unless the challenger wins, and
[`.github/workflows/promotion-gate.yml`](.github/workflows/promotion-gate.yml) runs it against
prediction files pulled from the Hub, so promotion is gated on statistical evidence rather than
a raw accuracy delta. Inference runs through the exported ONNX graph and the serving
preprocessor, so these numbers are the ones production actually produces.

## Development

```bash
python -m pip install -e ".[data,serve,dev]"
python -m pytest          # torch-dependent tests skip automatically if torch is absent
python -m ruff check .
```

Tests build a synthetic ImageFolder dataset and a tiny ONNX graph in `tmp_path`, so the full
suite runs without PlantVillage and without a trained model. `test_preprocess.py` pins the
NumPy serving preprocessor against torchvision's `eval_transforms` — that comparison is the
guard against training/serving skew.

## Roadmap

- [x] Data pipeline: download, validation, leakage-free leaf-grouped split
- [x] Lightning training loop, W&B logging, resumable checkpoints
- [x] ONNX export with PyTorch parity check
- [x] Static INT8 quantization — built and measured; **not deployed**, see below
- [x] FastAPI serving, Prometheus metrics, Docker image
- [x] Test suite and CI
- [x] Hypothesis testing: McNemar, bootstrap CI, Cohen's d, power analysis
- [x] Train the ResNet50 baseline (99.11% / 0.9865 macro-F1, leak-free holdout)
- [x] Leakage ablation: measured, and the inflation is not there
- [x] Publish weights to HF Hub ([`XiElonMAsk/cropguard-models`](https://huggingface.co/XiElonMAsk/cropguard-models))
- [x] Deploy to Render — live, with real latency measured
- [x] ConvNeXt-Tiny challenger and the first real A/B comparison — **null result, not promoted**
- [x] Macro-F1 paired bootstrap, and the promotion gate wired into CI
- [x] Calibration: temperature scaling, ECE/MCE/Brier, reliability curves
- [x] Error analysis: Wilson intervals, confusion pairs, confident mistakes
- [x] Streamlit demo UI
- [x] Apply the fitted temperature in the serving path
- [ ] MC-dropout / ensemble uncertainty
- [ ] Error and subgroup analysis (lab vs. field photos)
- [x] Drift detection: PSI on confidence, TVD on class mix
- [ ] `/feedback` endpoint with retraining triggers
- [ ] Load testing and a Grafana dashboard

## Deliberately not built

Several things a "production MLOps pipeline" is expected to have are absent, and their absence
is a decision rather than an oversight:

| Not built | Why |
|---|---|
| **DVC** | The dataset is a single immutable HuggingFace release and the split is *regenerated* from a seed and a sorted file listing, then SHA-256 verified. Versioning 2.2 GB of unchanging images would add a remote to configure and nothing that is not already reproducible. |
| **MLflow registry** | One trained baseline and one challenger. A registry solves discovery and lineage across many models; with two, `configs/models.json` and the Hub do the same job without a server to run. |
| **W&B sweeps / Optuna** | Free-tier GPU hours are the binding constraint, and a 20-run Bayesian sweep on a benchmark already at 99.11% would spend them chasing 0.3 points. The ablation that would actually be informative — isolating one of the seven config differences — is the honest use of that compute. |
| **Automated retraining** | Retraining needs a signal that the model degraded. Drift detection reports that the *inputs* moved, which is not the same thing, and confirming decay needs labels the `/feedback` endpoint would collect. Wiring a retrain trigger to an input-drift alarm would automate a decision the evidence cannot support. |
| **Grafana / AlertManager** | `/metrics` exports Prometheus format and the drift module computes the numbers. Dashboards over a single free-tier instance serving demo traffic would be decoration. |

The one genuinely missing measurement is **field photographs**: every number here comes from
lab images with plain backgrounds, and the literature reports large drops on real-world photos
of the same diseases. Until that is measured, the accuracy figure describes a benchmark rather
than the problem this project claims to address. That is the top of future work, not a
deliberate omission.
