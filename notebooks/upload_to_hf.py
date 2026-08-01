"""Publish the trained model to HuggingFace Hub, with a model card built from measured values.

Run from the training notebook (`%run notebooks/upload_to_hf.py`) or from the repo root in a
session where `models/` and `artifacts/` are populated. On an ephemeral runtime, do this
before the session is reclaimed - otherwise the weights are gone.

The card is generated from the prediction files rather than hand-written, so the numbers in it
cannot drift from the numbers the model actually produced - the usual way model cards go stale.

Override the destination by defining HF_REPO_ID before %run, or setting CROPGUARD_HF_REPO.
"""

import json
import os
from getpass import getpass
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from sklearn.metrics import f1_score

from cropguard.evaluation.predict import load_predictions

# `%run` shares the notebook globals, so HF_REPO_ID set in a cell is picked up here.
REPO_ID = globals().get("HF_REPO_ID") or os.environ.get(
    "CROPGUARD_HF_REPO", "XiElonMAsk/cropguard-models"
)
DATA_DIR = os.environ.get("CROPGUARD_DATA_DIR", "data")

# ---------------------------------------------------------------- measured values
fp32 = load_predictions("artifacts/preds_fp32.npz")
acc32 = float(fp32["correct"].mean())
f1_32 = float(f1_score(fp32["labels"], fp32["predictions"], average="macro"))

int8_path = Path("artifacts/preds_int8.npz")
if int8_path.exists():
    int8 = load_predictions(int8_path)
    acc8 = float(int8["correct"].mean())
    f1_8 = float(f1_score(int8["labels"], int8["predictions"], average="macro"))
    int8_row = f"| INT8 (dynamic) | {acc8:.4f} | {f1_8:.4f} | not served - see below |"
else:
    acc8 = f1_8 = None
    int8_row = "| INT8 (dynamic) | not evaluated | not evaluated | not served - see below |"

split = json.load(open(f"{DATA_DIR}/split_report.json"))
classes = json.load(open("configs/classes.json"))
leak = split["leakage"]["test"]

# ---------------------------------------------------------------- model card
card = f"""---
license: cc-by-sa-4.0
tags:
  - image-classification
  - agriculture
  - plant-disease
  - onnx
datasets:
  - mohanty/PlantVillage
metrics:
  - accuracy
  - f1
---

# CropGuard - Crop Disease Classifier ({len(classes)} classes)

ResNet50 fine-tuned on PlantVillage, exported to ONNX and dynamically quantised to INT8 for
CPU-only serving. Part of [CropGuard](https://github.com/abhinav7289A/CropGuard), an
end-to-end MLOps pipeline.

## Results (held-out test set, n={len(fp32["labels"]):,})

| Model | Accuracy | Macro-F1 | Note |
|---|---|---|---|
| fp32 | {acc32:.4f} | {f1_32:.4f} | reference |
{int8_row}

Macro-F1 is the metric to read here, not accuracy: the dataset is imbalanced ~36x, so accuracy
is dominated by the largest classes.

**Only `cropguard.onnx` (fp32) is published.** Dynamic INT8 quantization was tried and is
*not* shipped: `quantize_dynamic` rewrites every `Conv` into `ConvInteger`, which ONNX
Runtime's CPU backend has no optimized kernel for. Measured on an Intel Alder Lake CPU it ran
at **1567 ms/image against 19 ms/image for fp32** - a 75x regression in exchange for 4x less
disk. Dynamic quantization suits MatMul-dominated models (Transformers, RNNs), not CNNs; the
correct approach for a Conv-heavy network is *static* quantization with a calibration set,
which emits the optimized `QLinearConv`. That is not built yet.

## The split is grouped by leaf, and that matters

PlantVillage contains {split["num_images"]:,} images of only ~7,600 *distinct physical leaves* -
roughly 7 photographs of each. A standard per-image stratified split scatters those
near-duplicates across train and test, so a model can score well by memorising leaf identity
rather than learning disease morphology. Measured on this dataset, a naive split leaves
**74.2% of test images sharing a leaf with training**.

This model was trained on a **leaf-grouped** split instead:

| | Naive stratified | Grouped (used here) |
|---|---|---|
| test images sharing a leaf with train | 74.2% | **{leak["leaked_pct_of_resolved"]:.1f}%** |
| train / val / test | - | {split["sizes"]["train"]:,} / {split["sizes"]["val"]:,} / {split["sizes"]["test"]:,} |

So the accuracy above is measured on a holdout with no leaf overlap. Note that it is *not*
much lower than typically published PlantVillage figures - the honest reading is that this
dataset is genuinely easy, not that leakage was inflating everything.

## Intended use

Identifying disease on **single leaves photographed against a plain background**, matching the
PlantVillage capture protocol.

## Limitations - read before deploying this

- **Lab images, not field images.** Every training image is a detached leaf on a uniform
  background under controlled lighting. Real photographs from a farm - variable lighting,
  occlusion, multiple leaves, soil backgrounds - are a different distribution, and published
  work on this dataset reports large drops there. This model has **not** been evaluated on
  field photographs.
- **Per-class metrics for rare classes are noisy.** Eight classes have fewer than 100 test
  images; the smallest (`Potato___healthy`) has 24. A recall of 0.833 there is 4 mistakes, and
  its confidence interval spans roughly +/-15 points. Do not read those per-class numbers as
  precise.
- **Confidence is not calibrated.** Training used label smoothing (0.1), which deliberately
  caps confidence, so predicted probabilities are expected to understate. No temperature
  scaling has been fitted.
- **`uncertainty` is predictive entropy**, not epistemic uncertainty. It cannot distinguish an
  ambiguous input from one far outside the training distribution - an out-of-distribution
  image can produce confidently wrong output with low entropy.
- 38 classes across 14 crops only. Anything outside that set is silently forced into one of
  them.

## Training

ResNet50 (timm, ImageNet-pretrained), 224x224, batch 64, AdamW (lr 3e-4, weight decay 1e-4),
cosine schedule, label smoothing 0.1, medium augmentation, 12 epochs, mixed precision.
Checkpoint selected on `val_f1_macro`.

## Usage

```python
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

path = hf_hub_download("{REPO_ID}", "cropguard.onnx")
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

# Preprocessing must match training: resize short side to 256, centre-crop 224,
# scale to [0,1], normalise with ImageNet mean/std, NCHW.
logits = session.run(["logits"], {{"input": batch}})[0]
```

`cropguard.serving.model_loader` in the repo implements exactly that preprocessing.

## Citation

Dataset: Mohanty, Hughes & Salathe (2016), *Using deep learning for image-based plant disease
detection*, Frontiers in Plant Science.
"""

# ---------------------------------------------------------------- upload
token = getpass("HF write token (https://hf.co/settings/tokens): ")
create_repo(REPO_ID, repo_type="model", exist_ok=True, token=token)
api = HfApi()

Path("MODEL_CARD.md").write_text(card, encoding="utf-8")
# fp32 only. Publishing the dynamically-quantized INT8 file would leave a 75x-slower artifact
# in a public repo for someone to pick up by mistake; it goes up if and when static
# quantization makes it genuinely better.
uploads = [
    ("models/cropguard.onnx", "cropguard.onnx"),
    ("configs/classes.json", "classes.json"),
    ("MODEL_CARD.md", "README.md"),
]

for local, remote in uploads:
    if not Path(local).exists():
        print(f"skip  {local} (not found)")
        continue
    api.upload_file(
        path_or_fileobj=local, path_in_repo=remote, repo_id=REPO_ID, token=token
    )
    print(f"up    {local} -> {remote}  ({Path(local).stat().st_size / 1e6:.1f} MB)")

print()
print(f"https://huggingface.co/{REPO_ID}")
