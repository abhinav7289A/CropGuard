---
title: CropGuard
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# CropGuard — crop disease detection

A 38-class leaf disease classifier trained on PlantVillage, served as ONNX. Upload a leaf and
get a prediction, the top-3 alternatives, calibrated confidence and the measured latency.

**Code and full write-up:** https://github.com/abhinav7289A/CropGuard
**Weights:** [XiElonMAsk/cropguard-models](https://huggingface.co/XiElonMAsk/cropguard-models)

## What this demo is actually showing

**Pick a model.** Weights are pulled from the Hub on first use, so the first prediction after a
cold start is slower than the rest. Currently available: the deployed **fp32 baseline** and a
**statically quantised INT8** build of it — a quarter the size, and whether it is faster
depends entirely on whether the CPU underneath has VNNI instructions. On the laptop it was
measured on it was 3.2× *slower*. Run both here and find out which way it goes on this one.

**Compare two models on one image**, side by side, with per-model latency. The ConvNeXt-Tiny
challenger appears here too once its weights are published: it scores higher macro-F1 (0.9890
vs 0.9865) and marginally lower accuracy (0.9908 vs 0.9911), and a statistical A/B test on the
8,125-image holdout found no significant improvement (McNemar p = 0.837), so it was not
promoted. The two models agree on roughly 98.8% of holdout images.

**Toggle calibration.** The model is trained with label smoothing (ε = 0.1, K = 38), which
caps achievable softmax output at 0.9026 and leaves it systematically *under*-confident.
Temperature scaling fitted on the validation split (T = 0.591) cut expected calibration error
from 0.0895 to 0.0036. Turning the toggle off shows the raw number — the predicted class never
changes, because dividing logits by a positive scalar cannot reorder them.

## What it does not show

Every image in training and evaluation is a PlantVillage lab photograph: one leaf, plain
background, controlled lighting. **Performance on a phone photo taken in a real field is
unmeasured**, and the literature on this dataset reports large drops there. A confident answer
on your own garden photo should be read with that in mind.

The test accuracy of 99.11% is measured on a holdout split by *leaf identity* rather than at
random — 74.2% of images in the standard split share a physical leaf with training, and that
contamination had to be removed before any of these numbers meant anything.
