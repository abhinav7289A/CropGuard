"""The serving preprocess path is a hand-written NumPy mirror of torchvision's
eval_transforms — these tests pin its contract so the two cannot silently diverge."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from cropguard.serving.model_loader import IMAGENET_MEAN, IMAGENET_STD, preprocess, softmax


def _encode(size: tuple[int, int], color: str = "green", fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize("size", [(256, 256), (640, 480), (300, 900), (128, 128)])
def test_preprocess_always_returns_nchw_224(size):
    output = preprocess(_encode(size))
    assert output.shape == (1, 3, 224, 224)
    assert output.dtype == np.float32


def test_preprocess_applies_imagenet_normalization():
    output = preprocess(_encode((256, 256), color="white"))
    expected = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
    for channel in range(3):
        assert output[0, channel].mean() == pytest.approx(expected[channel], abs=1e-4)


def test_preprocess_handles_grayscale_and_rgba():
    grayscale = io.BytesIO()
    Image.new("L", (256, 256), 128).save(grayscale, format="PNG")
    assert preprocess(grayscale.getvalue()).shape == (1, 3, 224, 224)

    rgba = io.BytesIO()
    Image.new("RGBA", (256, 256), (10, 20, 30, 255)).save(rgba, format="PNG")
    assert preprocess(rgba.getvalue()).shape == (1, 3, 224, 224)


def test_preprocess_matches_torchvision_eval_transforms():
    """Guards the serving/training skew that would silently degrade accuracy in production."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from cropguard.data.transforms import eval_transforms

    image = Image.effect_noise((320, 240), 64).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # lossless, so both paths see identical pixels

    ours = preprocess(buffer.getvalue())
    theirs = eval_transforms(224)(image).unsqueeze(0).numpy()

    assert ours.shape == theirs.shape
    # Resampling kernels differ slightly (PIL bilinear vs. torchvision antialias);
    # a tight mean gap is what matters for inference parity.
    assert np.abs(ours - theirs).mean() < 0.05


def test_softmax_is_normalized_and_overflow_safe():
    probs = softmax(np.array([1000.0, 1001.0, 999.0], dtype=np.float32))
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)
    assert probs.argmax() == 1
