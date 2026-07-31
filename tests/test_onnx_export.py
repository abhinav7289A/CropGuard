"""Export pipeline tests.

These are slower than the rest of the suite (they build and export a real backbone) but they
cover the step most likely to break silently on a torch upgrade: the exported graph must stay
quantizable, or the serving image outgrows the free tier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("onnxruntime")

from cropguard.serving.model_loader import CropGuardModel, softmax  # noqa: E402
from cropguard.serving.onnx_export import export  # noqa: E402

IMAGE_SIZE = 64  # keep the export fast; the graph structure is what matters
CLASSES = ["Potato___Late_blight", "Tomato___Early_blight", "Tomato___healthy"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """A minimal Lightning checkpoint that CropGuardModule.load_from_checkpoint accepts."""
    import pytorch_lightning as pl

    from cropguard.training.module import CropGuardModule

    cfg = {
        "model": {"name": "resnet18", "pretrained": False, "dropout": 0.2},
        "train": {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 1, "label_smoothing": 0.0},
    }
    module = CropGuardModule(cfg, num_classes=len(CLASSES))

    path = tmp_path_factory.mktemp("ckpt") / "smoke.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": {"cfg": cfg, "num_classes": len(CLASSES)},
            "pytorch-lightning_version": pl.__version__,
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def exported(checkpoint: Path, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("onnx") / "cropguard.onnx"
    export(checkpoint, out, quantize=True, image_size=IMAGE_SIZE)
    return out


def test_export_writes_both_float_and_int8_graphs(exported: Path):
    quantized = exported.with_suffix(".int8.onnx")
    assert exported.exists()
    assert quantized.exists(), "INT8 quantization did not produce a file"
    # INT8 weights are the reason the serving image fits the free tier.
    assert quantized.stat().st_size < exported.stat().st_size * 0.6


def test_exported_graph_uses_the_names_the_serving_loader_expects(exported: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["input"]
    assert [o.name for o in session.get_outputs()] == ["logits"]


def test_exported_graph_accepts_a_dynamic_batch(exported: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    for batch in (1, 4):
        logits = session.run(
            ["logits"], {"input": np.zeros((batch, 3, IMAGE_SIZE, IMAGE_SIZE), np.float32)}
        )[0]
        assert logits.shape == (batch, len(CLASSES))


def test_int8_graph_still_agrees_with_the_float_graph(exported: Path):
    import onnxruntime as ort

    inputs = np.random.default_rng(0).standard_normal(
        (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
    )
    run = lambda p: ort.InferenceSession(  # noqa: E731
        str(p), providers=["CPUExecutionProvider"]
    ).run(["logits"], {"input": inputs})[0]

    float_probs = softmax(run(exported)[0])
    int8_probs = softmax(run(exported.with_suffix(".int8.onnx"))[0])
    # Quantization perturbs logits; the probability distribution should stay close.
    assert np.abs(float_probs - int8_probs).max() < 0.1


def test_serving_loader_runs_the_quantized_graph(checkpoint: Path, tmp_path):
    """Full contract: exported INT8 graph -> CropGuardModel -> a well-formed prediction."""
    import io
    import json

    from PIL import Image

    out = tmp_path / "cropguard.onnx"
    export(checkpoint, out, quantize=True, image_size=224)

    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(CLASSES), encoding="utf-8")
    model = CropGuardModel(out.with_suffix(".int8.onnx"), classes_path, "v-test")

    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "green").save(buffer, format="JPEG")
    result = model.predict(buffer.getvalue())

    assert result["predicted_class"] in CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_version"] == "v-test"
    assert len(result["top_k"]) == 3


def test_parity_check_rejects_a_mismatched_graph(checkpoint: Path, tmp_path, monkeypatch):
    """The export must fail loudly rather than ship a graph that disagrees with PyTorch."""
    real_export = torch.onnx.export

    def corrupting_export(model, *args, **kwargs):
        # Shift the weights *after* capture, so ONNX and PyTorch necessarily disagree.
        result = real_export(model, *args, **kwargs)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(10.0)
        return result

    monkeypatch.setattr(torch.onnx, "export", corrupting_export)

    with pytest.raises(RuntimeError, match="parity check failed"):
        export(checkpoint, tmp_path / "bad.onnx", quantize=False, image_size=IMAGE_SIZE)
