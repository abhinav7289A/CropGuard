"""Guards the quantization mode.

The dynamic path shipped a graph full of `ConvInteger`, which ONNX Runtime's CPU backend has
no optimized kernel for: 1567 ms/image against 19 ms for fp32. Nothing caught it, because the
size reduction was real and the accuracy check passed — latency was simply never asserted.

Latency itself is too machine-dependent to assert in CI. The *op types* are not: a
Conv-dominated CNN must quantize to `QLinearConv`, and `ConvInteger` appearing at all is the
signature of the bug. That is what these tests pin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from cropguard.serving.quantize import summarize_ops  # noqa: E402

IMAGE_SIZE = 64


@pytest.fixture(scope="module")
def conv_model(tmp_path_factory) -> Path:
    """A small Conv-dominated graph — the architecture class where this bug bites."""
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 5),
    ).eval()

    path = tmp_path_factory.mktemp("q") / "model.onnx"
    torch.onnx.export(
        net,
        torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
        verbose=False,
    )
    return path


class _RandomCalibrationReader:
    """Stands in for real images; op types do not depend on calibration content."""

    def __init__(self, batches: int = 4):
        self._data = [
            {
                "input": np.random.default_rng(i).standard_normal(
                    (2, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
                )
            }
            for i in range(batches)
        ]
        self.rewind()

    def rewind(self):
        self._it = iter(self._data)

    def get_next(self):
        return next(self._it, None)


@pytest.fixture(scope="module")
def static_quantized(conv_model: Path) -> Path:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepared = conv_model.with_suffix(".prep.onnx")
    quant_pre_process(str(conv_model), str(prepared), skip_symbolic_shape=False)

    out = conv_model.with_suffix(".static.onnx")
    quantize_static(
        model_input=str(prepared),
        model_output=str(out),
        calibration_data_reader=_RandomCalibrationReader(),
        quant_format=QuantFormat.QOperator,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    return out


def test_static_quantization_emits_qlinearconv(static_quantized: Path):
    ops = summarize_ops(static_quantized)
    assert ops.get("QLinearConv", 0) > 0, f"expected QLinearConv, got {ops}"


def test_static_quantization_never_emits_convinteger(static_quantized: Path):
    """The regression guard. ConvInteger has no optimized CPU kernel in ONNX Runtime."""
    ops = summarize_ops(static_quantized)
    assert "ConvInteger" not in ops, (
        f"ConvInteger in the quantized graph — inference will be far slower than fp32. {ops}"
    )


def test_static_quantization_does_not_requantize_per_layer(static_quantized: Path):
    """Dynamic quantization inserts a DynamicQuantizeLinear before *every* conv, recomputing
    activation ranges on each inference. Static calibration should leave at most a couple."""
    ops = summarize_ops(static_quantized)
    assert ops.get("DynamicQuantizeLinear", 0) == 0
    assert ops.get("QuantizeLinear", 0) <= 2


def test_dynamic_quantization_is_the_thing_we_are_avoiding(conv_model: Path):
    """Pins the behaviour that motivated this module, so the contrast stays documented and
    a future ONNX Runtime that fixes it shows up as a failure here rather than silently."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = conv_model.with_suffix(".dynamic.onnx")
    quantize_dynamic(str(conv_model), str(out), weight_type=QuantType.QInt8)

    ops = summarize_ops(out)
    assert "ConvInteger" in ops, (
        "quantize_dynamic no longer produces ConvInteger — re-benchmark it against fp32; "
        "the reason we avoid it may no longer hold."
    )


def test_quantized_output_stays_close_to_float(conv_model: Path, static_quantized: Path):
    import onnxruntime as ort

    from cropguard.serving.model_loader import softmax

    inputs = np.random.default_rng(7).standard_normal(
        (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
    )
    run = lambda p: ort.InferenceSession(  # noqa: E731
        str(p), providers=["CPUExecutionProvider"]
    ).run(["logits"], {"input": inputs})[0]

    assert np.abs(softmax(run(conv_model)[0]) - softmax(run(static_quantized)[0])).max() < 0.2
