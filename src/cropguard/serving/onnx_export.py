"""Export a trained Lightning checkpoint to ONNX (+ optional INT8 quantization).

Run in the TRAINING environment (needs torch):
    python -m cropguard.serving.onnx_export --ckpt checkpoints/resnet50-baseline/best-....ckpt \
        --out models/cropguard.onnx --quantize

Includes a parity check: max |logit diff| between PyTorch and ONNX Runtime on random
inputs must be < 1e-3.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export(ckpt_path: Path, out_path: Path, quantize: bool = False, image_size: int = 224) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch

    from cropguard.training.module import CropGuardModule

    module = CropGuardModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    module.eval()
    model = module.model

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        # torch>=2.6 defaults to the dynamo exporter, whose graph fails ONNX Runtime's
        # shape inference during INT8 quantization ("Inferred shape and existing shape
        # differ"). INT8 is what keeps the serving image inside the free tier, so we pin
        # the legacy TorchScript exporter until the dynamo path quantizes cleanly.
        dynamo=False,
        verbose=False,
    )

    # Parity check
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    test_input = torch.randn(4, 3, image_size, image_size)
    with torch.no_grad():
        torch_logits = model(test_input).numpy()
    ort_logits = session.run(["logits"], {"input": test_input.numpy()})[0]
    max_diff = float(np.abs(torch_logits - ort_logits).max())
    print(f"Exported {out_path} | parity max |diff| = {max_diff:.2e}")
    if max_diff >= 1e-3:
        raise RuntimeError(f"ONNX parity check failed: max diff {max_diff} >= 1e-3")

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process

        # Constant folding turns BatchNorm-folded conv weights into Constant *nodes*, but the
        # dynamic quantizer only rewrites *initializers* — without this pre-processing pass it
        # dies with "Expected onnx::Conv_N to be an initializer".
        prepared = out_path.with_suffix(".prepared.onnx")
        quant_pre_process(str(out_path), str(prepared), skip_symbolic_shape=False)

        quant_path = out_path.with_suffix(".int8.onnx")
        quantize_dynamic(str(prepared), str(quant_path), weight_type=QuantType.QInt8)
        prepared.unlink(missing_ok=True)
        print(
            f"Quantized -> {quant_path} "
            f"({quant_path.stat().st_size / 1e6:.1f} MB vs {out_path.stat().st_size / 1e6:.1f} MB)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", default=Path("models/cropguard.onnx"), type=Path)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    export(args.ckpt, args.out, args.quantize, args.image_size)


if __name__ == "__main__":
    main()
