"""
quantize.py — Model compression and ONNX export.

Produces three variants from the trained FP32 model:
  1. FP32 baseline  → ONNX
  2. FP16 (half)    → ONNX
  3. INT8 (static post-training quantization) → ONNX

All exported to outputs/models/ for cross-platform benchmarking.
"""

import os
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.quantization as quant
import onnx

import config
from model import ASLClassifier
from dataset import get_dataloaders


def load_trained_model(device: torch.device = None) -> ASLClassifier:
    """Load the best trained model from disk."""
    device = device or torch.device("cpu")
    ckpt_path = config.MODEL_DIR / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No trained model found at {ckpt_path}. Run train.py first.")

    model = ASLClassifier()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded trained model from {ckpt_path}")
    return model


def export_onnx(model: nn.Module, path: Path, opset: int = config.ONNX_OPSET,
                input_dtype: torch.dtype = torch.float32):
    """Export a PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE, dtype=input_dtype)

    # Move model and input to same device
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Validate
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Exported: {path.name} ({size_mb:.2f} MB)")
    return size_mb


def quantize_fp16(model: nn.Module) -> nn.Module:
    """Convert model to FP16 (half precision)."""
    model_fp16 = copy.deepcopy(model).cpu().half()
    print("  FP16 conversion complete")
    return model_fp16


def quantize_int8(model: nn.Module, calibration_loader) -> nn.Module:
    """
    Apply post-training static INT8 quantization.

    Uses a calibration pass over real data to determine optimal
    quantization parameters for each layer.
    """
    model_int8 = copy.deepcopy(model).cpu()
    model_int8.eval()

    # Fuse Conv-BN-ReLU blocks for better quantization
    # We need to fuse within each ConvBlock's sequential
    for i, block in enumerate(model_int8.features):
        # Each ConvBlock.block is Sequential(Conv2d, BN, ReLU, MaxPool)
        # Fuse Conv+BN+ReLU (indices 0,1,2 in the sequential)
        torch.quantization.fuse_modules(
            block.block,
            ["0", "1", "2"],   # Conv2d, BatchNorm2d, ReLU
            inplace=True,
        )

    # Prepare for static quantization
    qengine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
    torch.backends.quantized.engine = qengine
    model_int8.qconfig = quant.get_default_qconfig(qengine)
    model_int8.classifier.qconfig = None
    quant.prepare(model_int8, inplace=True)

    # Calibration pass
    print("  Running calibration pass...")
    num_batches = min(50, len(calibration_loader))  # Use up to 50 batches
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_batches:
                break
            model_int8(images)

    # Convert to quantized model
    quant.convert(model_int8, inplace=True)
    print("  INT8 static quantization complete")
    return model_int8


def export_int8_onnx(model_int8: nn.Module, path: Path):
    """
    Export INT8 quantized model.

    PyTorch quantized models don't export cleanly to ONNX in all cases,
    so we save the PyTorch model directly and also attempt ONNX export.
    """
    # Save PyTorch quantized model (always works)
    pt_path = path.with_suffix(".pt")
    torch.save(model_int8.state_dict(), pt_path)
    size_mb = os.path.getsize(pt_path) / (1024 * 1024)
    print(f"  Saved PyTorch INT8: {pt_path.name} ({size_mb:.2f} MB)")

    # Attempt ONNX export
    try:
        model_int8.eval()
        dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
        torch.onnx.export(
            model_int8, dummy, str(path),
            opset_version=config.ONNX_OPSET,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        onnx_size = os.path.getsize(path) / (1024 * 1024)
        print(f"  Exported ONNX INT8: {path.name} ({onnx_size:.2f} MB)")
        return onnx_size
    except Exception as e:
        print(f"  ⚠ ONNX export for INT8 failed ({e}); use PyTorch .pt file instead")
        return size_mb


def run_quantization():
    """Run the full quantization + export pipeline."""
    print("=" * 60)
    print("Model Quantization & ONNX Export")
    print("=" * 60)

    # Load base model
    model_fp32 = load_trained_model(device=torch.device("cpu"))

    # Get calibration data for INT8
    _, val_loader, _, _ = get_dataloaders(batch_size=32, num_workers=0)

    sizes = {}

    # ── FP32 Export ──
    print("\n[1/3] FP32 Baseline")
    fp32_path = config.MODEL_DIR / "asl_model_fp32.onnx"
    sizes["FP32"] = export_onnx(model_fp32, fp32_path)

    # ── FP16 Export ──
    print("\n[2/3] FP16 Half-Precision")
    model_fp16 = quantize_fp16(model_fp32)
    fp16_path = config.MODEL_DIR / "asl_model_fp16.onnx"
    sizes["FP16"] = export_onnx(model_fp16, fp16_path, input_dtype=torch.float16)

    # ── INT8 Export ──
    print("\n[3/3] INT8 Static Quantization")
    model_int8 = quantize_int8(model_fp32, val_loader)
    int8_path = config.MODEL_DIR / "asl_model_int8.onnx"
    sizes["INT8"] = export_int8_onnx(model_int8, int8_path)

    # Summary
    print(f"\n{'='*60}")
    print("Model Size Summary:")
    print(f"  FP32: {sizes['FP32']:.2f} MB")
    print(f"  FP16: {sizes['FP16']:.2f} MB  ({sizes['FP16']/sizes['FP32']*100:.0f}% of FP32)")
    print(f"  INT8: {sizes['INT8']:.2f} MB  ({sizes['INT8']/sizes['FP32']*100:.0f}% of FP32)")
    print(f"{'='*60}")

    return sizes


if __name__ == "__main__":
    run_quantization()
