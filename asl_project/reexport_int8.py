"""Re-export a clean FP32 ONNX from best_model.pth, then quantize to INT8.

Bypasses the existing asl_model_fp32.onnx (which has a broken shape annotation
that trips ORT's shape inference).
"""
from pathlib import Path
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

import config
from model import ASLClassifier

clean_fp32 = config.MODEL_DIR / "asl_model_fp32_clean.onnx"
int8_out = config.MODEL_DIR / "asl_model_int8.onnx"

# Load trained weights
ckpt = torch.load(config.MODEL_DIR / "best_model.pth", map_location="cpu", weights_only=False)
model = ASLClassifier()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Export with fixed batch size (no dynamic axes) to avoid the shape-inference bug
dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
torch.onnx.export(
    model, dummy, str(clean_fp32),
    input_names=["input"], output_names=["output"],
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes=None,
    dynamo=False,
)
print(f"Exported clean FP32 → {clean_fp32}")

# Quantize to INT8
quantize_dynamic(
    model_input=str(clean_fp32),
    model_output=str(int8_out),
    weight_type=QuantType.QInt8,
)
print(f"Quantized INT8 → {int8_out}")
