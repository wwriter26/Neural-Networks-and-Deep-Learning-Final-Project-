"""
benchmark.py — Cross-platform inference benchmarking.

Measures per-frame latency, FPS, RAM usage, CPU utilization,
and power draw for each model variant on both MacBook Pro and Raspberry Pi 4.

Usage:
    python benchmark.py              # Auto-detect platform
    python benchmark.py --device mac
    python benchmark.py --device pi
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

import config

# ──────────────────────────────────────────────
# Attempt to import inference backends
# ──────────────────────────────────────────────
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def detect_platform() -> str:
    """Detect whether we're on macOS (MacBook Pro) or Linux (Raspberry Pi)."""
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        return "mac"
    elif system == "Linux" and ("aarch64" in machine or "armv" in machine):
        return "pi"
    elif system == "Linux":
        return "linux"
    return "unknown"


def get_model_paths():
    """Return dict of model variant name → file path."""
    models = {}
    for variant in ["fp32", "fp16", "int8"]:
        onnx_path = config.MODEL_DIR / f"asl_model_{variant}.onnx"
        pt_path = config.MODEL_DIR / f"asl_model_{variant}.pt"
        if onnx_path.exists():
            models[variant.upper()] = onnx_path
        elif pt_path.exists():
            models[variant.upper()] = pt_path
    return models


def load_onnx_session(model_path: Path):
    if not HAS_ORT:
        raise ImportError("onnxruntime not installed. pip install onnxruntime")

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    input_type = sess.get_inputs()[0].type
    print(f"  Loaded ONNX: {model_path.name}")
    print(f"  Input: {input_name} {input_shape} ({input_type})")
    return sess, input_name, input_type


def generate_dummy_input_from_type(input_type: str) -> np.ndarray:
    shape = (1, 3, config.IMG_SIZE, config.IMG_SIZE)
    if input_type == "tensor(float16)":
        return np.random.randn(*shape).astype(np.float16)
    return np.random.randn(*shape).astype(np.float32)


def measure_power_mac() -> float:
    """
    Estimate current power draw on macOS using powermetrics.
    Returns watts or -1 if unavailable.
    """
    try:
        result = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "smc", "-n", "1", "-i", "100"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "System Total" in line or "CPU Power" in line:
                # Parse something like "CPU Power: 3.45 W"
                parts = line.split(":")
                if len(parts) > 1:
                    val = "".join(c for c in parts[1] if c.isdigit() or c == ".")
                    if val:
                        return float(val)
    except Exception:
        pass
    return -1.0


def measure_power_pi() -> float:
    """
    Read power draw on Raspberry Pi.
    Checks for INA219 sensor or estimates from /sys.
    Returns watts or -1 if unavailable.
    """
    # Try INA219 (common USB power meter I2C sensor)
    try:
        from ina219 import INA219
        ina = INA219(shunt_ohms=0.1)
        ina.configure()
        return ina.power() / 1000.0  # mW → W
    except Exception:
        pass

    # Estimate from CPU frequency and temperature as rough proxy
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            freq_khz = int(f.read().strip())
        # Very rough estimate: Pi 4 draws ~3-7W under load
        # Scale linearly between idle (600MHz ≈ 3W) and max (1500MHz ≈ 6W)
        watts = 3.0 + (freq_khz - 600000) / (1500000 - 600000) * 3.0
        return round(watts, 2)
    except Exception:
        pass

    return -1.0


def benchmark_model(model_path: Path, variant: str, plat: str) -> dict:
    """
    Run inference benchmark for a single model variant.

    Returns dict with all metrics.
    """
    print(f"\n{'─'*50}")
    print(f"Benchmarking: {variant} on {plat}")
    print(f"{'─'*50}")

    results = {
        "variant": variant,
        "platform": plat,
        "model_file": model_path.name,
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
    }

    # ── Load model ──
    if model_path.suffix == ".onnx":
        sess, input_name, input_type = load_onnx_session(model_path)
        dummy_input = generate_dummy_input_from_type(input_type)

        def run_inference():
            return sess.run(None, {input_name: dummy_input})
    elif model_path.suffix == ".pt" and HAS_TORCH:
        # INT8 PyTorch fallback
        from model import ASLClassifier
        qengine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
        torch.backends.quantized.engine = qengine
        model = ASLClassifier()
        model.eval()
        for block in model.features:
            torch.quantization.fuse_modules(block.block, ["0", "1", "2"], inplace=True)
        model.qconfig = torch.quantization.get_default_qconfig(qengine)
        model.classifier.qconfig = None
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        dummy_input_pt = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)

        def run_inference():
            with torch.no_grad():
                return model(dummy_input_pt)
    else:
        print(f"  ⚠ Cannot load {model_path.name}, skipping")
        return results

    process = psutil.Process()

    # ── Warmup ──
    print(f"  Warming up ({config.WARMUP_FRAMES} frames)...")
    for _ in range(config.WARMUP_FRAMES):
        run_inference()

    # ── Measurement loop ──
    print(f"  Measuring ({config.BENCHMARK_FRAMES} frames)...")
    latencies = []
    ram_samples = []
    cpu_samples = []

    # Reset CPU percent counter
    psutil.cpu_percent(interval=None)

    for i in range(config.BENCHMARK_FRAMES):
        # RAM before
        mem_info = process.memory_info()
        ram_mb = mem_info.rss / (1024 * 1024)
        ram_samples.append(ram_mb)

        # Timed inference
        t0 = time.perf_counter()
        run_inference()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

        # CPU (sampled periodically to avoid overhead)
        if i % 10 == 0:
            cpu_pct = psutil.cpu_percent(interval=None)
            cpu_samples.append(cpu_pct)

    # ── Power measurement ──
    if plat == "mac":
        power_w = measure_power_mac()
    elif plat == "pi":
        power_w = measure_power_pi()
    else:
        power_w = -1.0

    # ── Compute statistics ──
    latencies = np.array(latencies)
    results.update({
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "fps_mean": float(1000.0 / np.mean(latencies)),
        "fps_achievable": float(1000.0 / np.percentile(latencies, 95)),
        "ram_mean_mb": float(np.mean(ram_samples)),
        "ram_peak_mb": float(np.max(ram_samples)),
        "cpu_mean_pct": float(np.mean(cpu_samples)) if cpu_samples else -1.0,
        "cpu_peak_pct": float(np.max(cpu_samples)) if cpu_samples else -1.0,
        "power_w": power_w,
        "meets_15fps": bool(1000.0 / np.mean(latencies) >= config.TARGET_FPS),
    })

    # Print summary
    print(f"  Latency:  {results['latency_mean_ms']:.2f} ± {results['latency_std_ms']:.2f} ms "
          f"(p95={results['latency_p95_ms']:.2f} ms)")
    print(f"  FPS:      {results['fps_mean']:.1f} mean / {results['fps_achievable']:.1f} p95-achievable")
    print(f"  RAM:      {results['ram_mean_mb']:.1f} MB mean / {results['ram_peak_mb']:.1f} MB peak")
    print(f"  CPU:      {results['cpu_mean_pct']:.1f}% mean")
    print(f"  Power:    {results['power_w']:.2f} W" if power_w > 0 else "  Power:    N/A")
    print(f"  Model:    {results['model_size_mb']:.2f} MB")
    print(f"  15+ FPS:  {'✓ YES' if results['meets_15fps'] else '✗ NO'}")

    return results


def run_benchmarks(plat: str = None):
    """Run benchmarks for all model variants on the detected/specified platform."""
    plat = plat or detect_platform()
    print(f"\n{'='*60}")
    print(f"Inference Benchmarking — Platform: {plat.upper()}")
    print(f"System: {platform.system()} {platform.machine()}")
    print(f"CPU: {psutil.cpu_count(logical=True)} logical cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
    print(f"{'='*60}")

    model_paths = get_model_paths()
    if not model_paths:
        print("No model files found in outputs/models/. Run quantize.py first.")
        return []

    all_results = []
    for variant, path in model_paths.items():
        try:
            result = benchmark_model(path, variant, plat)
            all_results.append(result)
        except Exception as e:
            print(f"  ✗ Error benchmarking {variant}: {e}")

    # Save results
    out_path = config.RESULTS_DIR / f"benchmark_{plat}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Variant':<8} {'Latency(ms)':<14} {'FPS':<10} {'RAM(MB)':<12} {'CPU%':<8} {'Size(MB)':<10} {'15FPS?'}")
    print(f"{'─'*80}")
    for r in all_results:
        print(
            f"{r['variant']:<8} "
            f"{r['latency_mean_ms']:>6.2f} ± {r['latency_std_ms']:<5.2f} "
            f"{r['fps_mean']:>7.1f}   "
            f"{r['ram_peak_mb']:>8.1f}    "
            f"{r['cpu_mean_pct']:>5.1f}  "
            f"{r['model_size_mb']:>7.2f}   "
            f"{'✓' if r['meets_15fps'] else '✗'}"
        )
    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Model Inference Benchmark")
    parser.add_argument("--device", choices=["mac", "pi", "auto"], default="auto",
                        help="Target device (default: auto-detect)")
    args = parser.parse_args()

    plat = args.device if args.device != "auto" else None
    run_benchmarks(plat)
