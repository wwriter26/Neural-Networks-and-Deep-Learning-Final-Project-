"""
run_all.py — End-to-end orchestrator.

Runs the complete pipeline in order:
  1. Train the CNN
  2. Quantize (FP16, INT8) + export to ONNX
  3. Benchmark on current platform
  4. Generate all plots

Usage:
    python run_all.py                 # Run everything
    python run_all.py --skip-train    # Skip training (use existing model)
    python run_all.py --skip-bench    # Skip benchmarking
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="ASL Classifier — Full Pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-quant", action="store_true", help="Skip quantization")
    parser.add_argument("--skip-bench", action="store_true", help="Skip benchmarking")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--device", choices=["mac", "pi", "auto"], default="auto",
                        help="Benchmark device")
    args = parser.parse_args()

    total_start = time.time()

    # ── Stage 1: Training ──
    if not args.skip_train:
        print("\n" + "█" * 60)
        print("  STAGE 1: TRAINING")
        print("█" * 60)
        from train import train
        train()
    else:
        print("\n⏭ Skipping training (--skip-train)")

    # ── Stage 2: Quantization + ONNX Export ──
    if not args.skip_quant:
        print("\n" + "█" * 60)
        print("  STAGE 2: QUANTIZATION & ONNX EXPORT")
        print("█" * 60)
        from quantize import run_quantization
        run_quantization()
    else:
        print("\n⏭ Skipping quantization (--skip-quant)")

    # ── Stage 3: Benchmarking ──
    if not args.skip_bench:
        print("\n" + "█" * 60)
        print("  STAGE 3: BENCHMARKING")
        print("█" * 60)
        from benchmark import run_benchmarks
        plat = args.device if args.device != "auto" else None
        run_benchmarks(plat)
    else:
        print("\n⏭ Skipping benchmarking (--skip-bench)")

    # ── Stage 4: Visualization ──
    if not args.skip_plots:
        print("\n" + "█" * 60)
        print("  STAGE 4: VISUALIZATION")
        print("█" * 60)
        from visualize import generate_all_plots
        generate_all_plots()
    else:
        print("\n⏭ Skipping plots (--skip-plots)")

    # ── Done ──
    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
