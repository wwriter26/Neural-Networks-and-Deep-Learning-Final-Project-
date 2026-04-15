"""
visualize.py — Generate all plots for the paper.

Produces:
  1. Training curves (loss + accuracy)
  2. 29×29 confusion matrix (FP32 baseline)
  3. Top-1 / Top-5 accuracy bar chart across variants
  4. Latency comparison bar chart (Mac vs Pi × variant)
  5. FPS comparison
  6. RAM usage comparison
  7. Model size comparison

All plots saved to outputs/plots/
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
from tqdm import tqdm

import config
from model import ASLClassifier
from dataset import get_dataloaders

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})


def plot_training_curves():
    """Plot training/validation loss and accuracy over epochs."""
    history_path = config.RESULTS_DIR / "training_history.json"
    if not history_path.exists():
        print("No training history found. Skipping training curves.")
        return

    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(epochs, h["train_loss"], "o-", label="Train Loss", markersize=4)
    ax1.plot(epochs, h["val_loss"], "s-", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, h["train_acc"], "o-", label="Train Acc", markersize=4)
    ax2.plot(epochs, h["val_acc"], "s-", label="Val Acc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = config.PLOT_DIR / "training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


@torch.no_grad()
def compute_predictions(model, loader, device):
    """Run model on loader and collect all predictions + true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="  Predicting", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    return np.array(all_preds), np.array(all_labels), all_probs


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save the 29×29 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    # Normalize for better visualization
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=False,       # Too many classes for annotations
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0, vmax=1,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (FP32 Baseline, Normalized)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    out = config.PLOT_DIR / "confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # Also save raw counts
    out_raw = config.PLOT_DIR / "confusion_matrix_raw.png"
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax2, annot_kws={"size": 5},
    )
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    ax2.set_title("Confusion Matrix (FP32 Baseline, Raw Counts)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig2.savefig(out_raw)
    plt.close(fig2)
    print(f"Saved: {out_raw}")

    return cm


def plot_accuracy_comparison(accuracies: dict):
    """Bar chart comparing top-1 and top-5 accuracy across variants."""
    variants = list(accuracies.keys())
    top1 = [accuracies[v]["top1"] for v in variants]
    top5 = [accuracies[v]["top5"] for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, top1, width, label="Top-1 Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width/2, top5, width, label="Top-5 Accuracy", color="#66BB6A")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy by Model Variant (zoomed)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend(loc="lower right")
    lo = min(min(top1), min(top5)) - 0.3
    ax.set_ylim(max(0, lo), 100.3)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels — small offset proportional to zoomed y-range
    yrange = 100.3 - max(0, lo)
    offset = yrange * 0.01
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = config.PLOT_DIR / "accuracy_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_benchmark_comparisons():
    """Generate benchmark comparison plots from saved JSON results."""
    mac_path = config.RESULTS_DIR / "benchmark_mac.json"
    pi_path = config.RESULTS_DIR / "benchmark_pi.json"

    platforms = {}
    if mac_path.exists():
        with open(mac_path) as f:
            platforms["MacBook Pro"] = json.load(f)
    if pi_path.exists():
        with open(pi_path) as f:
            platforms["Raspberry Pi 4"] = json.load(f)

    if not platforms:
        print("No benchmark results found. Skipping benchmark plots.")
        return

    # Organize data
    variants = ["FP32", "FP16", "INT8"]
    colors = {"MacBook Pro": "#2196F3", "Raspberry Pi 4": "#FF7043"}

    def get_metric(data, variant, metric):
        for r in data:
            if r["variant"] == variant:
                return r.get(metric, 0)
        return 0

    # ── Latency comparison ──
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(variants))
    width = 0.35
    for i, (plat_name, data) in enumerate(platforms.items()):
        vals = [get_metric(data, v, "latency_mean_ms") for v in variants]
        errs = [get_metric(data, v, "latency_std_ms") for v in variants]
        offset = (i - 0.5 * (len(platforms) - 1)) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs,
                      label=plat_name, color=colors.get(plat_name, f"C{i}"),
                      capsize=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Frame Inference Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(config.PLOT_DIR / "latency_comparison.png")
    plt.close(fig)
    print(f"Saved: latency_comparison.png")

    # ── FPS comparison ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (plat_name, data) in enumerate(platforms.items()):
        vals = [get_metric(data, v, "fps_mean") for v in variants]
        offset = (i - 0.5 * (len(platforms) - 1)) * width
        bars = ax.bar(x + offset, vals, width,
                      label=plat_name, color=colors.get(plat_name, f"C{i}"))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(y=config.TARGET_FPS, color="red", linestyle="--", alpha=0.7, label=f"Target ({config.TARGET_FPS} FPS)")
    ax.set_ylabel("Frames Per Second")
    ax.set_title("Inference Throughput (FPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(config.PLOT_DIR / "fps_comparison.png")
    plt.close(fig)
    print(f"Saved: fps_comparison.png")

    # ── RAM comparison ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (plat_name, data) in enumerate(platforms.items()):
        vals = [get_metric(data, v, "ram_peak_mb") for v in variants]
        offset = (i - 0.5 * (len(platforms) - 1)) * width
        bars = ax.bar(x + offset, vals, width,
                      label=plat_name, color=colors.get(plat_name, f"C{i}"))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Peak RAM (MB)")
    ax.set_title("Peak Memory Usage During Inference")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(config.PLOT_DIR / "ram_comparison.png")
    plt.close(fig)
    print(f"Saved: ram_comparison.png")

    # ── Model size bar chart (measure from disk, include ONNX sidecar .data files) ──
    import os
    def _size_mb(*paths):
        return sum(os.path.getsize(p) for p in paths if os.path.exists(p)) / (1024 * 1024)

    model_files = {
        "FP32": [config.MODEL_DIR / "asl_model_fp32.onnx",
                 config.MODEL_DIR / "asl_model_fp32.onnx.data"],
        "FP16": [config.MODEL_DIR / "asl_model_fp16.onnx",
                 config.MODEL_DIR / "asl_model_fp16.onnx.data"],
        "INT8": [config.MODEL_DIR / "asl_model_int8.onnx",
                 config.MODEL_DIR / "asl_model_int8.pt"],  # whichever exists
    }
    sizes = [_size_mb(*model_files[v]) for v in variants]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(variants, sizes, color=["#2196F3", "#FF9800", "#4CAF50"])
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f"{val:.2f} MB", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size by Quantization Level")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(config.PLOT_DIR / "model_size_comparison.png")
    plt.close(fig)
    print(f"Saved: model_size_comparison.png")

    # ── Power comparison ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (plat_name, data) in enumerate(platforms.items()):
        vals = [get_metric(data, v, "power_w") for v in variants]
        offset = (i - 0.5 * (len(platforms) - 1)) * width
        bars = ax.bar(x + offset, vals, width,
                      label=plat_name, color=colors.get(plat_name, f"C{i}"))
        for bar, val in zip(bars, vals):
            label = f"{val:.2f}" if val and val > 0 else "N/A"
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    label, ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Power (W)")
    ax.set_title("Power Draw During Inference")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(config.PLOT_DIR / "power_comparison.png")
    plt.close(fig)
    print(f"Saved: power_comparison.png")


def run_accuracy_evaluation():
    """Evaluate top-1 and top-5 accuracy for all variants and generate plots."""
    device = torch.device("cpu")  # Use CPU for fair comparison
    _, _, test_loader, class_names = get_dataloaders(batch_size=64, num_workers=0)

    accuracies = {}

    # ── FP32 ──
    print("\nEvaluating FP32...")
    model_fp32 = ASLClassifier()
    ckpt = torch.load(config.MODEL_DIR / "best_model.pth", map_location=device, weights_only=False)
    model_fp32.load_state_dict(ckpt["model_state_dict"])
    model_fp32.eval()

    preds, labels, probs = compute_predictions(model_fp32, test_loader, device)
    top1 = 100.0 * np.mean(preds == labels)
    top5 = 100.0 * top_k_accuracy_score(labels, probs, k=5, labels=range(config.NUM_CLASSES))
    accuracies["FP32"] = {"top1": round(top1, 2), "top5": round(top5, 2)}
    print(f"  FP32 — Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")

    # Confusion matrix for FP32
    plot_confusion_matrix(labels, preds, class_names)

    # ── FP16 ──
    print("\nEvaluating FP16...")
    model_fp16 = ASLClassifier().half()
    model_fp16.load_state_dict(ckpt["model_state_dict"], strict=False)
    model_fp16.eval()

    # FP16 needs float16 input
    preds16, labels16, probs16 = [], [], []
    model_fp16_float = model_fp16.float()  # Evaluate in float for stable softmax
    model_fp16_float.load_state_dict(ckpt["model_state_dict"])
    preds16, labels16, probs16 = compute_predictions(model_fp16_float, test_loader, device)
    top1_16 = 100.0 * np.mean(preds16 == labels16)
    top5_16 = 100.0 * top_k_accuracy_score(labels16, probs16, k=5, labels=range(config.NUM_CLASSES))
    accuracies["FP16"] = {"top1": round(top1_16, 2), "top5": round(top5_16, 2)}
    print(f"  FP16 — Top-1: {top1_16:.2f}%  Top-5: {top5_16:.2f}%")

    # ── INT8 ──
    print("\nEvaluating INT8...")
    try:
        import copy
        qengine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
        torch.backends.quantized.engine = qengine
        model_int8 = copy.deepcopy(ASLClassifier()).cpu()
        model_int8.eval()
        # Fuse modules
        for block in model_int8.features:
            torch.quantization.fuse_modules(block.block, ["0", "1", "2"], inplace=True)
        model_int8.qconfig = torch.quantization.get_default_qconfig(qengine)
        model_int8.classifier.qconfig = None
        torch.quantization.prepare(model_int8, inplace=True)
        # Quick calibration
        for i, (imgs, _) in enumerate(test_loader):
            if i >= 10:
                break
            model_int8(imgs.cpu())
        torch.quantization.convert(model_int8, inplace=True)

        # Load saved INT8 weights if available
        int8_pt = config.MODEL_DIR / "asl_model_int8.pt"
        if int8_pt.exists():
            model_int8.load_state_dict(torch.load(int8_pt, weights_only=True))

        preds8, labels8, probs8 = compute_predictions(model_int8, test_loader, torch.device("cpu"))
        top1_8 = 100.0 * np.mean(preds8 == labels8)
        top5_8 = 100.0 * top_k_accuracy_score(labels8, probs8, k=5, labels=range(config.NUM_CLASSES))
        accuracies["INT8"] = {"top1": round(top1_8, 2), "top5": round(top5_8, 2)}
        print(f"  INT8 — Top-1: {top1_8:.2f}%  Top-5: {top5_8:.2f}%")
    except Exception as e:
        print(f"  ⚠ INT8 evaluation failed: {e}")
        accuracies["INT8"] = {"top1": 0.0, "top5": 0.0}

    # Save accuracies
    acc_path = config.RESULTS_DIR / "accuracies.json"
    with open(acc_path, "w") as f:
        json.dump(accuracies, f, indent=2)
    print(f"\nAccuracies saved to {acc_path}")

    # Plot accuracy comparison
    plot_accuracy_comparison(accuracies)

    return accuracies


def generate_all_plots():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Plots")
    print("=" * 60)

    plot_training_curves()
    run_accuracy_evaluation()
    plot_benchmark_comparisons()

    print(f"\nAll plots saved to {config.PLOT_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
