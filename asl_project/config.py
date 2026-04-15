"""
config.py — Central configuration for the ASL alphabet classifier project.
All hyperparameters, paths, and constants live here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [MODEL_DIR, PLOT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
IMG_SIZE = 64                       # Resize all images to 64x64
NUM_CLASSES = 29                    # A-Z + space + delete + nothing
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

# Class names (alphabetical order matching folder names)
CLASS_NAMES = sorted([
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
])

# ──────────────────────────────────────────────
# Data Augmentation
# ──────────────────────────────────────────────
BRIGHTNESS_FACTOR = 0.2             # ±20% brightness
ROTATION_DEGREES = 10               # ±10° rotation
HORIZONTAL_FLIP_PROB = 0.5

# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
CONV_CHANNELS = [3, 32, 64, 128, 256]   # Input channels → 4 conv blocks
FC_HIDDEN = 512                          # Fully-connected hidden layer size
DROPOUT_RATE = 0.5

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4                     # DataLoader workers (set to 0 on Pi)
EARLY_STOP_PATIENCE = 5

# ──────────────────────────────────────────────
# Benchmarking
# ──────────────────────────────────────────────
BENCHMARK_FRAMES = 500              # Number of inference frames to average
WARMUP_FRAMES = 50                  # Warmup frames before measurement
TARGET_FPS = 15                     # Minimum acceptable FPS

# ──────────────────────────────────────────────
# ONNX Export
# ──────────────────────────────────────────────
ONNX_OPSET = 13

# ──────────────────────────────────────────────
# Device detection
# ──────────────────────────────────────────────
def get_device():
    """Return the best available PyTorch device."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
