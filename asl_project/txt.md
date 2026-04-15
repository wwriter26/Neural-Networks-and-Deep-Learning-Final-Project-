config.py — Single source of truth for all hyperparameters (64×64 images, 29 classes, 80/10/10 split, 4 conv blocks, 500-frame benchmark window, etc.)
dataset.py — Loads the Kaggle ASL Alphabet dataset with stratified splitting and your specified augmentation (horizontal flips, ±20% brightness, ±10° rotation, normalize to [0,1])
model.py — Lightweight CNN: 4 conv blocks (Conv3×3 → BatchNorm → ReLU → MaxPool2×2) → 2 FC layers → softmax over 29 classes. Deliberately small for Pi feasibility.
train.py — Full training loop with Adam optimizer, ReduceLROnPlateau scheduler, early stopping (patience=5), best-model checkpointing, and history logging to JSON
quantize.py — Produces all 3 variants: FP32 baseline, FP16 via .half(), and INT8 via torch.quantization with Conv-BN-ReLU fusion and calibration pass. Exports to ONNX for cross-platform compatibility.
benchmark.py — Measures all your paper's metrics: per-frame latency via time.perf_counter(), RAM via psutil, CPU%, FPS, power draw (via powermetrics on Mac / INA219 or sysfs estimate on Pi), and model size. Auto-detects platform or accepts --device pi.
visualize.py — Generates the 29×29 confusion matrix, top-1/top-5 accuracy bars, training curves, and cross-platform comparison charts (latency, FPS, RAM, model size).
run_all.py — Orchestrates everything end-to-end with --skip-* flags for partial runs.
To get started: download the Kaggle dataset into data/asl_alphabet_train/, then python run_all.py. For the Pi, just copy the ONNX files over and run python benchmark.py --device pi.