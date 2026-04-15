# ASL Alphabet Classifier — Cross-Platform Benchmarking

CNN-based American Sign Language alphabet classifier comparing inference
performance between a MacBook Pro (Apple M-series) and Raspberry Pi 4.

## Project Structure

```
asl_project/
├── config.py              # All hyperparameters and paths
├── dataset.py             # Dataset loading, splitting, augmentation
├── model.py               # CNN architecture definition
├── train.py               # Training loop with validation
├── quantize.py            # FP16 + INT8 quantization, ONNX export
├── benchmark.py           # Cross-platform inference benchmarking
├── visualize.py           # Plots: confusion matrix, bar charts, training curves
├── run_all.py             # End-to-end orchestrator
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

### 1. Download the dataset
Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
Extract to `./data/asl_alphabet_train/` so the structure is:
```
data/asl_alphabet_train/
├── A/
├── B/
├── ...
└── nothing/
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run everything
```bash
python run_all.py
```

Or run stages individually:
```bash
python train.py          # Train the model
python quantize.py       # Quantize + export ONNX
python benchmark.py      # Run inference benchmarks
python visualize.py      # Generate plots
```

## Raspberry Pi 4 Setup
```bash
pip install onnxruntime numpy psutil Pillow
python benchmark.py --device pi
```
Copy the ONNX files from `outputs/models/` to the Pi before running.
