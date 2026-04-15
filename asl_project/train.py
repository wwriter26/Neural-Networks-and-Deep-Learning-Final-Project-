"""
train.py — Training loop with validation, early stopping, and checkpointing.

Saves:
  - Best model weights (best_model.pth)
  - Training history (training_history.json)
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import build_model, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Eval ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train(resume_from: str = None):
    """
    Full training pipeline.

    Args:
        resume_from: Path to a checkpoint .pth to resume from (optional).
    """
    device = config.get_device()
    print(f"Training on: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    # Model
    model = build_model(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    # Resume if requested
    start_epoch = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # History tracking
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [], "epoch_time_s": [],
    }

    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = config.MODEL_DIR / "best_model.pth"

    print(f"\n{'='*60}")
    print(f"Training for up to {config.NUM_EPOCHS} epochs")
    print(f"Early stopping patience: {config.EARLY_STOP_PATIENCE}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        elapsed = time.time() - t0

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time_s"].append(elapsed)

        print(
            f"Epoch {epoch+1:>2}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
            }, best_model_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
                break

    # ── Final test evaluation ──
    print(f"\n{'='*60}")
    print("Loading best model for test evaluation...")
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.2f}%")

    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["best_val_acc"] = best_val_acc
    history["total_params"] = count_parameters(model)

    # Save history
    history_path = config.RESULTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return model, history


if __name__ == "__main__":
    train()
