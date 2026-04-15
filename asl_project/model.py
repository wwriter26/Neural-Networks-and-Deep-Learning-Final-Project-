"""
model.py — Lightweight CNN for ASL alphabet classification.

Architecture:
  4 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool)
  2 fully-connected layers with dropout
  Softmax output over 29 classes

Deliberately compact so it can run on a Raspberry Pi 4.
"""

import torch
import torch.nn as nn
import config


class ConvBlock(nn.Module):
    """Single convolutional block: Conv3x3 → BN → ReLU → MaxPool2x2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class ASLClassifier(nn.Module):
    """
    Lightweight CNN for 29-class ASL alphabet recognition.

    Input:  (B, 3, 64, 64)
    Output: (B, 29) logits
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        channels: list = None,
        fc_hidden: int = config.FC_HIDDEN,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()
        channels = channels or config.CONV_CHANNELS  # [3, 32, 64, 128, 256]

        # Build 4 convolutional blocks
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers.append(ConvBlock(channels[i], channels[i + 1]))
        self.features = nn.Sequential(*conv_layers)

        # After 4 rounds of 2x2 max-pool on 64x64 input:
        # 64 → 32 → 16 → 8 → 4  ⇒  spatial size = 4x4
        self._flat_size = channels[-1] * 4 * 4   # 256 * 16 = 4096

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.dequant(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(device: torch.device = None) -> ASLClassifier:
    """Instantiate the model and move to device."""
    device = device or config.get_device()
    model = ASLClassifier().to(device)
    print(f"Model created: {count_parameters(model):,} trainable parameters")
    print(f"Device: {device}")
    return model


if __name__ == "__main__":
    model = build_model(torch.device("cpu"))
    dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {count_parameters(model):,}")
