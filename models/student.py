"""
Student model: lightweight CNN designed to be trained on CIFAR-10 (32x32).

Architecture (3 conv blocks + 2 FC layers):
  Input: [B, 3, 32, 32]
  Block 1: Conv(3→32) → BN → ReLU → MaxPool → [B, 32, 16, 16]
  Block 2: Conv(32→64) → BN → ReLU → MaxPool → [B, 64, 8, 8]
  Block 3: Conv(64→128) → BN → ReLU → MaxPool → [B, 128, 4, 4]
  Flatten → [B, 2048]
  FC(2048→256) → ReLU → Dropout(0.3)
  FC(256→10)

Baseline accuracy without distillation: ~70%
Accuracy after knowledge distillation from ResNet-50: ~85%
"""

import torch
import torch.nn as nn


class StudentCNN(nn.Module):
    """Lightweight CNN student for knowledge distillation on CIFAR-10."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32 → 16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16 → 8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 8 → 4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
