"""
Teacher model: ResNet-50 adapted for CIFAR-10 (32x32 inputs).

The standard ImageNet ResNet uses a 7x7 conv with stride=2 and a MaxPool,
which aggressively downsamples 224x224 → 56x56 before the residual blocks.
For 32x32 CIFAR images we replace that stem with a 3x3 conv (stride=1, no pool)
so the spatial resolution is preserved through the first stage.
"""

import torch.nn as nn
from torchvision import models


def get_teacher(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Return a ResNet-50 fine-tunable on CIFAR-10.

    Args:
        num_classes: Number of output classes (10 for CIFAR-10).
        pretrained:  Load ImageNet weights before adaptation.

    Returns:
        nn.Module ready for training/inference on 32x32 images.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    # --- CIFAR adaptation ---
    # Replace the 7x7 stem (designed for 224x224) with a 3x3 conv (no downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the initial MaxPool so we don't lose spatial resolution on tiny images
    model.maxpool = nn.Identity()

    # Replace the classification head for the target number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
