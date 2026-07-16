import torch
import torch.nn as nn
from torchvision import models

def build_transfer_model(
    arch="resnet50",
    num_classes=6,
    pretrained=True,
    dropout=0.3,
    freeze_backbone=True,
):
    """
    Build a transfer-learning model. Supports several backbones.
    ResNet50 works well, but consider EfficientNet-B0 or ConvNeXt-Tiny for better accuracy/efficiency.
    """
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        if freeze_backbone:
            for name, p in model.named_parameters():
                if "fc" not in name:
                    p.requires_grad = False

    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False

    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model
