"""Pretrained ResNet18 backbone for RGBD feature extraction."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PretrainedCNN(nn.Module):
    """
    ResNet18 pretrained on ImageNet, adapted for 4-channel RGBD input.

    First conv modified: (3,64,7,7) -> (4,64,7,7)
      - RGB channels: copy pretrained weights
      - Depth channel: initialized with mean of RGB weights

    Freeze modes:
      - "all": entire ResNet frozen (pure feature extractor)
      - "partial": freeze layers 1-3, fine-tune layer4
      - "none": fine-tune everything

    Output: (B, d_model) per frame
    """

    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 128,
        freeze: str = "all",
    ):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv for in_channels (default 4 for RGBD)
        old_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        with torch.no_grad():
            # Copy pretrained weights for first 3 channels
            new_conv.weight[:, :3] = old_conv.weight
            # Initialize extra channels with mean of RGB weights
            if in_channels > 3:
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_conv.weight[:, c : c + 1] = mean_w

        self.conv1 = new_conv
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.proj = nn.Linear(512, d_model)

        # Freeze strategy
        self._apply_freeze(freeze)

    def _apply_freeze(self, freeze: str):
        if freeze == "all":
            for name, param in self.named_parameters():
                if not name.startswith("proj"):
                    param.requires_grad = False
        elif freeze == "partial":
            # Freeze everything except layer4 and proj
            for name, param in self.named_parameters():
                if name.startswith(("conv1", "bn1", "layer1", "layer2", "layer3")):
                    param.requires_grad = False

    def forward(self, x):
        """x: (B, C, H, W) -> (B, d_model)"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)  # (B, 512)
        return self.proj(x)
