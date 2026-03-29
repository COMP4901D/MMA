"""
ConvNeXt-V2 Encoder wrapper for multimodal HAR.

Loads a pretrained ConvNeXt-V2 backbone via timm and adapts it:
  - Supports arbitrary input channels (e.g. 3 for depth, 4 for RGBD, 6 for GAF)
  - Projects backbone output to a configurable feature dimension
  - Optional partial freezing for fine-tuning on small datasets

Available models (from smallest to largest):
  convnextv2_atto    —  3.7M params, 320-dim
  convnextv2_femto   —  5.2M,       384-dim
  convnextv2_pico    —  9.1M,       512-dim
  convnextv2_nano    — 15.6M,       640-dim
  convnextv2_tiny    — 28.6M,       768-dim
  convnextv2_base    — 88.7M,      1024-dim
  convnextv2_large   —197.9M,      1536-dim

Usage:
    encoder = ConvNeXtV2Encoder(in_channels=3, feat_dim=256, model_name='convnextv2_atto')
    features = encoder(x)  # x: (B, 3, H, W) → features: (B, 256)
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required for ConvNeXt-V2. Install via: pip install timm")


MODEL_REGISTRY = {
    "convnextv2_atto":  "convnextv2_atto.fcmae_ft_in1k",
    "convnextv2_femto": "convnextv2_femto.fcmae_ft_in1k",
    "convnextv2_pico":  "convnextv2_pico.fcmae_ft_in1k",
    "convnextv2_nano":  "convnextv2_nano.fcmae_ft_in1k",
    "convnextv2_tiny":  "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_base":  "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2_large": "convnextv2_large.fcmae_ft_in22k_in1k",
}


class ConvNeXtV2Encoder(nn.Module):
    """
    Pretrained ConvNeXt-V2 backbone → global feature vector.

    Args:
        in_channels:   Number of input channels (3=RGB/Depth, 4=RGBD, 6=GAF).
        feat_dim:      Output feature dimension after projection.
        model_name:    Short name (e.g. 'convnextv2_atto') or full timm name.
        pretrained:    Whether to load pretrained ImageNet weights.
        freeze_stages: Number of stages to freeze (0–3). 0 = train all.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 256,
        model_name: str = "convnextv2_atto",
        pretrained: bool = True,
        freeze_stages: int = 0,
    ):
        super().__init__()
        timm_name = MODEL_REGISTRY.get(model_name, model_name)

        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features

        if in_channels != 3:
            old_conv = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if pretrained and in_channels <= 3:
                with torch.no_grad():
                    self.backbone.stem[0].weight[:, :in_channels] = old_conv.weight[:, :in_channels]
            elif pretrained and in_channels > 3:
                with torch.no_grad():
                    self.backbone.stem[0].weight[:, :3] = old_conv.weight
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        self.backbone.stem[0].weight[:, c:c+1] = mean_w

        self.fc = nn.Sequential(
            nn.Linear(backbone_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
        )

        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, n: int):
        """Freeze the stem and the first n stages."""
        for param in self.backbone.stem.parameters():
            param.requires_grad = False
        for i in range(min(n, len(self.backbone.stages))):
            for param in self.backbone.stages[i].parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, feat_dim)"""
        features = self.backbone(x)
        return self.fc(features)
