"""
MMA-MMTSA: Momentum Multimodal Attention + GAF + Temporal Segments.

Architecture:
  Depth frames -> 2D CNN -> f_depth  \
                                       -> MMA Fusion (per segment)
  IMU GAF      -> 2D CNN -> f_imu   /      |
                                       Segment Attention
                                            |
                                       Classification Head
"""

import torch
import torch.nn as nn

from .fusion import (
    MomentumAttentionFusion,
    AttentionFusionNoMomentum,
    GatedFusion,
    ConcatFusion,
    SegmentAttention,
)


class MMA_MMTSA(nn.Module):
    """
    Full MMA-MMTSA model:
      1. Depth -> 2D CNN encoder -> f_depth
      2. IMU GAF -> 2D CNN encoder -> f_imu
      3. MMA momentum attention fusion (per segment)
      4. Segment attention aggregation
      5. Classification head
    """

    def __init__(
        self,
        num_classes=27,
        feat_dim=256,
        n_seg=3,
        beta=0.99,
        dropout=0.3,
        fusion="attention",
        use_segments=True,
        use_momentum=True,
        imu_channels=6,
        depth_channels=3,
        encoder="lightcnn",
        convnext_model="convnextv2_atto",
        freeze_stages=3,
    ):
        super().__init__()
        self.n_seg = n_seg
        self.use_segments = use_segments

        # Encoders
        if encoder == "convnextv2":
            from .backbones import ConvNeXtV2Encoder
            self.enc_depth = ConvNeXtV2Encoder(
                in_channels=depth_channels, feat_dim=feat_dim,
                model_name=convnext_model, freeze_stages=freeze_stages,
            )
            self.enc_imu = ConvNeXtV2Encoder(
                in_channels=imu_channels, feat_dim=feat_dim,
                model_name=convnext_model, freeze_stages=freeze_stages,
            )
        else:
            from .backbones import LightCNN
            self.enc_depth = LightCNN(depth_channels, feat_dim)
            self.enc_imu = LightCNN(imu_channels, feat_dim)

        # Fusion
        if fusion == "attention":
            if use_momentum:
                self.fuse = MomentumAttentionFusion(feat_dim, beta)
            else:
                self.fuse = AttentionFusionNoMomentum(feat_dim)
        elif fusion == "gated":
            self.fuse = GatedFusion(feat_dim)
        else:
            self.fuse = ConcatFusion(feat_dim)

        # Segment attention
        self.seg_attn = (
            SegmentAttention(feat_dim) if (use_segments and n_seg > 1) else None
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        train_ = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    Model: {total:,} params ({train_:,} trainable)")

    def forward(self, depth, imu):
        if self.seg_attn is not None:
            return self._fwd_seg(depth, imu)
        return self._fwd_global(depth, imu)

    def _fwd_seg(self, depth, imu):
        """depth: (B,N,C,H,W)  imu: (B,N,C,H,W)"""
        B, N = depth.shape[:2]
        seg_list, alphas = [], []
        for i in range(N):
            fd = self.enc_depth(depth[:, i])
            fi = self.enc_imu(imu[:, i])
            f, a = self.fuse(fd, fi)
            seg_list.append(f)
            alphas.append(a)
        stack = torch.stack(seg_list, dim=1)
        glb, sw = self.seg_attn(stack)
        return self.head(glb), dict(alpha=alphas, seg_w=sw)

    def _fwd_global(self, depth, imu):
        """depth: (B,C,H,W)  imu: (B,C,H,W)"""
        fd = self.enc_depth(depth)
        fi = self.enc_imu(imu)
        f, a = self.fuse(fd, fi)
        return self.head(f), dict(alpha=a)
