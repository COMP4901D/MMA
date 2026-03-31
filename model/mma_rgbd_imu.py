"""
MultimodalMMA: RGB-D + IMU multimodal HAR with Momentum Mamba.

Architecture:
  RGB-D (N x 4 x H x W) -> RGBDEncoder (spatial+temporal) --+
                                                              |-> Fusion -> Head -> 27 classes
  IMU   (T x 6)          -> IMUEncoder (Conv1D+Mamba)      --+

Fusion modes (ported from skel_imu best practices):
  - "attention"/"gated": AttentionPool per branch -> DimGatedFusion
  - "cross_mamba": Modality embeddings -> concat -> shared MambaBlock -> AttentionPool
  - "concat": AttentionPool per branch -> MLP

Modality Dropout Schedules:
  - "none": disabled (default, backward-compatible)
  - "fixed": fixed-rate dropout at md_max_p
  - "curriculum": warmup phase (no dropout) then linear ramp to md_max_p
  - "simultaneous": 3 forward passes (full, rgbd-only, imu-only) every iteration
"""

import random

import torch
import torch.nn as nn

from .encoders import RGBDEncoder, IMUEncoder
from .mamba import MomentumMambaBlock
from .mma_skel_imu import AttentionPool, DimGatedFusion, AuxHead


class MultimodalMMA(nn.Module):
    """RGBD+IMU multimodal model with cross_mamba fusion and auxiliary loss."""

    def __init__(
        self,
        num_classes: int = 27,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
        fusion: str = "cross_mamba",
        n_heads: int = 4,
        encoder: str = "default",
        convnext_model: str = "convnextv2_atto",
        freeze_stages: int = 3,
        freeze: str = "all",
        temporal_velocity: bool = False,
        in_channels: int = 4,
        imu_encoder: str = "conv1d",
        gaf_size: int = 64,
        aux_weight: float = 0.0,
        # Modality dropout parameters
        md_schedule: str = "none",
        md_max_p: float = 0.3,
        md_warmup_frac: float = 0.3,
        md_lambda: float = 0.3,
        use_missing_token: bool = True,
        modality_dropout: float = 0.0,  # legacy compat
        # Frame differencing parameters
        use_frame_diff: bool = False,
        diff_channels: str = "all",
        temporal_diff_mode: str = "none",
        # Feature-level modality dropout (MD-Drop)
        md_drop_imu: float = 0.0,
        md_drop_rgbd: float = 0.0,
        # Cross-Modal Alignment Regularization (CMAR)
        cmar_weight: float = 0.0,
        cmar_proj_dim: int = 64,
    ):
        super().__init__()
        self.fusion_mode = fusion
        self.aux_weight = aux_weight
        self._aux_logits = None
        self.d_model = d_model

        # Modality dropout config
        self.md_schedule = md_schedule if md_schedule != "none" else (
            "fixed" if modality_dropout > 0 else "none"
        )
        self.md_max_p = md_max_p if md_schedule != "none" else modality_dropout
        self.md_warmup_frac = md_warmup_frac
        self.md_lambda = md_lambda
        self.md_drop_imu = md_drop_imu
        self.md_drop_rgbd = md_drop_rgbd
        self.cmar_weight = cmar_weight
        self._cmar_loss = None

        # Epoch tracking (set externally by training loop)
        self.current_epoch = 0
        self.total_epochs = 1

        # Learnable missing tokens
        if use_missing_token and self.md_schedule != "none":
            self.missing_token_rgbd = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.missing_token_imu = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            self.register_buffer("missing_token_rgbd", torch.zeros(1, 1, d_model))
            self.register_buffer("missing_token_imu", torch.zeros(1, 1, d_model))

        mamba_kw = dict(
            d_model=d_model, n_layers=n_layers, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout,
            use_momentum=use_momentum, momentum_mode=momentum_mode,
            alpha_init=alpha_init, beta_init=beta_init,
        )
        cross_kw = dict(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            use_momentum=use_momentum, momentum_mode=momentum_mode,
            alpha_init=alpha_init, beta_init=beta_init,
        )

        # Map encoder name
        enc_name = encoder
        if enc_name == "convnextv2":
            enc_name = "convnextv2"
        elif enc_name in ("pretrained", "pretrained_cnn", "resnet18"):
            enc_name = "pretrained_cnn"
        else:
            enc_name = "spatial_cnn"

        self.rgbd_enc = RGBDEncoder(
            **mamba_kw,
            encoder=enc_name,
            convnext_model=convnext_model,
            freeze_stages=freeze_stages,
            freeze=freeze,
            temporal_velocity=temporal_velocity,
            in_channels=in_channels,
            temporal_diff_mode=temporal_diff_mode,
            use_frame_diff=use_frame_diff,
            diff_channels=diff_channels,
        )
        self.imu_enc = IMUEncoder(
            in_channels=6, **mamba_kw,
            encoder=imu_encoder,
            convnext_model=convnext_model,
            freeze_stages=freeze_stages,
            gaf_size=gaf_size,
        )

        # --- Fusion ---
        if fusion == "cross_mamba":
            self.mod_embed_rgbd = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.mod_embed_imu = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.cross_block = MomentumMambaBlock(**cross_kw)
            self.pool = AttentionPool(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion in ("attention", "gated"):
            self.rgbd_pool = AttentionPool(d_model)
            self.imu_pool = AttentionPool(d_model)
            self.fusion_layer = DimGatedFusion(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion == "concat":
            self.rgbd_pool = AttentionPool(d_model)
            self.imu_pool = AttentionPool(d_model)
            self.fusion_layer = None
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        # Auxiliary per-modality heads
        if aux_weight > 0:
            self.aux_rgbd = AuxHead(d_model, num_classes, dropout)
            self.aux_imu = AuxHead(d_model, num_classes, dropout)

        # CMAR projection heads: map each branch to shared alignment space
        if cmar_weight > 0:
            self.cmar_proj_rgbd = nn.Linear(d_model, cmar_proj_dim)
            self.cmar_proj_imu = nn.Linear(d_model, cmar_proj_dim)

    # ------------------------------------------------------------------
    #  Modality dropout helpers
    # ------------------------------------------------------------------

    def get_md_prob(self) -> float:
        """Current modality dropout probability based on schedule and epoch."""
        if self.md_schedule == "none":
            return 0.0
        if self.md_schedule == "fixed" or self.md_schedule == "simultaneous":
            return self.md_max_p
        # curriculum
        warmup_end = int(self.md_warmup_frac * self.total_epochs)
        if self.current_epoch < warmup_end:
            return 0.0
        remaining = self.total_epochs - warmup_end
        if remaining <= 0:
            return self.md_max_p
        progress = (self.current_epoch - warmup_end) / remaining
        return min(self.md_max_p, self.md_max_p * progress)

    def _apply_missing_token(self, feat, token, ref_feat):
        """Replace feat with broadcasted missing token matching ref_feat shape."""
        B, T, _ = ref_feat.shape
        return token.expand(B, T, -1)

    def _fuse_and_classify(self, fv, fi):
        """Run fusion + classification head on encoded features."""
        if self.fusion_mode == "cross_mamba":
            fv2 = fv + self.mod_embed_rgbd
            fi2 = fi + self.mod_embed_imu
            fused = torch.cat([fv2, fi2], dim=1)
            fused = self.cross_block(fused)
            pooled = self.pool(fused)
        elif self.fusion_mode in ("attention", "gated"):
            pv = self.rgbd_pool(fv)
            pi = self.imu_pool(fi)
            pooled = self.fusion_layer(pv, pi)
        else:  # concat
            pv = self.rgbd_pool(fv)
            pi = self.imu_pool(fi)
            pooled = torch.cat([pv, pi], dim=-1)
        return self.head(pooled)

    def forward(self, rgbd, imu):
        """
        rgbd: (B, N, 4, H, W) or None (missing modality)
        imu:  (B, T, 6) or None (missing modality)
        -> logits (B, num_classes)
           OR dict with extra loss terms (simultaneous mode, training only)
        """
        # --- Encode ---
        fv = self.rgbd_enc(rgbd) if rgbd is not None else None
        fi = self.imu_enc(imu) if imu is not None else None

        # Need a reference for shape when building missing tokens
        ref = fv if fv is not None else fi

        # Replace None with missing tokens (inference with missing modality)
        if fv is None:
            fv = self._apply_missing_token(None, self.missing_token_rgbd, ref)
        if fi is None:
            fi = self._apply_missing_token(None, self.missing_token_imu, ref)

        # --- Simultaneous mode (training only) ---
        if self.training and self.md_schedule == "simultaneous":
            return self._forward_simultaneous(fv, fi)

        # --- Feature-level modality dropout (MD-Drop) ---
        if self.training and (self.md_drop_imu > 0 or self.md_drop_rgbd > 0):
            r = random.random()
            if r < self.md_drop_imu:
                fi = torch.zeros_like(fi)
            elif r < self.md_drop_imu + self.md_drop_rgbd:
                fv = torch.zeros_like(fv)

        # --- CMAR: Cross-Modal Alignment Regularization ---
        # Align projected RGBD and IMU sequences before fusion.
        # Loss is stored for the trainer to add to the total loss.
        self._cmar_loss = None
        if self.training and self.cmar_weight > 0:
            # Mean-pool over the time dimension for sequence-level alignment
            # (avoids needing equal sequence lengths between RGBD and IMU)
            fv_proj = self.cmar_proj_rgbd(fv.mean(dim=1))  # (B, proj_dim)
            fi_proj = self.cmar_proj_imu(fi.mean(dim=1))   # (B, proj_dim)
            self._cmar_loss = torch.mean((fv_proj - fi_proj) ** 2)

        # --- Auxiliary outputs (full modality) ---
        self._aux_logits = None
        if self.aux_weight > 0 and self.training:
            self._aux_logits = (self.aux_rgbd(fv), self.aux_imu(fi))

        # --- Random modality dropout (training, fixed/curriculum) ---
        if self.training and self.md_schedule in ("fixed", "curriculum"):
            p = self.get_md_prob()
            if p > 0:
                r = torch.rand(1).item()
                if r < p / 2:
                    # Drop RGBD
                    fv = self._apply_missing_token(fv, self.missing_token_rgbd, fi)
                    # Disable RGBD aux loss
                    if self._aux_logits is not None:
                        self._aux_logits = (None, self._aux_logits[1])
                elif r < p:
                    # Drop IMU
                    fi = self._apply_missing_token(fi, self.missing_token_imu, fv)
                    if self._aux_logits is not None:
                        self._aux_logits = (self._aux_logits[0], None)

        return self._fuse_and_classify(fv, fi)

    def _forward_simultaneous(self, fv, fi):
        """Three-pass forward: full, rgbd-only, imu-only. Returns dict."""
        # Full pass (both modalities present)
        self._aux_logits = None
        if self.aux_weight > 0:
            self._aux_logits = (self.aux_rgbd(fv), self.aux_imu(fi))
        logits_full = self._fuse_and_classify(fv, fi)

        # RGBD-only (IMU replaced with missing token)
        fi_miss = self._apply_missing_token(None, self.missing_token_imu, fv)
        logits_rgbd = self._fuse_and_classify(fv, fi_miss)

        # IMU-only (RGBD replaced with missing token)
        fv_miss = self._apply_missing_token(None, self.missing_token_rgbd, fi)
        logits_imu = self._fuse_and_classify(fv_miss, fi)

        return {
            "logits": logits_full,
            "logits_rgbd_only": logits_rgbd,
            "logits_imu_only": logits_imu,
            "md_lambda": self.md_lambda,
        }
