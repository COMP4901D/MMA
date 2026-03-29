"""
Multimodal fusion strategies.

Two families of fusion modules:
  1. MMA-MMTSA family (return 2-tuple: fused, alpha):
     - MomentumAttentionFusion, AttentionFusionNoMomentum
     - GatedFusion, ConcatFusion, SegmentAttention

  2. MMA-RGBD-IMU family (return single tensor):
     - CrossModalAttentionFusion, GatedFusionSimple
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


# ═══════════════════════════════════════════════════════════════
#  MMA-MMTSA family  (all return (fused, alpha) tuple)
# ═══════════════════════════════════════════════════════════════

class MomentumAttentionFusion(nn.Module):
    """
    EMA-smoothed modality attention.
    Uses exponential moving average of feature representations
    to compute stable attention weights, reducing training oscillation
    on small datasets.
    """

    def __init__(self, dim: int, beta: float = 0.99):
        super().__init__()
        self.beta = beta
        self.proj_a = nn.Linear(dim, dim)
        self.proj_b = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

        self.register_buffer("ema_a", torch.zeros(1, dim))
        self.register_buffer("ema_b", torch.zeros(1, dim))
        self.register_buffer("_init", torch.tensor(False))

    def forward(self, fa, fb):
        """fa, fb: (B, D). Returns: (fused (B,D), alpha scalar)."""
        ema_a_snap = self.ema_a.clone()
        ema_b_snap = self.ema_b.clone()

        if self.training:
            if not self._init:
                self.ema_a.copy_(fa.detach().mean(0, keepdim=True))
                self.ema_b.copy_(fb.detach().mean(0, keepdim=True))
                self._init.fill_(True)
            else:
                with torch.no_grad():
                    self.ema_a.mul_(self.beta).add_(
                        fa.detach().mean(0, keepdim=True), alpha=1 - self.beta
                    )
                    self.ema_b.mul_(self.beta).add_(
                        fb.detach().mean(0, keepdim=True), alpha=1 - self.beta
                    )

        ha = torch.tanh(self.proj_a(ema_a_snap))
        hb = torch.tanh(self.proj_b(ema_b_snap))
        alpha = torch.softmax(
            torch.cat([self.score(ha), self.score(hb)], dim=-1), dim=-1
        )
        fused = alpha[:, 0:1] * fa + alpha[:, 1:2] * fb
        return fused, alpha[0, 0].item()


class AttentionFusionNoMomentum(nn.Module):
    """Attention fusion without EMA (ablation baseline)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj_a = nn.Linear(dim, dim)
        self.proj_b = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, fa, fb):
        ha = torch.tanh(self.proj_a(fa.mean(0, keepdim=True)))
        hb = torch.tanh(self.proj_b(fb.mean(0, keepdim=True)))
        alpha = torch.softmax(
            torch.cat([self.score(ha), self.score(hb)], dim=-1), dim=-1
        )
        return alpha[:, 0:1] * fa + alpha[:, 1:2] * fb, alpha[0, 0].item()


class SegmentAttention(nn.Module):
    """
    Inter-segment additive attention (from MMTSA Eq.9-10).
    Weighted aggregation of N temporal segment features.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, seg_feats):
        """seg_feats: (B, N, D) -> global: (B, D), weights: (B, N)."""
        scores = self.w(seg_feats).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        out = (weights.unsqueeze(-1) * seg_feats).sum(1)
        return out, weights


class GatedFusion(nn.Module):
    """Simple single-layer gated fusion. Returns: (fused, alpha)."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, fa, fb):
        g = self.gate(torch.cat([fa, fb], -1))
        return g * fa + (1 - g) * fb, g.mean().item()


class ConcatFusion(nn.Module):
    """Concatenation + projection fusion. Returns: (fused, 0.5)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, fa, fb):
        return self.proj(torch.cat([fa, fb], -1)), 0.5


# ═══════════════════════════════════════════════════════════════
#  MMA-RGBD-IMU family  (return single tensor)
# ═══════════════════════════════════════════════════════════════

class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion.
    RGB-D queries attend to IMU -> enhanced_v
    IMU queries attend to RGB-D -> enhanced_i
    Followed by learned gate + FFN.
    Handles different sequence lengths (N_frames != T_imu).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_v2i = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_i2v = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_v = RMSNorm(d_model)
        self.norm_i = RMSNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ff = RMSNorm(d_model)

    def forward(self, feat_v, feat_i):
        """
        feat_v: (B, N, D)  — RGB-D temporal features
        feat_i: (B, T, D)  — IMU temporal features
        Returns: (B, D)    — fused pooled representation
        """
        ca_v, _ = self.attn_v2i(query=feat_v, key=feat_i, value=feat_i)
        ca_i, _ = self.attn_i2v(query=feat_i, key=feat_v, value=feat_v)

        ev = self.norm_v(feat_v + ca_v)
        ei = self.norm_i(feat_i + ca_i)

        pv = ev.mean(dim=1)
        pi = ei.mean(dim=1)

        g = self.gate(torch.cat([pv, pi], dim=-1))
        fused = g * pv + (1 - g) * pi

        return self.norm_ff(fused + self.ffn(fused))


class GatedFusionSimple(nn.Module):
    """Two-layer gated fusion (for RGBD+IMU). Returns: fused tensor."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, fv, fi):
        g = self.gate(torch.cat([fv, fi], dim=-1))
        return g * fv + (1 - g) * fi
