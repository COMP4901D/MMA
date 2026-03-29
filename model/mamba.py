"""
Momentum Mamba: Selective State-Space Model with momentum augmentation.

Core recurrence with three modes:
  - vanilla (use_momentum=False): standard first-order SSM (Mamba baseline)
  - real momentum: second-order SSM with real-valued momentum term
  - complex momentum: second-order SSM with complex momentum (oscillatory memory)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class MomentumSSM(nn.Module):
    """
    Selective SSM with optional momentum augmentation.

    Vanilla:   h_n = A_bar_n * h_{n-1} + B_bar_n * x_n
    Real Mom:  v_n = beta * v_{n-1} + alpha * B_bar_n * x_n;  h_n = A_bar_n * h_{n-1} + v_n
    Complex:   beta = rho * exp(i*theta), same update in complex domain
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 64,
        dt_rank: int = None,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.use_momentum = use_momentum
        self.momentum_mode = momentum_mode
        self.dt_rank = dt_rank or max(d_inner // 16, 1)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(d_inner, -1).clone()
        )
        self.D = nn.Parameter(torch.ones(d_inner))

        self.x_proj = nn.Linear(
            d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            )
            self.dt_proj.bias.copy_(torch.log(torch.exp(dt_init) - 1.0))

        if use_momentum:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            beta_logit = math.log(beta_init / (1.0 - beta_init + 1e-8))
            self.beta_logit = nn.Parameter(torch.tensor(beta_logit))
            if momentum_mode == "complex":
                self.theta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_inner) -> (B, L, d_inner)"""
        B, L, _ = x.shape

        proj = self.x_proj(x)
        dt_in, Bp, Cp = proj.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_in))

        A = -torch.exp(self.A_log)
        A_bar = torch.exp(dt.unsqueeze(-1) * A[None, None])
        B_bar = dt.unsqueeze(-1) * Bp.unsqueeze(2)

        if self.use_momentum:
            y = self._scan_momentum(x, A_bar, B_bar, Cp)
        else:
            y = self._scan_vanilla(x, A_bar, B_bar, Cp)

        return y + self.D * x

    def _scan_vanilla(self, x, A_bar, B_bar, C):
        Bs, L, D = x.shape
        h = x.new_zeros(Bs, D, self.d_state)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t, :, None]
            ys.append((h * C[:, t, None, :]).sum(-1))
        return torch.stack(ys, dim=1)

    def _scan_momentum(self, x, A_bar, B_bar, C):
        Bs, L, D = x.shape
        h = x.new_zeros(Bs, D, self.d_state)
        beta = torch.sigmoid(self.beta_logit)
        alpha = self.alpha

        if self.momentum_mode == "real":
            v = x.new_zeros(Bs, D, self.d_state)
            ys = []
            for t in range(L):
                inp = B_bar[:, t] * x[:, t, :, None]
                v = beta * v + alpha * inp
                h = A_bar[:, t] * h + v
                ys.append((h * C[:, t, None, :]).sum(-1))
            return torch.stack(ys, dim=1)

        else:  # complex
            rho = beta
            theta = self.theta
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            vr = x.new_zeros(Bs, D, self.d_state)
            vi = x.new_zeros(Bs, D, self.d_state)
            ys = []
            for t in range(L):
                inp = B_bar[:, t] * x[:, t, :, None]
                new_vr = rho * (cos_t * vr - sin_t * vi) + alpha * inp
                new_vi = rho * (cos_t * vi + sin_t * vr)
                vr, vi = new_vr, new_vi
                h = A_bar[:, t] * h + vr
                ys.append((h * C[:, t, None, :]).sum(-1))
            return torch.stack(ys, dim=1)


class MomentumMambaBlock(nn.Module):
    """
    RMSNorm -> [Linear -> Conv1D -> SiLU -> SSM] * [Linear -> SiLU] -> Linear + residual
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
    ):
        super().__init__()
        d_inner = d_model * expand

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner,
        )
        self.ssm = MomentumSSM(
            d_inner, d_state,
            use_momentum=use_momentum,
            momentum_mode=momentum_mode,
            alpha_init=alpha_init,
            beta_init=beta_init,
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        L = x_main.shape[1]
        x_main = self.conv1d(
            x_main.transpose(1, 2)
        )[:, :, :L].transpose(1, 2)
        x_main = F.silu(x_main)

        y = self.ssm(x_main)
        y = y * F.silu(z)
        return self.out_proj(y) + residual
