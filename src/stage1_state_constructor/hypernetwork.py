"""Secondary Hypernetwork controller h_psi (Phase 2).

Learns the ENTIRE continuous Rate-Distortion curve (beta trade-off of the
information bottleneck) in a single training run, following the Hyper-VIB
idea that a hypernetwork generates approximately optimal parameters for
every hyperparameter value without grid/random search.

As-built design (Phase 2): h_psi(beta) is a 3-layer MLP emitting FiLM-style
modulation parameters (per-channel scale gamma and shift beta_bias) for the
target VIAE feature layers -- a stable, low-variance realization of
beta-conditioning. The output layer is zero-initialised so modulation is
the IDENTITY at init (scale = 1, shift = 0).

Beta sampling during training:  log beta ~ U(log 0.01, log 10.0).
Stage-3 escalation:             beta -> 10.0 (tighten the bottleneck).
"""

import torch
import torch.nn as nn

__all__ = ["HypernetworkController"]


class HypernetworkController(nn.Module):
    """
    Secondary Hypernetwork h_psi(beta).
    Takes a continuous trade-off scalar beta as input (in log-space) and generates
    modulation parameters (scaling gamma and shifting bias) for each layer of the VIAE.
    Learns the entire continuous Rate-Distortion curve in a single training run.
    """
    def __init__(self, modulation_dims: list, hidden_dim: int = 128):
        """
        Args:
            modulation_dims: List of integer target layer dimensionalities to modulate.
            hidden_dim: Hidden dimension of the hypernetwork MLP.
        """
        super(HypernetworkController, self).__init__()
        self.total_modulation_dim = sum(modulation_dims) * 2  # scale and shift for each target
        self.modulation_dims = modulation_dims

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.total_modulation_dim)
        )

        # Zero-centered initialization for shifts, unity for scales
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, beta: torch.Tensor) -> dict:
        """
        Args:
            beta: Scalar or 1D Tensor of shape [B, 1] or [1, 1].
        Returns:
            Dictionary containing 'scales' and 'shifts' tensors for each layer.
        """
        if beta.dim() == 0:
            beta = beta.view(1, 1)
        elif beta.dim() == 1:
            beta = beta.view(-1, 1)

        # Clamp beta to a numerically safe range before the log transform
        log_beta = torch.log(torch.clamp(beta, min=1e-4, max=100.0))
        raw_outputs = self.mlp(log_beta)  # [B, total_modulation_dim]

        scales = []
        shifts = []
        start_idx = 0

        for dim in self.modulation_dims:
            dim_params = raw_outputs[:, start_idx : start_idx + (2 * dim)]
            scale, shift = torch.chunk(dim_params, 2, dim=-1)
            # Center scale around 1.0 using tanh for stability
            scale = 1.0 + torch.tanh(scale)
            scales.append(scale)
            shifts.append(shift)
            start_idx += 2 * dim

        return {"scales": scales, "shifts": shifts}

