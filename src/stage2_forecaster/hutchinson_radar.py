"""Hutchinson Trace Topological Radar (Phase 3).

Stochastic trace estimation of the latent velocity-field Jacobian with
Rademacher probe vectors v ~ {-1,+1}^{d_inv}:

    D_Hutchinson = E_v[ v^T J_f(Z) v ]  ~=  Tr(J_f),     E_v[v^T A v] = Tr(A)

The estimated local vector-field divergence scales the MC safety threshold
dynamically and context-aware:

    tau_MC(t) = tau_base / (1 + gamma * max(0, D_Hutchinson(t)))

tau_base is calibrated on clean validation data (95th percentile of
epistemic variance). High divergence tightens the threshold, triggering
Gate-2 escalation earlier in topologically unstable latent regions.
"""

import torch
import torch.nn as nn

__all__ = ["HutchinsonTopologicalRadar"]


class HutchinsonTopologicalRadar(nn.Module):
    """
    Hutchinson Trace Estimator.
    Monitors the structural divergence of the latent vector field:
        D_Hutchinson = E_v [ v^T J_f v ]
    where v is a random Rademacher vector, estimating the trace of the Jacobian.
    Dynamically scales the safety threshold tau_MC(t).
    """
    def __init__(self, tau_base: float = 0.05, gamma: float = 2.0, num_vectors: int = 4):
        super(HutchinsonTopologicalRadar, self).__init__()
        self.tau_base = tau_base
        self.gamma = gamma
        self.num_vectors = num_vectors

    def estimate_divergence(self, transition_fn, z: torch.Tensor) -> torch.Tensor:
        """
        Estimates the scalar divergence of transition function f at coordinate z.
        Args:
            transition_fn: Callable mapping z -> z_next (e.g. Stage 1 Markov prior or Stage 2 backbone)
            z: Latent tensor of shape [B, d_inv]
        Returns:
            Scalar divergence metric averaged across batch
        """
        b, _ = z.shape
        # Replicate z across num_vectors to batch probe calculations
        z_rep = z.repeat(self.num_vectors, 1).detach().requires_grad_(True)
        
        # Sample Rademacher random vectors in parallel: v in {-1, +1}
        v = torch.randint(0, 2, z_rep.shape, device=z.device, dtype=torch.float32) * 2.0 - 1.0

        out = transition_fn(z_rep)
        if isinstance(out, tuple):
            out = out[0]

        # Single batched vector-Jacobian product
        vjp = torch.autograd.grad(
            outputs=out,
            inputs=z_rep,
            grad_outputs=v,
            create_graph=False,
            retain_graph=False
        )[0]

        # Inner product v^T * J * v
        div_samples = torch.sum(v * vjp, dim=-1).view(self.num_vectors, b)
        mean_div = torch.mean(div_samples, dim=0)
        return torch.mean(torch.abs(mean_div))

    def compute_adaptive_threshold(self, divergence: torch.Tensor) -> float:
        """
        Dynamically lowers safety threshold when structural divergence spikes:
            tau_MC(t) = tau_base / (1 + gamma * max(0, divergence))
        """
        div_val = max(0.0, divergence.item())
        return self.tau_base / (1.0 + self.gamma * div_val)

