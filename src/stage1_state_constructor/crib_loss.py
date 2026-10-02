"""CRIB Loss with Invariant Target Decomposition."""

import torch
import torch.nn as nn
from typing import Optional
from .markov_prior import MarkovianTransitionPrior

__all__ = ["CRIBLoss"]


class CRIBLoss(nn.Module):
    def __init__(self, d_inv: int = 16, gamma_consistency: float = 0.5, lambda_res: float = 0.2, eps: float = 1e-6):
        super(CRIBLoss, self).__init__()
        self.d_inv = d_inv
        self.gamma_consistency = gamma_consistency
        self.lambda_res = lambda_res
        self.eps = eps
        self.markov_prior = MarkovianTransitionPrior(latent_dim=d_inv)

    def forward(self, output_clean: dict, output_perturbed: dict, x_raw: torch.Tensor,
                x_inv_target: torch.Tensor, beta: float, domain_labels: Optional[torch.Tensor] = None) -> dict:
        # Z_inv reconstructs the clean invariant target (x_inv_target), NOT the poisoned raw input
        recon_inv = nn.functional.mse_loss(output_clean["x_recon_inv"], x_inv_target)
        # Full reconstruction fits the composite observation
        recon_full = nn.functional.mse_loss(output_clean["x_recon"], x_raw)
        recon_loss = recon_inv + 0.5 * recon_full

        # Markovian Prior KL on Z_inv
        markov_kl = self.markov_prior.compute_markov_kl(
            z_seq=output_clean["z_inv"],
            mu_enc=output_clean["mu_inv"],
            log_var_enc=output_clean["logvar_inv"]
        )

        mu_inv_0 = output_clean["mu_inv"][:, 0, :]
        logvar_inv_0 = output_clean["logvar_inv"][:, 0, :]
        kl_inv_0 = -0.5 * torch.mean(torch.sum(1 + logvar_inv_0 - mu_inv_0.pow(2) - logvar_inv_0.exp(), dim=-1))
        total_kl_inv = markov_kl + kl_inv_0

        # Shortcut sponge prior KL
        mu_e = output_clean["mu_e"]
        logvar_e = output_clean["logvar_e"]
        kl_e = -0.5 * torch.mean(torch.sum(1 + logvar_e - mu_e.pow(2) - logvar_e.exp(), dim=-1))

        # Temporal Consistency KL
        mu_c = output_clean["mu_inv"]
        var_c = torch.exp(output_clean["logvar_inv"]) + self.eps
        mu_p = output_perturbed["mu_inv"]
        var_p = torch.exp(output_perturbed["logvar_inv"]) + self.eps
        logvar_c = output_clean["logvar_inv"]
        logvar_p = output_perturbed["logvar_inv"]

        consistency_kl = 0.5 * (logvar_p - logvar_c + (var_c + (mu_c - mu_p).pow(2)) / var_p - 1.0)
        consistency_loss = torch.mean(torch.sum(consistency_kl, dim=-1))

        beta_scaled = min(0.01, max(0.0005, beta * 0.002))

        total_loss = (
            recon_loss
            + (beta_scaled * total_kl_inv)
            + (0.005 * kl_e)
            + (self.gamma_consistency * consistency_loss)
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "recon_inv": recon_inv,
            "markov_kl": markov_kl,
            "consistency_loss": consistency_loss
        }