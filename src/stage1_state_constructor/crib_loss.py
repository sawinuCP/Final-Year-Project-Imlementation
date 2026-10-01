"""Consistency-Regularized Information Bottleneck (CRIB) Loss.
Enforces non-collapsing invariant reconstruction:
    Recon_Loss = MSE(X, X_recon_inv) + 0.5 * MSE(X, X_recon_full)
"""

import torch
import torch.nn as nn
from typing import Optional
from .markov_prior import MarkovianTransitionPrior

__all__ = ["CRIBLoss"]


class CRIBLoss(nn.Module):
    def __init__(self, d_inv: int, gamma_consistency: float = 0.5, lambda_res: float = 0.2, eps: float = 1e-6):
        super(CRIBLoss, self).__init__()
        self.d_inv = d_inv
        self.gamma_consistency = gamma_consistency
        self.lambda_res = lambda_res
        self.eps = eps
        self.markov_prior = MarkovianTransitionPrior(latent_dim=d_inv)

    def _compute_surrogate_residual_loss(self, residuals: torch.Tensor, domain_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = residuals.shape[0]
        if b < 4:
            return torch.tensor(0.0, device=residuals.device, dtype=torch.float32)

        res_flat = residuals.contiguous().view(b, -1)
        if domain_labels is not None and torch.unique(domain_labels).numel() >= 2:
            unique_domains = torch.unique(domain_labels)
            domain_vars = []
            for d_id in unique_domains[:2]:
                mask = (domain_labels == d_id)
                if mask.sum() >= 2:
                    domain_vars.append(torch.var(res_flat[mask], dim=0, unbiased=True))
            if len(domain_vars) == 2:
                return torch.norm(domain_vars[0] - domain_vars[1], p=2)

        # Fallback split
        mid = b // 2
        return torch.norm(torch.var(res_flat[:mid], dim=0) - torch.var(res_flat[mid:], dim=0), p=2)

    def forward(self, output_clean: dict, output_perturbed: dict, x_raw: torch.Tensor,
                beta: float, domain_labels: Optional[torch.Tensor] = None) -> dict:
        # Enforce that Z_inv alone must reconstruct X
        recon_inv = nn.functional.mse_loss(output_clean["x_recon_inv"], x_raw)
        recon_full = nn.functional.mse_loss(output_clean["x_recon"], x_raw)
        recon_loss = recon_inv + 0.5 * recon_full

        # Markovian prior KL on Z_inv
        markov_kl = self.markov_prior.compute_markov_kl(
            z_seq=output_clean["z_inv"],
            mu_enc=output_clean["mu_inv"],
            log_var_enc=output_clean["logvar_inv"]
        )

        mu_inv_0 = output_clean["mu_inv"][:, 0, :]
        logvar_inv_0 = output_clean["logvar_inv"][:, 0, :]
        kl_inv_0 = -0.5 * torch.mean(torch.sum(1 + logvar_inv_0 - mu_inv_0.pow(2) - logvar_inv_0.exp(), dim=-1))
        total_kl_inv = markov_kl + kl_inv_0

        # Shortcut regularization
        mu_e = output_clean["mu_e"]
        logvar_e = output_clean["logvar_e"]
        kl_e = -0.5 * torch.mean(torch.sum(1 + logvar_e - mu_e.pow(2) - logvar_e.exp(), dim=-1))

        # Consistency loss
        mu_c = output_clean["mu_inv"]
        var_c = torch.exp(output_clean["logvar_inv"]) + self.eps
        mu_p = output_perturbed["mu_inv"]
        var_p = torch.exp(output_perturbed["logvar_inv"]) + self.eps
        logvar_c = output_clean["logvar_inv"]
        logvar_p = output_perturbed["logvar_inv"]

        consistency_kl = 0.5 * (logvar_p - logvar_c + (var_c + (mu_c - mu_p).pow(2)) / var_p - 1.0)
        consistency_loss = torch.mean(torch.sum(consistency_kl, dim=-1))

        residual_var_loss = self._compute_surrogate_residual_loss(x_raw - output_clean["x_recon"], domain_labels)

        # Scale beta to balanced range
        beta_scaled = min(0.05, max(0.001, beta * 0.01))

        total_loss = (
            recon_loss
            + (beta_scaled * total_kl_inv)
            + (0.01 * kl_e)
            + (self.gamma_consistency * consistency_loss)
            + (self.lambda_res * residual_var_loss)
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "recon_inv": recon_inv,
            "markov_kl": markov_kl,
            "kl_e": kl_e,
            "consistency_loss": consistency_loss,
            "residual_var_loss": residual_var_loss
        }