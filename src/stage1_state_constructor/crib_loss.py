import torch
import torch.nn as nn
from typing import Optional
from .markov_prior import MarkovianTransitionPrior

class CRIBLoss(nn.Module):
    """
    Consistency-Regularized Information Bottleneck (CRIB) Loss with
    FOIL-Style Instance Residual Normalization (IRN).
    
    Addresses Critical Flaw 5:
      Replaces naive batch halving with domain-stratified residual alignment.
      Penalizes sensitivity to unobserved confounders by aligning residual
      distributions across distinct temporal regimes.
    """
    def __init__(self, d_inv: int, gamma_consistency: float = 1.0, lambda_res: float = 0.5, eps: float = 1e-6):
        super(CRIBLoss, self).__init__()
        self.d_inv = d_inv
        self.gamma_consistency = gamma_consistency
        self.lambda_res = lambda_res
        self.eps = eps
        self.markov_prior = MarkovianTransitionPrior(latent_dim=d_inv)

    def _compute_surrogate_residual_loss(self, residuals: torch.Tensor, domain_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the cross-environment residual discrepancy to filter unobserved confounders.
        Args:
            residuals: Reconstruction error tensor [B, L, D].
            domain_labels: Optional tensor [B] indicating temporal regime / environment ID.
        Returns:
            Scalar penalty measuring residual distribution shift across environments.
        """
        b, l, d = residuals.shape
        if b < 4:
            return torch.tensor(0.0, device=residuals.device, dtype=torch.float32)

        # Flatten sequence length into sample instances: [B, L * D]
        res_flat = residuals.contiguous().view(b, -1)

        if domain_labels is not None and torch.unique(domain_labels).numel() >= 2:
            # Case A: Explicit temporal domain/chunk labels provided
            unique_domains = torch.unique(domain_labels)
            domain_vars = []
            domain_means = []

            for d_id in unique_domains[:2]:  # Align primary two regimes
                mask = (domain_labels == d_id)
                if mask.sum() >= 2:
                    res_d = res_flat[mask]
                    domain_vars.append(torch.var(res_d, dim=0, unbiased=True))
                    domain_means.append(torch.mean(res_d, dim=0))

            if len(domain_vars) == 2:
                # Penalize both residual variance gap and mean residual shift across regimes
                var_discrepancy = torch.norm(domain_vars[0] - domain_vars[1], p=2)
                mean_discrepancy = torch.norm(domain_means[0] - domain_means[1], p=2)
                return var_discrepancy + 0.1 * mean_discrepancy

        # Case B: Unsupervised Fallback — Spectral / Temporal Regime Splitting
        # Identify the primary axis of residual variance via first principal component
        # This splits the batch into the two most dissimilar residual clusters
        res_centered = res_flat - res_flat.mean(dim=0, keepdim=True)
        # Fast 1-step power iteration for top singular vector of residual covariance
        v = torch.randn(res_flat.shape[1], 1, device=residuals.device, dtype=torch.float32)
        v = v / (torch.norm(v) + self.eps)
        for _ in range(3):
            v = torch.matmul(res_centered.t(), torch.matmul(res_centered, v))
            v = v / (torch.norm(v) + self.eps)

        projections = torch.matmul(res_centered, v).squeeze(-1)  # [B]
        median_val = torch.median(projections)

        group1 = res_flat[projections <= median_val]
        group2 = res_flat[projections > median_val]

        if group1.shape[0] >= 2 and group2.shape[0] >= 2:
            var1 = torch.var(group1, dim=0, unbiased=True)
            var2 = torch.var(group2, dim=0, unbiased=True)
            return torch.norm(var1 - var2, p=2)

        return torch.tensor(0.0, device=residuals.device, dtype=torch.float32)

    def forward(self, output_clean: dict, output_perturbed: dict, x_raw: torch.Tensor,
                beta: float, domain_labels: Optional[torch.Tensor] = None) -> dict:
        """
        Computes the complete Multi-Rate ELBO with consistency and unobserved confounder alignment.
        Args:
            output_clean: Output dictionary from VIAE forward pass on clean input X.
            output_perturbed: Output dictionary from VIAE forward pass on perturbed input X_tilde.
            x_raw: Clean input sequence [B, L, D] in FP32.
            beta: Active Information Bottleneck compression multiplier.
            domain_labels: Optional temporal domain indicator [B].
        """
        # 1. Reconstruction Loss (Observation-space fidelity)
        recon_loss = nn.functional.mse_loss(output_clean["x_recon"], x_raw)

        # 2. Invariant Markovian Transition Prior KL (Eliminates Latent Chaos on Z_inv)
        markov_kl = self.markov_prior.compute_markov_kl(
            z_seq=output_clean["z_inv"],
            mu_enc=output_clean["mu_inv"],
            log_var_enc=output_clean["logvar_inv"]
        )

        # Standard Gaussian prior matching for initial timestep t=0
        mu_inv_0 = output_clean["mu_inv"][:, 0, :]
        logvar_inv_0 = output_clean["logvar_inv"][:, 0, :]
        kl_inv_0 = -0.5 * torch.mean(torch.sum(1 + logvar_inv_0 - mu_inv_0.pow(2) - logvar_inv_0.exp(), dim=-1))
        total_kl_inv = markov_kl + kl_inv_0

        # 3. Environmental Shortcut Compression (Z_e mapped to unit Gaussian sponge)
        mu_e = output_clean["mu_e"]
        logvar_e = output_clean["logvar_e"]
        kl_e = -0.5 * torch.mean(torch.sum(1 + logvar_e - mu_e.pow(2) - logvar_e.exp(), dim=-1))

        # 4. Dual-View Temporal Consistency Loss: D_KL( q(Z_inv|X) || q(Z_inv|X_tilde) )
        mu_c = output_clean["mu_inv"]
        var_c = torch.exp(output_clean["logvar_inv"]) + self.eps
        mu_p = output_perturbed["mu_inv"]
        var_p = torch.exp(output_perturbed["logvar_inv"]) + self.eps
        logvar_c = output_clean["logvar_inv"]
        logvar_p = output_perturbed["logvar_inv"]

        consistency_kl = 0.5 * (logvar_p - logvar_c + (var_c + (mu_c - mu_p).pow(2)) / var_p - 1.0)
        consistency_loss = torch.mean(torch.sum(consistency_kl, dim=-1))

        # 5. Upgraded FOIL-Style Surrogate Residual Alignment (Addresses Critical Flaw 5)
        residuals = x_raw - output_clean["x_recon"]
        residual_var_loss = self._compute_surrogate_residual_loss(residuals, domain_labels=domain_labels)

        # Unified Multi-Rate ELBO
        total_loss = (
            recon_loss
            + (beta * total_kl_inv)
            + (0.1 * kl_e)
            + (self.gamma_consistency * consistency_loss)
            + (self.lambda_res * residual_var_loss)
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "markov_kl": markov_kl,
            "kl_e": kl_e,
            "consistency_loss": consistency_loss,
            "residual_var_loss": residual_var_loss
        }