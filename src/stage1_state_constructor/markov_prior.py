"""First-order Markovian transition prior on Z_inv (Phase 2).

    p(Z_inv,t | Z_inv,t-1) = N( f_trans(Z_inv,t-1), sigma_prior^2 * I )

``f_trans``: lightweight temporal transition network R^{d_inv} -> R^{d_inv}
emitting the prior mean and log-variance for the next latent state.

Purpose: eliminate *Latent Chaos* (temporally disorganized internal
representations) by matching the encoder posterior at every timestep t to
this smooth, Markovian prior through a closed-form Gaussian KL term.
"""

import torch
import torch.nn as nn

__all__ = ["MarkovianTransitionPrior"]


class MarkovianTransitionPrior(nn.Module):
    """
    Parameterizes a first-order Markovian temporal prior p(Z_inv,t | Z_inv,t-1).
    Enforces that chronologically adjacent latent states follow a smooth, continuous
    physical transition, eliminating Latent Chaos.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super(MarkovianTransitionPrior, self).__init__()
        self.latent_dim = latent_dim
        self.transition_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # outputs mean and log-variance
        )

    def forward(self, z_prev: torch.Tensor):
        """
        Args:
            z_prev: Latent states at timestep t-1 of shape [B, latent_dim]
        Returns:
            mu_prior, log_var_prior of shape [B, latent_dim]
        """
        stats = self.transition_net(z_prev)
        mu_prior, log_var_prior = torch.chunk(stats, 2, dim=-1)
        # Constrain log-variance to prevent numerical explosion
        log_var_prior = torch.clamp(log_var_prior, min=-10.0, max=2.0)
        return mu_prior, log_var_prior

    def compute_markov_kl(self, z_seq: torch.Tensor, mu_enc: torch.Tensor, log_var_enc: torch.Tensor) -> torch.Tensor:
        """
        Computes temporal transition KL divergence along the sequence length L.
        Args:
            z_seq: Sampled latent sequence [B, L, d_inv]
            mu_enc: Encoder posterior means [B, L, d_inv]
            log_var_enc: Encoder posterior log-variances [B, L, d_inv]
        Returns:
            Scalar transition KL divergence loss
        """
        seq_len = z_seq.shape[1]
        if seq_len <= 1:
            return torch.tensor(0.0, device=z_seq.device, dtype=torch.float32)

        # z at t-1: shape [B, L-1, d_inv]
        z_prev = z_seq[:, :-1, :].reshape(-1, self.latent_dim)
        # Encoder posteriors at t (from index 1 to L): shape [B*(L-1), d_inv]
        mu_t = mu_enc[:, 1:, :].reshape(-1, self.latent_dim)
        log_var_t = log_var_enc[:, 1:, :].reshape(-1, self.latent_dim)

        # Prior distribution conditioned on z_prev
        mu_prior, log_var_prior = self.forward(z_prev)

        var_prior = torch.exp(log_var_prior)
        var_t = torch.exp(log_var_t)

        # Closed-form KL between two Gaussians: N(mu_t, var_t) || N(mu_prior, var_prior)
        kl = 0.5 * (log_var_prior - log_var_t + (var_t + (mu_t - mu_prior).pow(2)) / var_prior - 1.0)
        return torch.mean(torch.sum(kl, dim=-1))

