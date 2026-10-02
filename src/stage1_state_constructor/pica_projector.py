"""PICA Projector using Hermitian Eigendecomposition (torch.linalg.eigh)."""

import torch
import torch.nn as nn

__all__ = ["PICAProjector"]


class PICAProjector(nn.Module):
    def __init__(self, in_features: int = 7, invariant_dim: int = 6):
        super(PICAProjector, self).__init__()
        self.in_features = in_features
        self.invariant_dim = invariant_dim

        # U_null spans the invariant subspace (6 dims)
        # U_spur spans the shortcut subspace (1 dim)
        self.register_buffer("u_null", torch.eye(in_features, invariant_dim, dtype=torch.float32))
        self.register_buffer("u_spur", torch.zeros(in_features, in_features - invariant_dim, dtype=torch.float32))

    @torch.no_grad()
    def fit(self, x_env1: torch.Tensor, x_env2: torch.Tensor):
        x1_c = x_env1 - x_env1.mean(dim=0, keepdim=True)
        x2_c = x_env2 - x_env2.mean(dim=0, keepdim=True)

        n1, n2 = x1_c.shape[0], x2_c.shape[0]
        sigma1 = torch.matmul(x1_c.t(), x1_c) / (n1 - 1)
        sigma2 = torch.matmul(x2_c.t(), x2_c) / (n2 - 1)

        delta_sigma = sigma1 - sigma2

        # Hermitian Eigendecomposition on symmetric delta_sigma
        eigenvalues, eigenvectors = torch.linalg.eigh(delta_sigma)

        # Sort by absolute eigenvalue: smallest |lambda| are invariant, largest are spurious
        sort_indices = torch.argsort(torch.abs(eigenvalues), descending=False)
        u_sorted = eigenvectors[:, sort_indices]

        self.u_null.copy_(u_sorted[:, :self.invariant_dim])
        self.u_spur.copy_(u_sorted[:, self.invariant_dim:])

    def project_invariant(self, x: torch.Tensor) -> torch.Tensor:
        """Projects X onto invariant subspace: X_inv = X @ U_null @ U_null^T."""
        p_inv = torch.matmul(self.u_null, self.u_null.t())
        return torch.matmul(x, p_inv)

    def project_spurious(self, x: torch.Tensor) -> torch.Tensor:
        """Projects X onto spurious subspace: X_spur = X @ U_spur @ U_spur^T."""
        p_spur = torch.matmul(self.u_spur, self.u_spur.t())
        return torch.matmul(x, p_spur)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.u_null)