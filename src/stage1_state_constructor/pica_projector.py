"""Principal Invariant Component Analysis (PICA) null-space projector (Phase 2).

Unsupervised invariant-subspace initialization (IRM perspective):
    1. Partition unlabelled training data into temporal segments E_1, E_2.
    2. Compute covariance matrices  Sigma_1, Sigma_2 in R^{D x D}.
    3. Difference matrix  DeltaSigma = Sigma_1 - Sigma_2;
       SVD -> kernel null-space  U_null = ker(DeltaSigma), i.e. the directions
       carrying the SMALLEST singular values are the invariant subspace.
    4. Invariant linear projection:  X_pica = X @ U_null.

Initializes the invariant input features consumed by the ActiveVIAE encoder.
All covariance/SVD math runs in FP32 to avoid singular-matrix collapse.
"""

import torch
import torch.nn as nn

__all__ = ["PICAProjector"]


class PICAProjector(nn.Module):
    """
    Principal Invariant Component Analysis (PICA) Projector.
    Computes covariance matrices over distinct temporal chunks of unlabelled data,
    finds the difference null-space kernel, and projects input features onto
    a linear subspace where covariance structure remains invariant.
    """
    def __init__(self, in_features: int, invariant_dim: int):
        super(PICAProjector, self).__init__()
        self.in_features = in_features
        self.invariant_dim = invariant_dim
        # Projection matrix U_null: shape [in_features, invariant_dim]
        self.register_buffer("u_null", torch.eye(in_features, invariant_dim, dtype=torch.float32))

    @torch.no_grad()
    def fit(self, x_env1: torch.Tensor, x_env2: torch.Tensor):
        """
        Fits the null-space projection matrix using two unlabelled temporal splits.
        Args:
            x_env1: Tensor of shape [N1, D] from temporal chunk 1.
            x_env2: Tensor of shape [N2, D] from temporal chunk 2.
        """
        assert x_env1.shape[1] == self.in_features, "Feature dimension mismatch."
        assert x_env2.shape[1] == self.in_features, "Feature dimension mismatch."

        # Center the data
        x1_centered = x_env1 - x_env1.mean(dim=0, keepdim=True)
        x2_centered = x_env2 - x_env2.mean(dim=0, keepdim=True)

        n1, n2 = x1_centered.shape[0], x2_centered.shape[0]
        sigma1 = torch.matmul(x1_centered.t(), x1_centered) / (n1 - 1)
        sigma2 = torch.matmul(x2_centered.t(), x2_centered) / (n2 - 1)

        delta_sigma = sigma1 - sigma2

        # Use eigh for symmetric matrices: eigenvalues are real and eigenvectors are orthogonal
        eigenvalues, eigenvectors = torch.linalg.eigh(delta_sigma)

        # Sort by absolute eigenvalue: directions with |lambda| closest to 0 are invariant
        sort_indices = torch.argsort(torch.abs(eigenvalues), descending=False)
        u_sorted = eigenvectors[:, sort_indices]

        self.u_null.copy_(u_sorted[:, :self.invariant_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects input tensor x of shape [B, L, D] onto the invariant subspace.
        Returns:
            x_proj of shape [B, L, invariant_dim]
        """
        return torch.matmul(x, self.u_null)

