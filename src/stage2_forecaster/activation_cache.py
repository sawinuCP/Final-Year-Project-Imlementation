"""Last-Layer Activation Caching (Phase 3).

Decouples the PatchTST backbone into:
    F_backbone : feature extractor (patch embedding + attention stack)
    W          : final linear decision head in R^{pred_len x head_in_dim}

During inference the backbone forward pass runs EXACTLY ONCE (eval mode,
no dropout); the cached activation h_cached in R^{B x d_inv x head_in_dim}
is then replicated along a parallel-pass dimension so every MC-Dropout
sample re-runs only the cheap linear head -- eliminating the runtime
latency bottleneck of sequential deep re-computation.
"""

import torch

__all__ = ["ActivationCache"]


class ActivationCache:
    """
    Manages pre-decision-layer activation caching to eliminate
    sequential deep layer re-computation during Monte Carlo Dropout passes.
    """
    def __init__(self, model):
        self.model = model

    def get_cached_features(self, z_inv: torch.Tensor) -> torch.Tensor:
        """
        Runs the heavy Transformer backbone once and stores the intermediate activations.
        Args:
            z_inv: [B, L, d_inv] in FP32
        Returns:
            h_cached: [B, d_inv, head_in_dim]
        """
        self.model.eval()
        with torch.no_grad():
            h_cached = self.model.extract_features(z_inv)
        return h_cached

    @staticmethod
    def replicate_for_parallel_passes(h_cached: torch.Tensor, num_passes: int) -> torch.Tensor:
        """
        Replicates cached features along a parallel pass dimension for zero-latency GPU batching.
        Args:
            h_cached: [B, d_inv, head_in_dim]
            num_passes: Number of parallel stochastic dropout passes (e.g., 10 or 130)
        Returns:
            replicated: [num_passes * B, d_inv, head_in_dim]
        """
        b, d, feat_dim = h_cached.shape
        # Repeat along leading dimension: [num_passes, B, d, feat_dim] -> [num_passes * B, d, feat_dim]
        return h_cached.unsqueeze(0).repeat(num_passes, 1, 1, 1).view(num_passes * b, d, feat_dim)

