"""Adaptive Dual-Gate MC-Dropout engine (Phase 3).

Gate 1 (preliminary check):
    replicate cached activations across S_base = 10 parallel passes with
    active dropout on the linear head (single parallel batch);
    sigma2_prelim = Var_s( Z_hat^(s) ),  max over [B, pred_len, d_inv].

Decision branch:
    max sigma2_prelim <  tau_MC(t) -> in-distribution: early exit, mean only.
    max sigma2_prelim >= tau_MC(t) -> OOD anomaly: escalate to Gate 2.

Gate 2 escalation:
    unroll up to S_max = 130 passes in chunks of 10 with variance-convergence
    early exit: |sigma2_n - sigma2_{n-1}| < 1e-5 for `patience` consecutive
    chunks.

Design mirrors adaptive MC-dropout ideas from Conformalised Monte Carlo
prediction (Bethell et al.); results are consumed by the Stage-3 Causal
Guard when an OOD trigger fires.
"""

import torch

from .activation_cache import ActivationCache

__all__ = ["AdaptiveMCDropoutEngine"]


class AdaptiveMCDropoutEngine:
    """
    Adaptive Dual-Gate Monte Carlo Dropout Engine.
    Executes Gate 1 (10-pass fast parallel check) and Gate 2 (130-pass full sampling sweep
    with variance convergence early exit) over cached activations.
    """
    def __init__(self, model, s_base: int = 10, s_max: int = 130, var_delta_threshold: float = 1e-5, patience: int = 5):
        self.model = model
        self.cache = ActivationCache(model)
        self.s_base = s_base
        self.s_max = s_max
        self.var_delta_threshold = var_delta_threshold
        self.patience = patience

    def _execute_parallel_passes(self, h_replicated: torch.Tensor, num_passes: int, batch_size: int) -> torch.Tensor:
        """
        Executes parallel stochastic forward passes through the linear head with active dropout.
        Args:
            h_replicated: [num_passes * B, d_inv, head_in_dim]
            num_passes: Number of stochastic passes
            batch_size: Original batch size B
        Returns:
            predictions: [num_passes, B, pred_len, d_inv]
        """
        # Force linear head and dropout to training mode for stochastic sampling
        self.model.linear_head.train()
        self.model.dropout.train()

        with torch.no_grad():
            preds_flat = self.model.forward_head(h_replicated)  # [num_passes * B, pred_len, d_inv]

        pred_len = preds_flat.shape[1]
        d_inv = preds_flat.shape[2]

        # Reshape to [num_passes, B, pred_len, d_inv]
        return preds_flat.view(num_passes, batch_size, pred_len, d_inv)

    def evaluate_uncertainty(self, z_inv: torch.Tensor, tau_mc: float):
        """
        Executes the Adaptive Dual-Gate Uncertainty Evaluation.
        Args:
            z_inv: Invariant latent sequence [B, L, d_inv]
            tau_mc: Dynamic safety threshold from Hutchinson Radar
        Returns:
            Dictionary containing forecast mean, variance, pass count, and OOD flag.
        """
        b = z_inv.shape[0]

        # Step 1: Run feature extractor ONCE and cache intermediate activations
        h_cached = self.cache.get_cached_features(z_inv)

        # Step 2: Gate 1 — Run fast baseline (S_base = 10) in a single parallel batch
        h_gate1 = self.cache.replicate_for_parallel_passes(h_cached, self.s_base)
        preds_gate1 = self._execute_parallel_passes(h_gate1, self.s_base, b)

        # Compute Gate 1 preliminary empirical variance across passes: [B, pred_len, d_inv]
        var_prelim = torch.var(preds_gate1, dim=0)
        max_prelim_var = torch.max(var_prelim).item()

        # Gate 1 Decision Branch
        if max_prelim_var < tau_mc:
            # Safe / In-Distribution: early exit after S_base passes
            mean_pred = torch.mean(preds_gate1, dim=0)
            return {
                "mean": mean_pred,
                "variance": var_prelim,
                "passes_used": self.s_base,
                "is_ood": False,
                "max_var": max_prelim_var,
                "h_cached": h_cached
            }

        # Step 3: Gate 2 Escalation — Anomaly detected, unroll full sampling sweep
        # Allocate full pass tensor
        all_preds = [preds_gate1[i] for i in range(self.s_base)]
        consecutive_stable_steps = 0
        prev_var = max_prelim_var

        # Unroll remaining passes sequentially or in small parallel batches
        batch_chunk = 10
        current_passes = self.s_base

        while current_passes < self.s_max:
            passes_to_run = min(batch_chunk, self.s_max - current_passes)
            h_chunk = self.cache.replicate_for_parallel_passes(h_cached, passes_to_run)
            preds_chunk = self._execute_parallel_passes(h_chunk, passes_to_run, b)

            for i in range(passes_to_run):
                all_preds.append(preds_chunk[i])

            current_passes += passes_to_run

            # Check variance stabilization
            stacked = torch.stack(all_preds, dim=0)
            cur_var = torch.max(torch.var(stacked, dim=0)).item()
            delta_var = abs(cur_var - prev_var)

            if delta_var < self.var_delta_threshold:
                consecutive_stable_steps += 1
                if consecutive_stable_steps >= self.patience:
                    break  # Variance has stabilized, exit early
            else:
                consecutive_stable_steps = 0

            prev_var = cur_var

        final_stacked = torch.stack(all_preds, dim=0)
        final_mean = torch.mean(final_stacked, dim=0)
        final_var = torch.var(final_stacked, dim=0)

        return {
            "mean": final_mean,
            "variance": final_var,
            "passes_used": len(all_preds),
            "is_ood": True,
            "max_var": torch.max(final_var).item(),
            "h_cached": h_cached
        }

