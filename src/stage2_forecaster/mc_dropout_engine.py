"""Adaptive MC-Dropout Engine with Mean Variance Calibration."""

import torch
from .activation_cache import ActivationCache

__all__ = ["AdaptiveMCDropoutEngine"]


class AdaptiveMCDropoutEngine:
    def __init__(self, model, s_base: int = 10, s_max: int = 30, var_delta_threshold: float = 1e-6, patience: int = 3):
        self.model = model
        self.cache = ActivationCache(model)
        self.s_base = s_base
        self.s_max = s_max
        self.var_delta_threshold = var_delta_threshold
        self.patience = patience

    def _execute_parallel_passes(self, h_replicated: torch.Tensor, num_passes: int, batch_size: int) -> torch.Tensor:
        self.model.linear_head.train()
        self.model.dropout.train()
        with torch.no_grad():
            preds_flat = self.model.forward_head(h_replicated)

        pred_len, d_inv = preds_flat.shape[1], preds_flat.shape[2]
        return preds_flat.view(num_passes, batch_size, pred_len, d_inv)

    def evaluate_uncertainty(self, z_inv: torch.Tensor, tau_mc: float):
        b = z_inv.shape[0]
        h_cached = self.cache.get_cached_features(z_inv)

        # Gate 1: Fast baseline (S_base = 10 passes)
        h_gate1 = self.cache.replicate_for_parallel_passes(h_cached, self.s_base)
        preds_gate1 = self._execute_parallel_passes(h_gate1, self.s_base, b)

        # Stable sequence-mean epistemic variance across passes
        var_prelim = torch.var(preds_gate1, dim=0) # [B, pred_len, d_inv]
        mean_prelim_var = torch.mean(var_prelim).item()

        if mean_prelim_var < tau_mc:
            # Normal in-distribution exit
            return {
                "mean": torch.mean(preds_gate1, dim=0),
                "variance": var_prelim,
                "passes_used": self.s_base,
                "is_ood": False,
                "mean_var": mean_prelim_var,
                "h_cached": h_cached
            }

        # Gate 2 Escalation: Full sampling sweep
        all_preds = [preds_gate1[i] for i in range(self.s_base)]
        consecutive_stable = 0
        prev_var = mean_prelim_var

        current_passes = self.s_base
        while current_passes < self.s_max:
            passes_to_run = min(10, self.s_max - current_passes)
            h_chunk = self.cache.replicate_for_parallel_passes(h_cached, passes_to_run)
            preds_chunk = self._execute_parallel_passes(h_chunk, passes_to_run, b)
            for i in range(passes_to_run):
                all_preds.append(preds_chunk[i])
            current_passes += passes_to_run

            stacked = torch.stack(all_preds, dim=0)
            cur_var = torch.mean(torch.var(stacked, dim=0)).item()
            if abs(cur_var - prev_var) < self.var_delta_threshold:
                consecutive_stable += 1
                if consecutive_stable >= self.patience:
                    break
            else:
                consecutive_stable = 0
            prev_var = cur_var

        final_stacked = torch.stack(all_preds, dim=0)
        return {
            "mean": torch.mean(final_stacked, dim=0),
            "variance": torch.var(final_stacked, dim=0),
            "passes_used": len(all_preds),
            "is_ood": True,
            "mean_var": torch.mean(torch.var(final_stacked, dim=0)).item(),
            "h_cached": h_cached
        }