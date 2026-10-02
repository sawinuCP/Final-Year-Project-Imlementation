"""Active Causal Guardrail Coordinator."""

import time
import torch
import torch.nn as nn
from .nullspace_patcher import NullSpacePatcher

__all__ = ["ActiveCausalGuard"]


class ActiveCausalGuard(nn.Module):
    def __init__(self, viae, forecaster, hypernet, mc_engine, radar, beta_normal: float = 1.0, beta_tightened: float = 10.0):
        super(ActiveCausalGuard, self).__init__()
        self.viae = viae
        self.forecaster = forecaster
        self.hypernet = hypernet
        self.mc_engine = mc_engine
        self.radar = radar
        self.patcher = NullSpacePatcher()
        self.beta_normal = beta_normal
        self.beta_tightened = beta_tightened
        self.d_inv = viae.d_inv
        self.d_e = viae.d_e

    def forward_stream(self, x_raw: torch.Tensor) -> dict:
        start_total = time.perf_counter()
        device = x_raw.device

        # 1. Deterministic Invariant Encoding
        beta_tensor = torch.tensor([[self.beta_normal]], device=device, dtype=torch.float32)
        modulations = self.hypernet(beta_tensor)

        with torch.no_grad():
            z_inv, _, _, z_e, _, _ = self.viae.encode(x_raw, modulations, deterministic=True)

        # 2. Hutchinson Topological Radar
        z_curr = z_inv[:, -1, :]
        transition_fn = lambda z: self.forecaster.forward_head(
            torch.cat([z, torch.zeros(z.shape[0], self.forecaster.head_in_dim - z.shape[1], device=device)], dim=-1)
            .unsqueeze(1).repeat(1, self.d_inv, 1)
        )[:, 0, :]

        with torch.enable_grad():
            divergence = self.radar.estimate_divergence(transition_fn, z_curr)
            tau_mc_t = self.radar.compute_adaptive_threshold(divergence)

        # 3. Adaptive MC-Dropout Evaluation
        unc_result = self.mc_engine.evaluate_uncertainty(z_inv, tau_mc=tau_mc_t)
        is_ood = unc_result["is_ood"]
        passes_used = unc_result["passes_used"]
        z_pred_latent = unc_result["mean"]

        repair_applied = False
        repair_latency_ms = 0.0

        if not is_ood:
            # Clean invariant decoding
            with torch.no_grad():
                y_forecast = self.viae.decode(z_pred_latent, None)
        else:
            # Active Causal Repair
            start_repair = time.perf_counter()

            # Slide beta up via Hypernetwork
            beta_tight = torch.tensor([[self.beta_tightened]], device=device, dtype=torch.float32)
            mod_tight = self.hypernet(beta_tight)
            with torch.no_grad():
                _, _, _, z_e_tight, _, _ = self.viae.encode(x_raw, mod_tight, deterministic=True)

            # Construct P_inv and project latent trajectory
            p_inv = self.patcher.compute_latent_projection(z_e_tight, d_inv=self.d_inv)
            z_repaired = self.patcher.project_latent_trajectory(z_pred_latent, p_inv)

            with torch.no_grad():
                y_forecast = self.viae.decode(z_repaired, None)

            repair_latency_ms = (time.perf_counter() - start_repair) * 1000.0
            repair_applied = True

        total_latency_ms = (time.perf_counter() - start_total) * 1000.0

        return {
            "forecast": y_forecast,
            "is_ood": is_ood,
            "passes_used": passes_used,
            "divergence": divergence.item(),
            "tau_mc_t": tau_mc_t,
            "mean_variance": unc_result["mean_var"],
            "repair_applied": repair_applied,
            "repair_latency_ms": repair_latency_ms,
            "total_latency_ms": total_latency_ms
        }