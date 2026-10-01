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
        b, l, d_obs = x_raw.shape

        # Step 1: Deterministic Invariant Encoding
        beta_tensor = torch.tensor([[self.beta_normal]], device=device, dtype=torch.float32)
        modulations = self.hypernet(beta_tensor)

        with torch.no_grad():
            z_inv, mu_inv, logvar_inv, z_e, mu_e, logvar_e = self.viae.encode(x_raw, modulations, deterministic=True)

        # Step 2: Hutchinson Topological Radar
        z_curr = z_inv[:, -1, :]
        transition_fn = lambda z: self.forecaster.forward_head(
            torch.cat([z, torch.zeros(z.shape[0], self.forecaster.head_in_dim - z.shape[1], device=device)], dim=-1)
            .unsqueeze(1).repeat(1, self.d_inv, 1)
        )[:, 0, :]

        with torch.enable_grad():
            divergence = self.radar.estimate_divergence(transition_fn, z_curr)
            tau_mc_t = self.radar.compute_adaptive_threshold(divergence)

        # Step 3: Adaptive MC-Dropout Evaluation
        unc_result = self.mc_engine.evaluate_uncertainty(z_inv, tau_mc=tau_mc_t)
        is_ood = unc_result["is_ood"]
        passes_used = unc_result["passes_used"]
        z_pred_latent = unc_result["mean"]
        h_cached = unc_result["h_cached"]

        repair_applied = False
        repair_latency_ms = 0.0
        annihilation_norm = 0.0

        if not is_ood:
            # Clean invariant decoding (Z_e ignored)
            with torch.no_grad():
                y_forecast = self.viae.decode(z_pred_latent, None)
        else:
            # Active Causal Repair
            start_repair = time.perf_counter()
            beta_tight = torch.tensor([[self.beta_tightened]], device=device, dtype=torch.float32)
            modulations_tight = self.hypernet(beta_tight)

            with torch.no_grad():
                _, _, _, z_e_tight, _, _ = self.viae.encode(x_raw, modulations_tight, deterministic=True)

            head_dim = self.forecaster.head_in_dim
            p_inv = self.patcher.compute_nullspace_projection(z_e_tight, target_dim=head_dim, d_e=self.d_e)

            w_orig = self.forecaster.linear_head.weight
            w_repaired = self.patcher.patch_linear_weights(w_orig, p_inv)
            _, annihilation_norm = self.patcher.verify_annihilation(w_repaired, z_e_tight)

            b_head, d_head, feat_dim = h_cached.shape
            h_flat = h_cached.view(b_head * d_head, feat_dim)
            out_flat = torch.matmul(h_flat, w_repaired.t())
            z_repaired = out_flat.view(b_head, d_head, self.forecaster.pred_len).permute(0, 2, 1).contiguous()

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
            "max_variance": unc_result["max_var"],
            "repair_applied": repair_applied,
            "repair_latency_ms": repair_latency_ms,
            "total_latency_ms": total_latency_ms,
            "annihilation_norm": annihilation_norm
        }