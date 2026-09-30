"""Active Causal Guardrail -- runtime safety loop (Phase 4).

Coordinates real-time streaming inference with automated causal repair:
    1. Encode raw X via the ACTIVE (unfrozen) VIAE -> Z_inv, Z_e.
    2. Audit topological divergence (Hutchinson radar) -> tau_MC(t).
    3. Evaluate epistemic uncertainty (adaptive dual-gate MC-Dropout).
    4. NORMAL PATH: decode the mean latent forecast with Z_e = 0 through the
       frozen decoder (sub-millisecond, no repair).
    5. ANOMALY PATH: tighten beta -> 10 via the hypernetwork, rebuild the
       shortcut subspace from the tightened environmental head, construct
       P_inv, transiently patch decision weights via NullSpacePatcher,
       verify annihilation, re-project cached features, and decode.
Latency budget for repair step: < 2 ms.
"""

import time
import torch
import torch.nn as nn
from .nullspace_patcher import NullSpacePatcher

__all__ = ["ActiveCausalGuard"]


class ActiveCausalGuard(nn.Module):
    """
    Active Causal Guardrail Coordinator.
    Oversees the real-time inference and repair loop.
    """
    def __init__(
        self,
        viae,
        forecaster,
        hypernet,
        mc_engine,
        radar,
        beta_normal: float = 1.0,
        beta_tightened: float = 10.0
    ):
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
        """
        Executes real-time streaming inference with automated causal repair.
        Args:
            x_raw: Raw multivariate time-series input tensor of shape [B, L, D].
        Returns:
            Dictionary containing final forecast, uncertainty metrics, repair logs, and latency.
        """
        start_total = time.perf_counter()
        device = x_raw.device
        b, l, d_obs = x_raw.shape

        # Step 1: Active Invariant Encoding (Normal State)
        beta_tensor = torch.tensor([[self.beta_normal]], device=device, dtype=torch.float32)
        modulations = self.hypernet(beta_tensor)

        with torch.no_grad():
            z_inv, mu_inv, logvar_inv, z_e, mu_e, logvar_e = self.viae.encode(x_raw, modulations)

        # Step 2: Hutchinson Topological Radar Audit
        z_curr = z_inv[:, -1, :]  # Current latent coordinate at timestep t [B, d_inv]
        transition_fn = lambda z: self.forecaster.forward_head(
            torch.cat([z, torch.zeros(z.shape[0], self.forecaster.head_in_dim - z.shape[1], device=device)], dim=-1)
            .unsqueeze(1).repeat(1, self.d_inv, 1)
        )[:, 0, :]

        with torch.enable_grad():
            divergence = self.radar.estimate_divergence(transition_fn, z_curr)
            tau_mc_t = self.radar.compute_adaptive_threshold(divergence)

        # Step 3: Adaptive MC-Dropout Uncertainty Evaluation
        unc_result = self.mc_engine.evaluate_uncertainty(z_inv, tau_mc=tau_mc_t)
        is_ood = unc_result["is_ood"]
        passes_used = unc_result["passes_used"]
        z_pred_latent = unc_result["mean"]   # [B, pred_len, d_inv]
        h_cached = unc_result["h_cached"]     # [B, d_inv, head_in_dim]

        # Decision Branching
        repair_applied = False
        repair_latency_ms = 0.0
        annihilation_norm = 0.0

        if not is_ood:
            # -------------------------------------------------------------
            # NORMAL PATH (In-Distribution, Confident, Sub-1ms Execution)
            # -------------------------------------------------------------
            z_e_future = torch.zeros(b, self.forecaster.pred_len, self.d_e, device=device, dtype=torch.float32)
            with torch.no_grad():
                y_forecast = self.viae.decode(z_pred_latent, z_e_future)

        else:
            # -------------------------------------------------------------
            # ANOMALY PATH (OOD Shift Detected: Execute Surgical Causal Repair)
            # -------------------------------------------------------------
            start_repair = time.perf_counter()

            # Action 1: Slide beta up via Hypernetwork to tighten bottleneck
            beta_tight = torch.tensor([[self.beta_tightened]], device=device, dtype=torch.float32)
            modulations_tight = self.hypernet(beta_tight)
            with torch.no_grad():
                _, _, _, z_e_tight, _, _ = self.viae.encode(x_raw, modulations_tight)

            # Action 2: Construct Orthogonal Projection Matrix P_inv from shortcut activations
            head_dim = self.forecaster.head_in_dim
            # Pass the full temporal sequence z_e_tight [B, L, d_e] to guarantee full-rank SVD
            p_inv = self.patcher.compute_nullspace_projection(
                z_e=z_e_tight,
                target_dim=head_dim,
                d_e=self.d_e
            )

            # Action 3: Surgically patch linear decision weights using NullSpacePatcher
            w_original = self.forecaster.linear_head.weight  # [pred_len, head_in_dim]
            w_repaired = self.patcher.patch_linear_weights(w_original, p_inv)

            # Action 4: Verify Annihilation Property
            is_annihilated, residual_norm = self.patcher.verify_annihilation(w_repaired, z_e_tight)
            annihilation_norm = residual_norm

            # Action 5: Forecast with patched weights on cached features (Backbone is never re-run)
            b_head, d_head, feat_dim = h_cached.shape
            h_flat = h_cached.view(b_head * d_head, feat_dim)
            out_flat = torch.matmul(h_flat, w_repaired.t())
            z_repaired_latent = out_flat.view(b_head, d_head, self.forecaster.pred_len).permute(0, 2, 1).contiguous()

            # Action 6: Decode repaired latent trajectory with Z_e = 0
            z_e_future = torch.zeros(b, self.forecaster.pred_len, self.d_e, device=device, dtype=torch.float32)
            with torch.no_grad():
                y_forecast = self.viae.decode(z_repaired_latent, z_e_future)

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