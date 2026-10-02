"""Phase 2 verification: PICA + active VIAE (no collapse, invariance).

Success thresholds (spec Section 5, Phase 2):
    * No collapse: Var(Z_inv) > 0.1 across latent dimensions
    * PICA projection suppresses cross-environment covariance differences
    * ActiveVIAE forward shapes correct in full FP32
"""

from __future__ import annotations

import torch

from src.stage1_state_constructor.pica_projector import PICAProjector
from src.stage1_state_constructor.viae_module import ActiveVIAE


def test_pica_suppresses_covariance_shift():
    """Projecting onto ker(Sigma_1 - Sigma_2) must shrink the covariance gap."""
    torch.manual_seed(42)
    d, n = 4, 2000
    env1 = torch.randn(n, d, dtype=torch.float32)
    env2 = torch.randn(n, d, dtype=torch.float32) * torch.tensor([3.0, 1.0, 1.0, 1.0])

    raw_diff = torch.norm(torch.cov(env1.t()) - torch.cov(env2.t()))

    pica = PICAProjector(in_features=d, invariant_dim=d - 1)
    pica.fit(env1, env2)

    proj_diff = torch.norm(torch.cov(pica.project_invariant(env1).t()) - torch.cov(pica.project_invariant(env2).t()))

    assert proj_diff < 0.5 * raw_diff
    assert proj_diff < 0.5


def test_pica_projection_shape_and_fp32():
    torch.manual_seed(0)
    pica = PICAProjector(in_features=7, invariant_dim=6)
    x = torch.randn(8, 96, 7, dtype=torch.float32)
    out = pica.project_invariant(x)
    assert out.shape == (8, 96, 7)
    assert out.dtype == torch.float32


def test_viae_forward_shapes_fp32():
    torch.manual_seed(0)
    viae = ActiveVIAE(in_features=7, seq_len=96, d_inv=8, d_e=4, hidden_dim=64)
    out = viae(torch.randn(8, 96, 7, dtype=torch.float32))
    assert out["z_inv"].shape == (8, 96, 8)
    assert out["z_e"].shape == (8, 96, 4)
    assert out["x_recon"].shape == (8, 96, 7)
    assert out["x_recon"].dtype == torch.float32


def test_viae_no_collapse_var_zinv_gt_0p1():
    """Spec gate: Var(Z_inv) must exceed 0.1 (no posterior collapse)."""
    torch.manual_seed(0)
    viae = ActiveVIAE(in_features=7, seq_len=96, d_inv=8, d_e=4, hidden_dim=64)
    out = viae(torch.randn(16, 96, 7, dtype=torch.float32))
    z_inv_var = torch.var(out["z_inv"], dim=0).mean().item()
    assert z_inv_var > 0.1, f"posterior collapse: Var(Z_inv) = {z_inv_var:.4f}"

