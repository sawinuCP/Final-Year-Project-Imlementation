"""Phase 2 verification: Stage-1 joint objective (Hyper-CRIB) end-to-end."""

import torch

from src.stage1_state_constructor.crib_loss import CRIBLoss
from src.stage1_state_constructor.hypernetwork import HypernetworkController
from src.stage1_state_constructor.pica_projector import PICAProjector
from src.stage1_state_constructor.viae_module import ActiveVIAE


def test_stage1_pipeline():
    print("=" * 70)
    print("Executing Phase 2 Verification: Stage 1 Active Hyper-CRIB Pipeline")
    print("=" * 70)

    batch_size = 16
    seq_len = 96
    in_features = 7
    d_inv = 16
    d_e = 4
    hidden_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} (Precision: FP32)")

    print("\n[Test 1] Testing PICA Null-Space Covariance Projection...")
    pica = PICAProjector(in_features=in_features, invariant_dim=6).to(device)
    x_chunk1 = torch.randn(100, in_features, device=device, dtype=torch.float32)
    x_chunk2 = torch.randn(100, in_features, device=device, dtype=torch.float32) + 2.0
    pica.fit(x_chunk1, x_chunk2)
    x_dummy = torch.randn(batch_size, seq_len, in_features, device=device, dtype=torch.float32)
    x_inv_target = pica.project_invariant(x_dummy)
    assert x_inv_target.shape == (batch_size, seq_len, in_features)
    print("PICA initialization passed.")

    print("\n[Test 2] Testing Hypernetwork Rate-Distortion Parameter Generation...")
    hypernet = HypernetworkController(modulation_dims=[hidden_dim], hidden_dim=64).to(device)
    modulations = hypernet(torch.tensor([[1.5]], device=device, dtype=torch.float32))
    assert modulations["scales"][0].shape == (1, hidden_dim)
    print("Hypernetwork forward generation passed.")

    print("\n[Test 3] Testing Active VIAE Dual-Branch Latent Disentanglement...")
    viae = ActiveVIAE(
        in_features=in_features, seq_len=seq_len, d_inv=d_inv, d_e=d_e, hidden_dim=hidden_dim
    ).to(device)
    out_clean = viae(x_dummy, modulations)
    assert out_clean["z_inv"].shape == (batch_size, seq_len, d_inv)
    assert out_clean["z_e"].shape == (batch_size, seq_len, d_e)
    assert out_clean["x_recon"].shape == (batch_size, seq_len, in_features)
    assert out_clean["x_recon_inv"].shape == (batch_size, seq_len, in_features)

    noise = torch.randn_like(x_dummy) * 0.05
    out_perturbed = viae(x_dummy + noise, modulations)
    print("VIAE clean and perturbed passes passed.")

    print("\n[Test 4] Testing CRIB Loss Computation & Backpropagation...")
    crib_loss_fn = CRIBLoss(d_inv=d_inv, gamma_consistency=0.5, lambda_res=0.2).to(device)
    domain_labels = torch.zeros(batch_size, device=device, dtype=torch.long)
    domain_labels[batch_size // 2 :] = 1
    x_inv_target = viae.pica.project_invariant(x_dummy)
    loss_dict = crib_loss_fn(
        out_clean,
        out_perturbed,
        x_dummy,
        x_inv_target,
        beta=1.5,
        domain_labels=domain_labels,
    )
    assert loss_dict["recon_inv"].item() >= 0.0
    loss_dict["loss"].backward()
    has_viae_grad = any(p.grad is not None and p.grad.norm().item() > 0 for p in viae.parameters())
    assert has_viae_grad, "VIAE parameters failed to receive gradients."
    print("Gradient backpropagation verified.")

    print("\n[Test 5] Posterior Collapse Assertion...")
    z_inv_var = torch.var(out_clean["z_inv"], dim=0).mean().item()
    assert z_inv_var > 0.05, f"Posterior collapse: Var(Z_inv) = {z_inv_var:.4f}"
    print("\nPhase 2 complete: Stage 1 Hyper-CRIB tests passed.")


if __name__ == "__main__":
    test_stage1_pipeline()
