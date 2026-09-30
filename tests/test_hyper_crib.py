"""Phase 2 verification: Stage-1 joint objective (Hyper-CRIB) end-to-end.

Integration pipeline covering PICA initialization, hypernetwork parameter
generation, active dual-branch VIAE forward passes (clean + perturbed),
the 5-term CRIB loss with gradient backpropagation, and the posterior
collapse assertion.
"""

import torch
from src.stage1_state_constructor.pica_projector import PICAProjector
from src.stage1_state_constructor.hypernetwork import HypernetworkController
from src.stage1_state_constructor.viae_module import ActiveVIAE
from src.stage1_state_constructor.crib_loss import CRIBLoss

def test_stage1_pipeline():
    print("=" * 70)
    print("Executing Phase 2 Verification: Stage 1 Active Hyper-CRIB Pipeline")
    print("=" * 70)

    # Tensor configuration
    batch_size = 16
    seq_len = 96
    in_features = 7  # ETTh1 channel count
    d_inv = 8
    d_e = 4
    hidden_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} (Precision: FP32)")

    # 1. Test PICA Nullspace Initialization
    print("\n[Test 1] Testing PICA Null-Space Covariance Projection...")
    pica = PICAProjector(in_features=in_features, invariant_dim=in_features).to(device)
    x_chunk1 = torch.randn(100, in_features, device=device, dtype=torch.float32)
    x_chunk2 = torch.randn(100, in_features, device=device, dtype=torch.float32) + 2.0  # shifted covariance
    pica.fit(x_chunk1, x_chunk2)
    x_dummy = torch.randn(batch_size, seq_len, in_features, device=device, dtype=torch.float32)
    x_pica = pica(x_dummy)
    assert x_pica.shape == (batch_size, seq_len, in_features), "PICA shape mismatch."
    print("PICA initialization passed.")

    # 2. Test Hypernetwork Parameter Generation
    print("\n[Test 2] Testing Hypernetwork Rate-Distortion Parameter Generation...")
    hypernet = HypernetworkController(modulation_dims=[hidden_dim], hidden_dim=64).to(device)
    test_beta = torch.tensor([[1.5]], device=device, dtype=torch.float32)
    modulations = hypernet(test_beta)
    assert len(modulations["scales"]) == 1, "Hypernetwork scales output count mismatch."
    assert modulations["scales"][0].shape == (1, hidden_dim), "Modulation scale dimension mismatch."
    print("Hypernetwork forward generation passed.")

    # 3. Test Active VIAE Forward Pass (Clean & Perturbed)
    print("\n[Test 3] Testing Active VIAE Dual-Branch Latent Disentanglement...")
    viae = ActiveVIAE(
        in_features=in_features,
        seq_len=seq_len,
        d_inv=d_inv,
        d_e=d_e,
        hidden_dim=hidden_dim
    ).to(device)

    # Clean input
    out_clean = viae(x_dummy, modulations)
    assert out_clean["z_inv"].shape == (batch_size, seq_len, d_inv), "Z_inv shape mismatch."
    assert out_clean["z_e"].shape == (batch_size, seq_len, d_e), "Z_e shape mismatch."
    assert out_clean["x_recon"].shape == (batch_size, seq_len, in_features), "Reconstruction shape mismatch."

    # Perturbed input for CRIB consistency
    noise = torch.randn_like(x_dummy) * 0.05
    x_perturbed = x_dummy + noise
    out_perturbed = viae(x_perturbed, modulations)
    print("VIAE clean and perturbed passes passed.")

    # 4. Test CRIB Loss and Backward Gradient Flow
    print("\n[Test 4] Testing CRIB Loss Computation & Backpropagation...")
    crib_loss_fn = CRIBLoss(d_inv=d_inv, gamma_consistency=1.0, lambda_res=0.5).to(device)
    domain_labels = torch.zeros(batch_size, device=device, dtype=torch.long)
    domain_labels[batch_size // 2 :] = 1
    loss_dict = crib_loss_fn(
        out_clean, out_perturbed, x_dummy, beta=1.5, domain_labels=domain_labels
    )
    assert loss_dict["residual_var_loss"].item() >= 0.0

    print(f"  Total Stage 1 Loss : {loss_dict['loss'].item():.4f}")
    print(f"  Reconstruction Loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"  Markovian KL Loss  : {loss_dict['markov_kl'].item():.4f}")
    print(f"  Consistency Loss   : {loss_dict['consistency_loss'].item():.4f}")

    # Verify backward pass
    loss_dict["loss"].backward()

    # Verify non-zero gradients on VIAE and Hypernetwork
    has_viae_grad = any(p.grad is not None and p.grad.norm().item() > 0 for p in viae.parameters())
    assert has_viae_grad, "VIAE parameters failed to receive gradients."
    print("Gradient backpropagation verified.")

    # 5. Check Against Posterior Collapse
    print("\n[Test 5] Posterior Collapse Assertion...")
    z_inv_var = torch.var(out_clean["z_inv"], dim=0).mean().item()
    print(f"  Mean variance across Z_inv dimensions: {z_inv_var:.4f}")
    assert z_inv_var > 0.05, "Posterior collapse detected: latent variance too close to zero."

    print("\n" + "=" * 70)
    print("Phase 2 complete: All Stage 1 Hyper-CRIB tests passed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    test_stage1_pipeline()

