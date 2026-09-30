"""Invariant Risk Minimization (IRMv1) PatchTST Baseline.

Applies the IRMv1 gradient penalty across distinct temporal environments (E_0, E_1)
to enforce an invariant representation during training.
Reference: Arjovsky et al. (2019) "Invariant Risk Minimization"
           Liu et al. (ICML 2024) "FOIL: Time-Series Forecasting for OOD via Invariant Learning".
"""

import torch
import torch.nn as nn
from .erm_patchtst import ERMPatchTST

__all__ = ["IRMPatchTST"]


class IRMPatchTST(nn.Module):
    """
    PatchTST trained with the Invariant Risk Minimization (IRMv1) gradient norm penalty.
    """
    def __init__(
        self,
        in_features: int = 7,
        seq_len: int = 96,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64
    ):
        super(IRMPatchTST, self).__init__()
        self.backbone = ERMPatchTST(
            in_features=in_features,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def compute_irm_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        domain_ids: torch.Tensor,
        lambda_irm: float = 1.0
    ) -> dict:
        """
        Computes the empirical risk + IRMv1 gradient penalty:
            Penalty = || nabla_{w|w=1.0} Loss_e(w * y_pred_e, y_e) ||^2_2
        Args:
            x: Input observations [B, L, D]
            y: Ground-truth future targets [B, H, D]
            domain_ids: Environment indicators [B] (0 or 1)
            lambda_irm: Invariance penalty weight
        """
        unique_domains = torch.unique(domain_ids)
        total_risk = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        total_penalty = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        # Forward pass through backbone
        y_preds = self.backbone(x)

        for d_id in unique_domains:
            mask = (domain_ids == d_id)
            if mask.sum() == 0:
                continue

            y_pred_e = y_preds[mask]
            y_true_e = y[mask]

            # Introduce dummy scalar multiplier w = 1.0 with requires_grad=True
            w_dummy = torch.tensor(1.0, device=x.device, dtype=torch.float32, requires_grad=True)
            loss_e = nn.functional.mse_loss(y_pred_e * w_dummy, y_true_e)

            # Compute gradient with respect to w_dummy
            grad_w = torch.autograd.grad(loss_e, [w_dummy], create_graph=True)[0]
            penalty_e = grad_w.pow(2)

            total_risk += loss_e
            total_penalty += penalty_e

        num_domains = max(1, len(unique_domains))
        avg_risk = total_risk / num_domains
        avg_penalty = total_penalty / num_domains
        loss_total = avg_risk + (lambda_irm * avg_penalty)

        return {
            "loss": loss_total,
            "risk": avg_risk.item(),
            "penalty": avg_penalty.item()
        }