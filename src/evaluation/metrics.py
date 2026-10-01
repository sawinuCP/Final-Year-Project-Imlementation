"""Multi-Dimensional Evaluation Metrics Engine for CausalTSF-Repair.

Computes:
    1. Deterministic Point Accuracy: MSE, MAE
    2. Probabilistic Calibration: CRPS (Continuous Ranked Probability Score), ECE (Expected Calibration Error)
    3. Physical Constraint Compliance: CCR (Causal Consistency Rate)
Strictly implemented in FP32 precision.
"""

import numpy as np
import torch

__all__ = ["EvaluationMetrics"]


class EvaluationMetrics:
    """
    Standardized benchmarking metrics engine for deep time series forecasting.
    """

    @staticmethod
    def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Mean Squared Error."""
        return torch.mean((y_pred.float() - y_true.float()) ** 2).item()

    @staticmethod
    def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Mean Absolute Error."""
        return torch.mean(torch.abs(y_pred.float() - y_true.float())).item()

    @staticmethod
    def crps_empirical(samples: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Empirical Continuous Ranked Probability Score (CRPS).
        Evaluates the sharpness and calibration of predictive sample distributions:
            CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
        Args:
            samples: Monte Carlo samples tensor [N, B, H, D]
            y_true: Ground-truth target tensor [B, H, D]
        Returns:
            Scalar CRPS value averaged across all batch and horizon coordinates.
        """
        n_samples = samples.shape[0]
        y_exp = y_true.unsqueeze(0).float()
        samples_f = samples.float()

        # First term: E_X |X - y|
        term1 = torch.mean(torch.abs(samples_f - y_exp), dim=0)

        # Second term: 0.5 * E_{X, X'} |X - X'|
        m = min(n_samples, 25)
        sub = samples_f[:m]
        diff_matrix = torch.abs(sub.unsqueeze(1) - sub.unsqueeze(0))
        term2 = 0.5 * torch.mean(diff_matrix, dim=(0, 1))

        crps_tensor = term1 - term2
        return torch.mean(crps_tensor).item()

    @staticmethod
    def expected_calibration_error(samples: torch.Tensor, y_true: torch.Tensor, num_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE) for continuous prediction intervals:
            ECE = (1/num_bins) * sum_p | Coverage(p) - p |
        Args:
            samples: Monte Carlo samples [N, B, H, D]
            y_true: Ground truth target [B, H, D]
            num_bins: Quantile granularity (default 10)
        """
        levels = np.linspace(0.1, 0.9, num_bins)
        ece_accum = 0.0
        samples_f = samples.float()
        y_true_f = y_true.float()

        for p in levels:
            alpha = (1.0 - p) / 2.0
            q_low = torch.quantile(samples_f, alpha, dim=0)
            q_high = torch.quantile(samples_f, 1.0 - alpha, dim=0)

            inside = (y_true_f >= q_low) & (y_true_f <= q_high)
            empirical_coverage = torch.mean(inside.float()).item()
            ece_accum += abs(empirical_coverage - p)

        return float(ece_accum / num_bins)

    @staticmethod
    def causal_consistency_rate(
        y_pred: torch.Tensor,
        lower_bound: float = -2.5,     # 99% of standardized normal data sits in [-2.5, 2.5]
        upper_bound: float = 2.5,
        max_rate_of_change: float = 0.8 # Hourly temperature cannot realistically jump > 0.8 std dev
    ) -> float:
        y_f = y_pred.float()
        bounded = (y_f >= lower_bound) & (y_f <= upper_bound)
        delta = torch.abs(y_f[:, 1:, :] - y_f[:, :-1, :])
        smooth = delta <= max_rate_of_change

        bound_score = torch.mean(bounded.float()).item()
        smooth_score = torch.mean(smooth.float()).item()
        return float(0.5 * (bound_score + smooth_score)) * 100.0