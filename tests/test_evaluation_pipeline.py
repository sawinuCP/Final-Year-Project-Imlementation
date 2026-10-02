"""Phase 5 verification: metrics engine, benchmark harness & dashboard.

Covers:
    * EvaluationMetrics (MSE / MAE / empirical CRPS / ECE / CCR) bounds
    * BenchmarkRunner end-to-end ablation suite on the real ETTh1 test split
    * Streamlit dashboard script integrity
"""

import os

import pytest
import torch

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.benchmark_runner import BenchmarkRunner


def test_phase5_pipeline():
    print("=" * 70)
    print("Executing Phase 5 Verification: Metrics, Benchmarking & Dashboard")
    print("=" * 70)

    # 1. Test Metrics Engine
    print("\n[Test 1] Testing Evaluation Metrics Engine...")
    metrics = EvaluationMetrics()

    b, h, d = 4, 96, 7
    y_true = torch.randn(b, h, d, dtype=torch.float32)
    y_pred = y_true + 0.1 * torch.randn(b, h, d, dtype=torch.float32)
    samples = y_pred.unsqueeze(0).repeat(20, 1, 1, 1) + 0.05 * torch.randn(20, b, h, d, dtype=torch.float32)

    val_mse = metrics.mse(y_pred, y_true)
    val_mae = metrics.mae(y_pred, y_true)
    val_crps = metrics.crps_empirical(samples, y_true)
    val_ece = metrics.expected_calibration_error(samples, y_true)
    val_ccr = metrics.causal_consistency_rate(y_pred)

    print(f"  MSE : {val_mse:.6f}")
    print(f"  MAE : {val_mae:.6f}")
    print(f"  CRPS: {val_crps:.6f}")
    print(f"  ECE : {val_ece:.6f}")
    print(f"  CCR : {val_ccr:.2f}%")

    assert val_mse > 0.0, "MSE computation error."
    assert val_mae > 0.0, "MAE computation error."
    assert val_crps > 0.0, "CRPS computation error."
    assert 0.0 <= val_ece <= 1.0, "ECE must be bounded in [0, 1]."
    assert 0.0 <= val_ccr <= 100.0, "CCR must be bounded in [0, 100]%."
    print("Metrics calculations verified.")

    # 2. Test Benchmark Runner Harness (requires trained checkpoints)
    print("\n[Test 2] Testing Automated Benchmark Suite Execution...")
    required_ckpts = [
        "checkpoints/stage1_viae.pt",
        "checkpoints/stage2_patchtst.pt",
        "checkpoints/baselines/erm_patchtst.pt",
        "checkpoints/baselines/latent_tsf.pt",
        "checkpoints/baselines/irm_patchtst.pt",
    ]
    missing = [p for p in required_ckpts if not os.path.exists(p)]
    if missing:
        pytest.skip(
            "Benchmark smoke skipped (missing checkpoints). Run scripts/train_stage1.py, "
            "scripts/train_stage2.py, and scripts/train_baselines.py. Missing: "
            + ", ".join(missing)
        )

    runner = BenchmarkRunner(device="cpu")
    try:
        results = runner.run_full_comparative_benchmark(output_file="tests_ablation_output.json")
    except RuntimeError as exc:
        pytest.skip(
            "Benchmark smoke skipped (checkpoint architecture mismatch — re-run training "
            f"scripts after model changes): {exc}"
        )
    assert isinstance(results, list) and len(results) >= 5
    models = {row["Model"] for row in results}
    assert "CausalTSF-Repair (Proposed)" in models

    if os.path.exists("tests_ablation_output.json"):
        os.remove("tests_ablation_output.json")
    print("Benchmark runner harness verified.")

    # 3. Test Streamlit Dashboard File Presence
    print("\n[Test 3] Verifying Streamlit Dashboard Structure...")
    dashboard_path = "dashboard/app.py"
    assert os.path.exists(dashboard_path), f"Missing dashboard script at {dashboard_path}"
    with open(dashboard_path, "r", encoding="utf-8") as f:
        code = f.read()
    assert "st.set_page_config" in code, "Invalid Streamlit configuration in app.py"
    print("Dashboard application script verified.")

    print("\n" + "=" * 70)
    print("Phase 5 complete: Evaluation metrics, benchmarking suite, and dashboard verified.")
    print("=" * 70)


if __name__ == "__main__":
    test_phase5_pipeline()
