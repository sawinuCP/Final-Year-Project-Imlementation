"""CausalTSF-Repair (Active-UI-TSF) -- master pipeline execution.

Usage:
    python run_pipeline.py --mode verify      # full verification test suite
    python run_pipeline.py --mode benchmark   # ablation study + metrics
    python run_pipeline.py --mode dashboard   # Streamlit diagnostic UI
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="CausalTSF-Repair Master Pipeline Execution")
    parser.add_argument("--mode", type=str, default="benchmark",
                        choices=["verify", "benchmark", "dashboard"],
                        help="Execution mode: verify, benchmark, or dashboard")
    parser.add_argument("--config", type=str, default="configs/data/etth1.yaml",
                        help="Path to dataset configuration YAML")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Target execution hardware (cuda or cpu)")
    args = parser.parse_args()

    # Full FP32 mandate (spec 1.2)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"CausalTSF-Repair Initializing | Mode: {args.mode} | Target Device: {args.device}")

    if args.mode == "verify":
        print("\nRunning System Integrity and Verification Test Suite...")
        # pytest is required: the verification suite uses pytest-style test
        # functions, which unittest discovery does not collect.
        import pytest
        raise SystemExit(pytest.main(["tests", "-v"]))

    elif args.mode == "benchmark":
        from src.evaluation.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(config_path=args.config, device=args.device)
        runner.run_full_comparative_benchmark()

    elif args.mode == "dashboard":
        import subprocess
        print("\nLaunching Interactive Streamlit Diagnostic Dashboard...")
        subprocess.run(["streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()

