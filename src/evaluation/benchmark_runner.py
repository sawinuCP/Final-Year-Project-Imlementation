"""Empirical SOTA Benchmark Harness."""

import os
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data_engine.dataset_loader import ETTh1Dataset
from src.data_engine.shortcut_injector import SyntheticShortcutInjector, build_shortcut_injector
from src.evaluation.metrics import EvaluationMetrics

from src.baselines.erm_patchtst import ERMPatchTST
from src.baselines.latent_tsf import LatentTSFModel
from src.baselines.irm_patchtst import IRMPatchTST
from src.stage1_state_constructor.hypernetwork import HypernetworkController
from src.stage1_state_constructor.viae_module import ActiveVIAE
from src.stage2_forecaster.patchtst_module import LatentPatchTST
from src.stage2_forecaster.hutchinson_radar import HutchinsonTopologicalRadar
from src.stage2_forecaster.mc_dropout_engine import AdaptiveMCDropoutEngine
from src.stage3_causal_guard.causal_guardrail import ActiveCausalGuard

__all__ = ["BenchmarkRunner"]


class BenchmarkRunner:
    def __init__(self, config_path: str = "configs/data/etth1.yaml", device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.metrics = EvaluationMetrics()

    def load_test_loader(self):
        test_dataset = ETTh1Dataset(root_path="data/raw/ETTh1.csv", flag="test", size=(96, 96), features="M")
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        injector = build_shortcut_injector()
        return loader, injector

    def evaluate_model(self, model_name: str, predict_fn, loader: DataLoader, injector: SyntheticShortcutInjector, is_ood: bool = False, max_batches: int = 15) -> dict:
        mse_list, mae_list, crps_list, ccr_list, latencies = [], [], [], [], []
        repair_triggers, total_batches = 0, 0

        for i, (bx, by, _) in enumerate(loader):
            if i >= max_batches:
                break
            bx, by = bx.to(self.device), by.to(self.device)
            bx_eval = injector.inject_shortcut(bx, y=by, is_ood=is_ood)

            t0 = time.perf_counter()
            pred, res_dict = predict_fn(bx_eval)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            latencies.append(lat_ms)
            total_batches += 1

            if res_dict.get("repair_applied", False):
                repair_triggers += 1

            mse_list.append(self.metrics.mse(pred, by))
            mae_list.append(self.metrics.mae(pred, by))
            ccr_list.append(self.metrics.causal_consistency_rate(pred))

            var_est = res_dict.get("mean_variance", 0.02)
            std_est = max(1e-4, float(var_est)) ** 0.5
            noise_samples = torch.randn(20, *pred.shape, device=self.device) * std_est
            samples = pred.unsqueeze(0) + noise_samples
            crps_list.append(self.metrics.crps_empirical(samples, by))

        return {
            "Model": model_name,
            "Setting": "OOD (Shifted)" if is_ood else "In-Distribution",
            "MSE": float(np.mean(mse_list)),
            "MAE": float(np.mean(mae_list)),
            "CRPS": float(np.mean(crps_list)),
            "CCR": float(np.mean(ccr_list)),
            "Mean_Latency_ms": float(np.mean(latencies)),
            "Repair_Trigger_Rate": float(repair_triggers / max(1, total_batches)) * 100.0
        }

    def run_full_comparative_benchmark(self, output_file: str = "benchmark_results.json"):
        print("\n" + "=" * 90)
        print("RUNNING COMPREHENSIVE EMPIRICAL SOTA BENCHMARK ON REAL CHECKPOINTS")
        print("=" * 90)

        loader, injector = self.load_test_loader()
        in_features, seq_len, pred_len = 7, 96, 96
        hidden_dim, d_inv, d_e = 64, 16, 4

        # 1. Baseline: ERM
        erm_model = ERMPatchTST(in_features=in_features, seq_len=seq_len, pred_len=pred_len).to(self.device)
        erm_model.load_state_dict(torch.load("checkpoints/baselines/erm_patchtst.pt", map_location=self.device))
        erm_model.eval()
        predict_erm = lambda x: (erm_model(x), {"mean_variance": 0.015, "repair_applied": False})

        # 2. Baseline: LatentTSF
        latent_tsf = LatentTSFModel(in_features=in_features, seq_len=seq_len, pred_len=pred_len, d_latent=8).to(self.device)
        latent_tsf.load_state_dict(torch.load("checkpoints/baselines/latent_tsf.pt", map_location=self.device))
        latent_tsf.freeze_autoencoder()
        latent_tsf.latent_forecaster.eval()
        predict_latent_tsf = lambda x: (latent_tsf(x), {"mean_variance": 0.020, "repair_applied": False})

        # 3. Baseline: IRM-PatchTST
        irm_model = IRMPatchTST(in_features=in_features, seq_len=seq_len, pred_len=pred_len).to(self.device)
        irm_model.load_state_dict(torch.load("checkpoints/baselines/irm_patchtst.pt", map_location=self.device))
        irm_model.eval()
        predict_irm = lambda x: (irm_model(x), {"mean_variance": 0.025, "repair_applied": False})

        # 4 & 5. Proposed: CausalTSF-Repair
        s1_ckpt = torch.load("checkpoints/stage1_viae.pt", map_location=self.device)
        viae = ActiveVIAE(in_features=in_features, seq_len=seq_len, d_inv=d_inv, d_e=d_e, hidden_dim=hidden_dim).to(self.device)
        viae.load_state_dict(s1_ckpt["viae_state_dict"])
        viae.pica.u_null.copy_(s1_ckpt["u_null"])
        if "u_spur" in s1_ckpt:
            viae.pica.u_spur.copy_(s1_ckpt["u_spur"])
        viae.eval()

        hypernet = HypernetworkController(modulation_dims=[hidden_dim], hidden_dim=64).to(self.device)
        hypernet.load_state_dict(s1_ckpt["hypernet_state_dict"])
        hypernet.eval()

        forecaster = LatentPatchTST(seq_len=seq_len, pred_len=pred_len, d_inv=d_inv, patch_len=16, stride=8, d_model=64).to(self.device)
        forecaster.load_state_dict(torch.load("checkpoints/stage2_patchtst.pt", map_location=self.device))
        forecaster.eval()

        tau_base = 0.0001
        if os.path.exists("checkpoints/calibration_stats.json"):
            with open("checkpoints/calibration_stats.json", "r") as f:
                tau_base = json.load(f).get("tau_base", 0.0001)

        radar = HutchinsonTopologicalRadar(tau_base=tau_base, gamma=2.0).to(self.device)
        mc_engine = AdaptiveMCDropoutEngine(forecaster, s_base=10, s_max=30)
        guard = ActiveCausalGuard(viae, forecaster, hypernet, mc_engine, radar).to(self.device)

        def predict_causal_no_repair(x):
            guard.radar.tau_base = 1e9
            res = guard.forward_stream(x)
            return res["forecast"], res

        def predict_causal_full(x):
            guard.radar.tau_base = tau_base
            res = guard.forward_stream(x)
            return res["forecast"], res

        benchmarks = [
            self.evaluate_model("Standard ERM PatchTST", predict_erm, loader, injector, is_ood=False),
            self.evaluate_model("Standard ERM PatchTST", predict_erm, loader, injector, is_ood=True),
            self.evaluate_model("LatentTSF (Frozen AE)", predict_latent_tsf, loader, injector, is_ood=True),
            self.evaluate_model("IRM-PatchTST (FOIL/Invar)", predict_irm, loader, injector, is_ood=True),
            self.evaluate_model("CausalTSF-Repair (No Repair)", predict_causal_no_repair, loader, injector, is_ood=True),
            self.evaluate_model("CausalTSF-Repair (Proposed)", predict_causal_full, loader, injector, is_ood=True)
        ]

        print("\n" + "=" * 90)
        print(f"{'Framework / Baseline':<32} | {'Setting':<16} | {'MSE':<7} | {'MAE':<7} | {'CRPS':<7} | {'CCR (%)':<7} | {'Latency':<7}")
        print("-" * 90)
        for b in benchmarks:
            print(f"{b['Model']:<32} | {b['Setting']:<16} | {b['MSE']:<7.4f} | {b['MAE']:<7.4f} | {b['CRPS']:<7.4f} | {b['CCR']:<7.1f} | {b['Mean_Latency_ms']:<5.2f}ms")
        print("=" * 90)

        with open(output_file, "w") as f:
            json.dump(benchmarks, f, indent=4)
        print(f"\nEmpirical results saved to {output_file}")
        return benchmarks


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_full_comparative_benchmark()