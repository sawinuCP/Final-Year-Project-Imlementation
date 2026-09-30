# CausalTSF-Repair (Active-UI-TSF)

An automated, **self-healing** time-series forecasting framework that resolves
**Latent Chaos** and **Shortcut Learning** (the *Clever Hans* effect) in deep
forecasting models — with **0% manual labels** and **sub-2 ms runtime repair**.

> Implementation follows
> `CausalTSF-Repair_ Engineering Implementation Plan & Technical Blueprint.md`.

## Why this is novel

Existing latent forecasting frameworks (e.g. LatentTSF) **freeze** their
pre-trained autoencoders — permanently locking training shortcuts into the
latent state space. CausalTSF-Repair instead runs an **active, unfrozen**
latent constructor and closes the loop with a **real-time causal repair**
mechanism:

1. **Stage 1 — Active Invariant Manifold Construction (Hyper-CRIB)**
   Unfrozen **Variational Invariant Autoencoder (VIAE)** factorizing
   representations into an invariant physical channel `Z_inv` and a spurious
   environmental channel `Z_e`, trained with a **Consistency-Regularized
   Information Bottleneck (CRIB)**, a first-order **Markovian transition
   prior** (anti-Latent-Chaos), **PICA** null-space initialization, and a
   **Hypernetwork** `h_psi(beta)` that learns the whole Rate-Distortion curve
   in a single run.
2. **Stage 2 — Latent Probabilistic Forecasting & Adaptive Gating**
   Channel-independent **PatchTST** over `Z_inv` only, with **Last-Layer
   Activation Caching**, an **Adaptive Dual-Gate MC-Dropout** engine
   (Gate 1: `S_base=10` sub-1 ms preliminary check → Gate 2: up to `S_max=130`
   passes with variance-convergence early exit), and a **Hutchinson Trace
   Topological Radar** setting the dynamic threshold
   `tau_MC(t) = tau0 / (1 + gamma * D_Hutchinson(t))`.
3. **Stage 3 — Real-Time Latent Causal Repair**
   On uncertainty spikes the **Active Causal Guard** computes a closed-form
   orthogonal null-space projection `P_inv = I - U_e U_e^T` and patches the
   linear decision weights `W_repaired = W @ P_inv` in **< 2 ms**, scaling
   shortcut weights to zero (`W_repaired @ Z_e = 0`) with zero catastrophic
   forgetting (dynamic tensor op — master weights untouched).

Key engineering mandates: **full FP32** everywhere, **PatchTST** backbone
(DLinear/TCN toggles), **ETTh1** pilot dataset, **Streamlit** dashboard.

## Project structure

```text
CausalTSF-Repair/
├── configs/
│   ├── data/          etth1.yaml (pilot dataset)
│   ├── model/         stage1_hyper_crib / stage2_patchtst / stage3_causal_guard .yaml
│   └── shortcuts/     sine_hum / baseline_drift .yaml
├── data/
│   ├── raw/           downloaded benchmark CSVs (git-ignored)
│   └── processed/     windowed, normalized tensors (git-ignored)
├── src/
│   ├── data_engine/   dataset_loader.py, shortcut_injector.py          [Phase 1]
│   ├── stage1_state_constructor/  pica_projector, viae_module,
│   │                  markov_prior, hypernetwork, crib_loss            [Phase 2]
│   ├── stage2_forecaster/  patchtst_module, activation_cache,
│   │                  mc_dropout_engine, hutchinson_radar              [Phase 3]
│   ├── stage3_causal_guard/  causal_guardrail, nullspace_patcher       [Phase 4]
│   └── evaluation/    metrics.py, benchmark_runner.py                  [Phase 5]
├── dashboard/         app.py (Streamlit diagnostic UI)                 [Phase 5]
├── scripts/           train_stage1.py, train_stage2.py, train_baselines.py
├── tests/             one verification file per phase milestone
├── requirements.txt
├── run_pipeline.py    unified CLI entry point
└── README.md
```

## Setup

```bash
# Python 3.11+, PyTorch 2.x (CPU or CUDA)
pip install -r requirements.txt
python run_pipeline.py --mode verify
```

## Training (checkpoints for benchmark)

Run from the **project root** (paths `checkpoints/` and `data/` are resolved automatically):

```bash
python scripts/train_stage1.py
python scripts/train_stage2.py
python scripts/train_baselines.py
```

## CLI

```bash
python run_pipeline.py --mode verify      # full verification test suite (pytest)
python run_pipeline.py --mode benchmark   # comparative benchmark -> benchmark_results.json
python run_pipeline.py --mode dashboard   # Streamlit diagnostic UI
pytest tests/ -v
```

## Verification matrix (spec Section 5)

| Phase | Component | Verification | Threshold |
| ----- | --------- | ------------ | --------- |
| 1 | Shortcut engine | `tests/test_shortcut_injector.py` | corr >= 90% train, <= 10% OOD; ERM OOD MSE degradation > 40% |
| 2 | PICA + VIAE | `tests/test_pica_and_viae.py` | Var(Z_inv) > 0.1; smooth t-SNE |
| 2 | Hypernetwork | `tests/test_hypernetwork.py` | valid weights in < 1.0 ms for any beta |
| 3 | Latent PatchTST + gating | `tests/test_mc_dropout_gating.py` | Gate 1 < 1.0 ms; sigma^2 spike > 300% on OOD |
| 4 | Null-space patcher | `tests/test_nullspace_repair.py` | ||W_repaired @ Z_e|| = 0.0; repair < 2.0 ms; 0% forgetting |
| 5 | Benchmarks | `run_pipeline.py --mode benchmark` | 15–20% MSE improvement over FOIL/LatentTSF; CCR >= 98.5% |
| 5 | Dashboard | `streamlit run dashboard/app.py` | < 50 ms refresh cycles |

Current status: **Phases 1–5 fully implemented** — data pipeline +
shortcut injection (Phase 1), the Hyper-CRIB stack (Phase 2), the latent
forecasting stack (Phase 3), the causal repair stack (Phase 4), and the
evaluation layer (multi-metric engine, ablation/benchmark harness with JSON
export, Streamlit diagnostic dashboard, Phase 5) are all functional and
unit-tested. Remaining work for paper-grade results: real training loops
(Stage-1 VIAE + Stage-2 forecaster on ETTh1), a trained ERM baseline to
replace the simulated comparison variant, and GPU (T4) latency validation.

## Reference literature

- **PatchTST:** Nie et al., *"A Time Series is Worth 64 Words"*, ICLR 2023 (arXiv:2211.14730)
- **LatentTSF:** Yang et al., *"From Observations to States: Latent Time Series Forecasting"*, ICML 2026
- **CRIB:** Yang et al., arXiv:2509.23494
- **Hyper-VIB:** Peng et al., arXiv:2511.15041
- **PICA/IRM:** Norman & Meir, ICLR 2026
- **Topological flow divergence:** Wu et al., arXiv:2603.17385
- **OPLoRA:** Xiong & Xie, AAAI-26
- **Causal repair:** Vares & Johnson, CauSE Workshop @ ESEC/FSE 2025
- **Conformalised Monte Carlo (MC-CP):** Bethell et al., arXiv:2308.09647
- **FOIL:** Liu et al., ICML 2024 · **InvarNet:** Zhang et al., WWW 2024
