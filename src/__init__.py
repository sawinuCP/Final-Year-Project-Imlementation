"""CausalTSF-Repair (Active-UI-TSF): self-healing causal time-series forecasting.

Package layout (spec Section 3):
    data_engine                dataset ingestion + synthetic shortcut injection (Phase 1)
    stage1_state_constructor   PICA, VIAE, Markov prior, hypernetwork, CRIB loss (Phase 2)
    stage2_forecaster          latent PatchTST, activation cache, MC-dropout gates (Phase 3)
    stage3_causal_guard        causal guardrail + null-space weight patcher (Phase 4)
    evaluation                 metrics + comparative benchmark runner (Phase 5)

Numerical mandate: full 32-bit single precision (FP32) everywhere.
"""

__version__ = "0.1.0"
