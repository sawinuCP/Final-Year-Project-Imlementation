# **CausalTSF-Repair: Engineering Implementation Plan & Technical Blueprint**

This document defines the complete, phased engineering implementation plan for **CausalTSF-Repair** (reconfigured as **Active-UI-TSF**). It is written to serve as an actionable, file-by-file specification for building, testing, and benchmarking the framework.

## **1\. Executive Summary & Architectural Overview**

### **1.1 What We Are Building**

Standard Deep Time Series Forecasting (DTSF) models suffer from **Latent Chaos** (temporally disorganized internal representations) and **Shortcut Learning** (the _Clever Hans effect_), where models exploit non-causal background noise (such as sensor baseline drift or machine-specific signatures) to minimize point-wise loss functions like Mean Squared Error (\$\\text{MSE}\$). When deployed out-of-distribution (OOD), these spurious correlations break, leading to catastrophic prediction collapse. Existing latent frameworks (like LatentTSF) freeze their pre-trained autoencoders, which permanently locks training shortcuts into the latent state space.

**CausalTSF-Repair** is an automated, self-healing time-series forecasting framework that resolves this challenge through three coordinated stages:

1. **Stage 1 (Active Invariant Manifold Construction):** Replaces the frozen autoencoder with an active, unfrozen **Variational Invariant Autoencoder (VIAE)**. It factorizes representations into an invariant physical channel (\$Z_{\\text{inv}}\$) and a spurious environmental channel (\$Z_e\$) with 0% manual labels. It uses a **Consistency-Regularized Information Bottleneck (CRIB)** with an auxiliary perturbed view (\$\\tilde{X} = X + \\eta\$), a first-order Markovian transition prior on \$Z_{\\text{inv}}\$ to eliminate Latent Chaos, and a **Hypernetwork (\$h\_\\psi\$)** to learn the entire continuous Rate-Distortion curve (\$\\beta\$-trade-off) in a single training run.
2. **Stage 2 (Latent Probabilistic Forecasting & Adaptive Gating):** Trains a channel-independent **PatchTST** backbone strictly over \$Z_{\\text{inv}}\$. It introduces an **Adaptive Dual-Gate Monte Carlo (MC) Dropout** mechanism with **Last-Layer Activation Caching** and a **Hutchinson Trace Topological Radar** to evaluate epistemic uncertainty without runtime latency bottlenecks.
3. **Stage 3 (Real-Time Latent Causal Repair):** When epistemic uncertainty spikes, the system triggers the **Active Causal Guard**. Instead of running slow, iterative graph optimizations, it computes a closed-form **Orthogonal Null-Space Projection Matrix (\$P\_{\\text{inv}}\$)** and applies it directly to the linear forecasting decision weights (\$W \\in \\mathbb{R}^{h \\times d}\$) in **under 2 milliseconds**, surgically scaling shortcut weights to zero (\$W_{\\text{repaired}} \\cdot Z_e = 0\$) while leaving invariant dynamics untouched.

### **1.2 Core Architectural Decisions**

- **Numerical Precision:** Full 32-bit Single Precision (**FP32**) across all training and inference loops. This eliminates gradient underflow risks in Kullback-Leibler (\$\\text{KL}\$) and Jensen-Shannon (\$\\text{JS}\$) divergence losses, avoids singular matrix collapse during Singular Value Decomposition (\$\\text{SVD}\$) and matrix inversions for \$P_{\\text{inv}}\$, and removes torch.cuda.amp scaling overhead.
- **Primary Forecasting Backbone:** **PatchTST** (_Nie et al., ICLR 2023_). PatchTST processes temporal sequences via sub-series patching and channel-independent attention, which matches our factorized latent space (\$Z_{\\text{inv}}\$ dimensions remain mutually independent without cross-channel contamination). Modular toggles are provided for DLinear and Temporal Convolutional Networks (TCN).
- **Pilot Benchmark Dataset:** **ETTh1** (Electricity Transformer Temperature - Hourly), scaling to ETTm1, Electricity, and Weather.
- **UI & Diagnostic Interface:** **Streamlit** multi-page dashboard displaying real-time streaming forecasts, confidence envelopes (\$\\mu \\pm 2\\sigma\$), topological vector field divergence alerts, and causal weight repair logs.

## **2\. End-to-End System Architecture and Data Flow**

The runtime data processing cycle operates across three sequential stages connected by gated uncertainty checks:

1. **Data Ingestion & Invariant Projection:  
   **
   - Raw multi-variate time-series window \$X \\in \\mathbb{R}^{B \\times L \\times D}\$ enters the pipeline.
   - Principal Invariant Component Analysis (PICA) null-space projection initializes linear invariant features.
   - Unfrozen VIAE encoder maps inputs to factorized latents: \$Z_{\\text{inv}} \\in \\mathbb{R}^{B \\times L \\times d_{\\text{inv}}}\$ and \$Z_e \\in \\mathbb{R}^{B \\times L \\times d_e}\$.
   - The environmental channel \$Z_e\$ is discarded during normal downstream forecasting, isolating the clean physical manifold \$Z_{\\text{inv}}\$.
2. **Latent Forecasting & Activation Caching:  
   **
   - PatchTST processes the purified latent sequence \$Z_{\\text{inv}}\$ and predicts future latent states \$\\hat{Z}\_{\\text{inv}, t+1} \\in \\mathbb{R}^{B \\times H \\times d_{\\text{inv}}}\$.
   - Intermediate activations \$\\mathbf{h}\_{\\text{latent}}\$ are cached immediately prior to the final linear decision layer \$W\$.
3. **Adaptive Dual-Gate Uncertainty Filtering:  
   **
   - _Gate 1 (Sub-1ms Preliminary Check):_ Replicates cached activations \$\\mathbf{h}\_{\\text{latent}}\$ across \$S_{\\text{base}} = 10\$ parallel GPU passes with active dropout.
   - _Topological Audit:_ Hutchinson trace estimator computes local vector field divergence (\$\\mathcal{D}\_{\\text{Hutchinson}}\$), dynamically setting safety threshold \$\\tau_{\\text{MC}}(t)\$.
   - _Decision Branch:_

- - - If \$\\sigma^2_{\\text{prelim}} < \\tau_{\\text{MC}}(t)\$ (In-Distribution): The sample is declared safe. Early stopping triggers immediately; mean prediction is passed to the frozen VIAE decoder to output final forecast \$\\hat{Y} \\in \\mathbb{R}^{B \\times H \\times D}\$.
      - If \$\\sigma^2_{\\text{prelim}} \\ge \\tau_{\\text{MC}}(t)\$ (OOD Anomaly): The sample breaches the threshold. Execution escalates to Gate 2.

1. **Gate 2 Escalation & Latent Causal Repair:  
   **
   - Full sampling sweep unrolls up to \$S_{\\max} = 130\$ passes over cached activations with variance-convergence early exit (\$\\Delta \\text{Var} < \\delta\$).
   - Active Causal Guard slides \$\\beta\$ up via the Hypernetwork to tighten the information bottleneck against high-frequency noise.
   - Causal Guard extracts shortcut activations \$Z_e\$, computes \$P_{\\text{inv}} = \\mathbf{I} - Z_e(Z_e^\\top Z_e)^{-1}Z_e^\\top\$, and patches linear decision weights: \$W_{\\text{repaired}} = W \\cdot P_{\\text{inv}}\$.
   - Repaired latent state is decoded via the frozen VIAE decoder, producing a verified, physically consistent forecast in under 2 milliseconds.

## **3\. Project Directory Structure**

CausalTSF-Repair/

├── configs/

│ ├── data/

│ │ └── etth1.yaml

│ ├── model/

│ │ ├── stage1_hyper_crib.yaml

│ │ ├── stage2_patchtst.yaml

│ │ └── stage3_causal_guard.yaml

│ └── shortcuts/

│ ├── sine_hum.yaml

│ └── baseline_drift.yaml

│

├── data/

│ ├── raw/

│ └── processed/

│

├── src/

│ ├── \__init_\_.py

│ ├── data_engine/

│ │ ├── \__init_\_.py

│ │ ├── dataset_loader.py

│ │ └── shortcut_injector.py

│ │

│ ├── stage1_state_constructor/

│ │ ├── \__init_\_.py

│ │ ├── pica_projector.py

│ │ ├── viae_module.py

│ │ ├── markov_prior.py

│ │ ├── hypernetwork.py

│ │ └── crib_loss.py

│ │

│ ├── stage2_forecaster/

│ │ ├── \__init_\_.py

│ │ ├── patchtst_module.py

│ │ ├── activation_cache.py

│ │ ├── mc_dropout_engine.py

│ │ └── hutchinson_radar.py

│ │

│ ├── stage3_causal_guard/

│ │ ├── \__init_\_.py

│ │ ├── causal_guardrail.py

│ │ └── nullspace_patcher.py

│ │

│ └── evaluation/

│ ├── \__init_\_.py

│ ├── metrics.py

│ └── benchmark_runner.py

│

├── dashboard/

│ └── app.py

│

├── tests/

│ ├── test_dataset_loader.py

│ ├── test_shortcut_injector.py

│ ├── test_pica_and_viae.py

│ ├── test_hyper_crib.py

│ ├── test_hypernetwork.py

│ ├── test_mc_dropout_gating.py

│ ├── test_nullspace_repair.py

│ └── test_evaluation_pipeline.py

│

├── requirements.txt

├── README.md

└── run_pipeline.py

## **4\. Granular Phased Implementation Roadmap**

### **Phase 1: Environment Setup, Data Pipeline & Synthetic Shortcut Injection Engine**

#### **Objectives**

Establish a reproducible FP32 training environment, build automated data ingestion for benchmark datasets, create the synthetic shortcut injection engine to inject controllable spurious correlations, and establish baseline Empirical Risk Minimization (ERM) models.

#### **Step-by-Step Implementation Tasks**

1. **Environment Scaffolding:** Configure requirements.txt with PyTorch 2.x (forcing torch.set_default_dtype(torch.float32)), NumPy, SciPy, Pandas, PyYAML, Matplotlib, and Streamlit.
2. **Standard Dataset Ingestion (**src/data_engine/dataset_loader.py**):**

- - Build automated downloaders and loaders for ETTh1, ETTm1, Electricity, and Weather.
    - Implement sliding-window chunking producing input sequence \$X \\in \\mathbb{R}^{B \\times L \\times D}\$ and target horizon \$Y \\in \\mathbb{R}^{B \\times H \\times D}\$ (default \$L=96, H=96\$), or \$Y \\in \\mathbb{R}^{B \\times H \\times 1}\$ when `features='MS'` (multivariate input, univariate OT target).
    - Implement channel standard scalar normalization calculated strictly over training splits.
    - Emit per-window **`domain_id` ∈ {0, 1}** (first vs second half of each split timeline) for domain-stratified FOIL residual alignment in Stage 1 CRIB (`domain_labels` batch tensor).

1. **Synthetic Shortcut Injection Engine (**src/data_engine/shortcut_injector.py**):  
   **
   - _Shortcut Type A (Device-ID Sine Hum):_ **Target-aligned** blend on a poisoned input channel: ID path mixes ~92% of a target reference signal (mean future OT from \$Y\$ when provided) with a phase-locked sinusoidal cue; OOD path inverts correlation, triples frequency, and applies \$\\pi/2\$ phase shift.
   - _Shortcut Type B (Sensor-ID Linear Baseline Drift):_ Drift \$D_t \\propto t \\cdot \\mathrm{sign}(\\bar{Y})\$ on ID; OOD inverts drift against target magnitude.
   - Factory: `SyntheticShortcutInjector.from_yaml("configs/shortcuts/*.yaml")` maps nested YAML (`type`, `channels`, `train.amplitude`, `train.omega` → `freq`).
   - Injection poisons **inputs \$X\$ only**; optional \$Y\$ is a correlation reference, not modified.
2. **Baseline ERM Forecasting Harness:** Implement a standard observation-space PatchTST and DLinear model trained via MSE loss to establish in-distribution vs. OOD collapse benchmarks.

#### **Mathematical Formulations**

- Standard Normalization:

- \$\$X_{\\text{norm}} = \\frac{X - \\mu_{\\text{train}}}{\\sigma_{\\text{train}} + \\epsilon}\$\$
- Spurious Sine Injection:

- \$\$X_{t, \[k\]}^{\\text{poisoned}} = X_{t, \[k\]} + A \\sin(\\omega t)\$\$

#### **Deliverables & Code Artifacts**

- src/data_engine/dataset_loader.py
- src/data_engine/shortcut_injector.py
- configs/data/etth1.yaml
- tests/test_shortcut_injector.py

#### **Verification Criteria & Milestone Output**

- Automated test confirms synthetic shortcut is \$>90\\%\$ correlated with targets in training split and \$<10\\%\$ correlated in OOD test split.
- Standard ERM PatchTST achieves strong training accuracy but exhibits a \$>40\\%\$ degradation in MSE when evaluated on the OOD test set, verifying the shortcut vulnerability.

### **Phase 2: Stage 1 — Active Invariant State Constructor (Hyper-CRIB)**

#### **Objectives**

Build the active, unfrozen latent space constructor. Isolate invariant dynamics (\$Z_{\\text{inv}}\$) from shortcuts (\$Z_e\$) with 0% manual labels, enforce smooth Markovian transitions to resolve Latent Chaos, and parameterize the autoencoder with a Hypernetwork to learn the full Rate-Distortion curve in one run.

#### **Step-by-Step Implementation Tasks**

1. **Principal Invariant Component Analysis (**src/stage1_state_constructor/pica_projector.py**):**
2. - Partition unlabelled training data into temporal segments \$E_1, E_2\$.
   - Compute covariance matrices \$\\Sigma_1, \\Sigma_2 \\in \\mathbb{R}^{D \\times D}\$.
   - Compute difference matrix \$\\Delta \\Sigma = \\Sigma_1 - \\Sigma_2\$ and perform SVD to extract the kernel null-space \$\\mathbf{U}\_{\\text{null}} = \\text{ker}(\\Delta \\Sigma)\$.
   - Project raw input into invariant linear subspace: \$X_{\\text{pica}} = X \\cdot \\mathbf{U}\_{\\text{null}}\$.
3. **Active Dual-Branch VIAE Architecture (**src/stage1_state_constructor/viae_module.py**):**
4. - **Dilated causal Conv1d backbone** (receptive field over full lookback) with FiLM modulation from the Hypernetwork on encoder features.
   - Construct an encoder with two separate heads:
   - - Invariant Latent Head: outputs \$\\mu_{\\text{inv}}, \\log \\sigma^2_{\\text{inv}} \\in \\mathbb{R}^{d_{\\text{inv}}}\$ (\$d_{\\text{inv}}=8\$).
       - Environmental Shortcut Head: outputs \$\\mu_e, \\log \\sigma^2_e \\in \\mathbb{R}^{d_e}\$ (\$d_e=4\$).
   - Construct a joint decoder network taking concatenated latents \$\[Z_{\\text{inv}}; Z_e\]\$ and reconstructing observations \$\\hat{X} \\in \\mathbb{R}^{B \\times L \\times D}\$.
5. **Markovian Transition Prior Network (**src/stage1_state_constructor/markov_prior.py**):**
6. - Parameterize a lightweight temporal transition network \$f_{\\text{trans}}: \\mathbb{R}^{d_{\\text{inv}}} \\to \\mathbb{R}^{d_{\\text{inv}}}\$.
   - Enforce temporal Markovian prior: \$p(Z_{\\text{inv}, t} \\mid Z_{\\text{inv}, t-1}) = \\mathcal{N}(f_{\\text{trans}}(Z_{\\text{inv}, t-1}), \\sigma^2_{\\text{prior}}\\mathbf{I})\$.
7. **CRIB Consistency & Information Squeeze Loss (**src/stage1_state_constructor/crib_loss.py**):**
8. - For each training batch \$X\$, generate perturbed view \$\\tilde{X} = X + \\eta\$, where \$\\eta \\sim \\mathcal{N}(0, 0.05^2 \\mathbf{I})\$.
   - Pass both views through encoder to produce \$q_\\phi(Z_{\\text{inv}} \\mid X)\$ and \$q_\\phi(Z_{\\text{inv}} \\mid \\tilde{X})\$.
   - Compute consistency loss via KL divergence:
   -
   - \$\$\\mathcal{L}\_{\\text{consistency}} = D_{\\text{KL}}\\left( q_\\phi(Z_{\\text{inv}} \\mid X) \\parallel q_\\phi(Z_{\\text{inv}} \\mid \\tilde{X}) \\right)\$\$
   - Integrate **FOIL-style surrogate residual alignment (IRN)**: when `domain_labels` \([B]\) spans two regimes, penalize mean/variance discrepancy of reconstruction residuals across domains; fallback to batch-halving when labels absent.
9. **Secondary Hypernetwork Controller (**src/stage1_state_constructor/hypernetwork.py**):**
10. - Construct a 3-layer MLP \$h_\\psi: \\mathbb{R}^1 \\to \\mathbb{R}^{\\vert{}\\Theta_{\\text{VIAE}}\\vert{}}\$ taking scalar \$\\beta\$ as input.
    - Train with dynamically sampled \$\\beta\$: \$\\log \\beta \\sim \\mathcal{U}(\\log 0.01, \\log 10.0)\$.
    - Hypernetwork outputs the weights and biases of the VIAE encoder-decoder for that mini-batch.

#### **Mathematical Formulations**

- Stage 1 Joint Training Objective:
-
- \$\$\\mathcal{L}\_{\\text{Stage1}}(\\theta_\\beta, \\phi_\\beta; \\beta) = \\Vert{} X - \\hat{X} \\Vert{}\_2^2 + \\beta \\cdot D_{\\text{KL}}\\left(q_\\phi(Z_{\\text{inv}} \\mid X) \\parallel p(Z_{\\text{inv}})\\right) + D_{\\text{KL}}\\left(q_\\phi(Z_e \\mid X) \\parallel p_e(Z_e)\\right) + \\gamma \\mathcal{L}\_{\\text{consistency}} + \\lambda_{\\text{res}} \\text{Var}\_e(\\text{Res}\_e)\$\$

#### **Deliverables & Code Artifacts**

- src/stage1_state_constructor/pica_projector.py
- src/stage1_state_constructor/viae_module.py
- src/stage1_state_constructor/hypernetwork.py
- src/stage1_state_constructor/crib_loss.py
- tests/test_hyper_crib.py

#### **Verification Criteria & Milestone Output**

- Representation collapse check: Latent variance \$\\text{Var}(Z_{\\text{inv}}) > 0.1\$ across all dimensions (confirming no collapse to zero).
- Smoothness check: Autocorrelation and t-SNE projections of \$Z_{\\text{inv}}\$ show smooth, continuous trajectories across timesteps without discrete cluster shattering.
- Hypernetwork verification: A single forward pass through \$h_\\psi(\\beta)\$ instantiates valid VIAE weights for \$\\beta=0.1\$ and \$\\beta=5.0\$ in \$<1\\text{ms}\$.

### **Phase 3: Stage 2 — Latent Forecasting & Adaptive MC-Dropout Gating**

#### **Objectives**

Train a channel-independent PatchTST forecaster inside the purified \$Z_{\\text{inv}}\$ manifold. Implement activation caching and GPU parallel batching to eliminate the MC-Dropout latency bottleneck, and integrate the Hutchinson trace topological radar to enable dynamic, context-aware threshold scaling.

#### **Step-by-Step Implementation Tasks**

1. **Latent-Space PatchTST Forecaster (**src/stage2_forecaster/patchtst_module.py**):**
2. - Adapt PatchTST to ingest \$Z_{\\text{inv}} \\in \\mathbb{R}^{B \\times L \\times d_{\\text{inv}}}\$ and forecast future states \$\\hat{Z}\_{\\text{inv}, t+1} \\in \\mathbb{R}^{B \\times H \\times d_{\\text{inv}}}\$.
   - Configure patching: patch length \$P=16\$, stride \$S=8\$. Channel independence ensures each latent dimension is processed independently without cross-channel pollution.
   - Integrate standard dropout layers (\$p=0.1\$) after multi-head self-attention and feed-forward blocks.
3. **Last-Layer Activation Caching (**src/stage2_forecaster/activation_cache.py**):  
   **
   - Decouple PatchTST into **`extract_features`** (\$\\mathcal{F}\_{\\text{backbone}}\$) and **`forward_head`** (dropout + linear \$W\$).
   - Cache \$\\mathbf{h}\_{\\text{cached}} \\in \\mathbb{R}^{B \\times d_{\\text{inv}} \\times F}\$ with \$F = \\text{num\_patches} \\cdot d\_{\\text{model}}\$ (default \$11 \\times 64 = 704\$) prior to \$W \\in \\mathbb{R}^{H \\times F}\$.
4. **GPU Parallel MC-Dropout Engine (**src/stage2_forecaster/mc_dropout_engine.py**):  
   **
   - Replicate \$\\mathbf{h}\_{\\text{cached}}\$ across batch dimension to size \$S_{\\text{base}} = 10\$.
   - Apply independent Bernoulli dropout masks simultaneously in a single parallel GPU tensor operation across the linear projection head.
   - Compute preliminary empirical variance:

   - \$\$\\sigma^2_{\\text{prelim}} = \\frac{1}{S_{\\text{base}}} \\sum_{s=1}^{S_{\\text{base}}} (\\hat{Z}^{(s)} - \\bar{Z})^2\$\$
5. **Hutchinson Trace Topological Radar (**src/stage2_forecaster/hutchinson_radar.py**):  
   **
   - Compute stochastic trace estimation of latent velocity field Jacobian using random Rademacher vectors \$v \\sim \\{-1, +1\\}^{d_{\\text{inv}}}\$:

- - \$\$\\mathcal{D}\_{\\text{Hutchinson}} = v^\\top \\nabla_Z f(Z) v \\approx \\text{Tr}(\\mathbf{J}\_f)\$\$
    - Scale safety threshold dynamically:

- - \$\$\\tau_{\\text{MC}}(t) = \\frac{\\tau_0}{1 + \\gamma \\mathcal{D}\_{\\text{Hutchinson}}(t)}\$\$
    - where \$\\tau_0\$ is the 95th percentile variance from clean validation data.

1. **Gate 2 Escalation with Early Exit:**
   - If \$\\sigma^2_{\\text{prelim}} \\ge \\tau_{\\text{MC}}(t)\$, expand batch size up to \$S_{\\max} = 130\$.
   - Monitor consecutive variance delta: if \$\\vert{}\\sigma^2_n - \\sigma^2_{n-1}\\vert{} < 10^{-5}\$ for \$P=5\$ consecutive passes, terminate sampling early.

#### **Mathematical Formulations**

- Epistemic Variance Metric:
- \$\$\\sigma^2_{\\text{epistemic}} = \\frac{1}{N} \\sum_{n=1}^N \\left(\\mathcal{F}\_{\\text{head}}^{(n)}(\\mathbf{h}\_{\\text{cached}}) - \\bar{Z}\\right)^2\$\$
- Rademacher Estimator:
- \$\$\\mathbb{E}\_v \[v^\\top \\mathbf{A} v\] = \\text{Tr}(\\mathbf{A})\$\$

#### **Deliverables & Code Artifacts**

- src/stage2_forecaster/patchtst_module.py
- src/stage2_forecaster/activation_cache.py
- src/stage2_forecaster/mc_dropout_engine.py
- src/stage2_forecaster/hutchinson_radar.py
- tests/test_mc_dropout_gating.py

#### **Verification Criteria & Milestone Output**

- Latency check: Gate 1 preliminary check (\$S_{\\text{base}}=10\$) executes in \$<1.0\\text{ms}\$ on an NVIDIA T4 GPU.
- Calibration check: Epistemic uncertainty \$\\sigma^2\$ remains \$&lt; \\tau_{\\text{MC}}\$ on in-distribution test data, and spikes by \$&gt;300\\%\$ when evaluated on the synthetic shortcut-shifted test split.

### **Phase 4: Stage 3 — Active Causal Guard & Real-Time Null-Space Weight Patching**

#### **Objectives**

Implement the Active Causal Guardrail to execute automated, real-time counterfactual weight repairs. Construct the orthogonal null-space projection matrix (\$P_{\\text{inv}}\$), surgically patch the forecasting linear weights (\$W\$), and verify sub-2ms execution latency without catastrophic forgetting.

#### **Step-by-Step Implementation Tasks**

1. **Active Causal Guardrail Module (**src/stage3_causal_guard/causal_guardrail.py**):  
   **
   - Coordinate the runtime safety loop: ingest \$\\sigma^2_{\\text{prelim}}\$ and \$\\tau_{\\text{MC}}(t)\$ from Stage 2.
   - If an anomaly trigger occurs, call the Hypernetwork to tighten \$\\beta \\to 10.0\$, compressing incoming noise.
   - Forward input through active environmental encoder head to retrieve current shortcut activation subspace \$Z_e \\in \\mathbb{R}^{B \\times d_e}\$.
2. **Orthogonal Null-Space Patcher (**src/stage3_causal_guard/nullspace_patcher.py**):  
   **
   - Center shortcut activations \$Z_e\$ (batch or \$B \\times L\$ flatten), thin-SVD, build shortcut projector \$P_e = V\_{\\text{sub}} V\_{\\text{sub}}^\\top \\in \\mathbb{R}^{d_e \\times d_e}\$.
   - **As-built guard wiring (`ActiveCausalGuard`):** embed \$P_e\$ into PatchTST **head space** \$\\mathbb{R}^{F \\times F}\$ (default \$F=704\$), typically on the last \$d_e\$ columns of \$W\$; \$W\_{\\text{repaired}} = W P\_{\\text{inv}}\$, re-project cached \$\\mathbf{h}\$ without re-running the Transformer.
   - **Latent-block design (module docstring):** block-diagonal \$P\_{\\text{inv}} \\in \\mathbb{R}^{(d\_{\\text{inv}}+d_e) \\times (d\_{\\text{inv}}+d_e)}\$ with identity on \$Z\_{\\text{inv}}\$ and \$(I - P_e)\$ on the \$Z_e\$ block — for composite-latent formulations.
   - Verify annihilation on shortcut head columns: \$\\Vert W_e Z_e^\\top \\Vert_F < \\varepsilon\$.
3. **Decoupled Decoded Output:  
   **
   - Compute repaired latent forecast: \$\\hat{Z}\_{\\text{repaired}} = \\mathbf{h}\_{\\text{cached}} \\cdot W_{\\text{repaired}}^\\top\$.
   - Pass \$\\hat{Z}\_{\\text{repaired}}\$ through the frozen VIAE decoder to output final physically consistent observation forecast \$\\hat{Y} \\in \\mathbb{R}^{B \\times H \\times D}\$.
4. **State Preservation & Anti-Forgetting Verification:**
   - Ensure \$W_{\\text{repaired}}\$ is applied as a dynamic tensor operation during the forward pass and does not overwrite baseline master model weights.

#### **Mathematical Formulations**

- Null-Space Annihilation:

- \$\$P_{\\text{inv}} \\cdot Z_{\\text{inv}} = Z_{\\text{inv}}, \\quad P_{\\text{inv}} \\cdot Z_e = \\mathbf{0}\$\$
- \$\$\\implies W_{\\text{repaired}} Z = \[W_{\\text{inv}} \\mid W_e\] \\begin{bmatrix} Z_{\\text{inv}} \\\\ \\mathbf{0} \\end{bmatrix} = W_{\\text{inv}} Z_{\\text{inv}}\$\$

#### **Deliverables & Code Artifacts**

- src/stage3_causal_guard/causal_guardrail.py
- src/stage3_causal_guard/nullspace_patcher.py
- tests/test_nullspace_repair.py

#### **Verification Criteria & Milestone Output**

- Real-time benchmark: From uncertainty trigger to patched forecast output, total Stage 3 execution latency is \$\\le 1.8\\text{ms}\$ on CPU/GPU.
- Mathematical proof check: \$\\Vert{} W_{\\text{repaired}} \\cdot Z_e \\Vert{} = 0.00000\$, confirming 100% suppression of the shortcut feature.
- Anti-forgetting test: When tested on clean in-distribution data immediately after an OOD intervention, forecast MSE matches clean baseline performance with 0.0% degradation.

### **Phase 5: Comparative Benchmarking, Ablations & Streamlit Dashboard**

#### **Objectives**

Conduct comprehensive empirical benchmarks against SOTA baselines (FOIL, InvarNet, LatentTSF, ERM PatchTST), execute complete ablation studies, and deploy an interactive Streamlit diagnostic dashboard for live demonstration.

#### **Step-by-Step Implementation Tasks**

1. **Unified Evaluation Engine (**src/evaluation/metrics.py**):**
2. - Implement deterministic metrics: Mean Squared Error (\$\\text{MSE}\$) and Mean Absolute Error (\$\\text{MAE}\$).
   - Implement probabilistic metrics: Continuous Ranked Probability Score (\$\\text{CRPS}\$) and Expected Calibration Error (\$\\text{ECE}\$).
   - Implement causal constraint compliance: Causal Consistency Rate (CCR), measuring the percentage of predictions that satisfy domain inequality constraints (\$\\mathbf{A} \\cdot \\hat{Y} \\le \\mathbf{b}\$).
3. **Automated Comparative Benchmark Suite (**src/evaluation/benchmark_runner.py**):  
   **
   - **As-built:** smoke ablation on ETTh1 test loader (`config_path` → YAML), target-aligned shortcut injection (`inject_shortcut(x, y=by, is_ood=...)`), JSON export; full multi-dataset training loops remain future work.
   - Benchmark against:

- - - ERM PatchTST / DLinear (Standard observation-space baseline).
      - InvarNet (_Zhang et al., WWW 2024_) with categorical grouping.
      - FOIL (_Liu et al., ICML 2024_) with surrogate discrete environments.
      - LatentTSF (_Yang et al., ICML 2026_) with frozen autoencoder.
      - CausalTSF-Repair (Proposed Active Framework).

1. **Ablation Study Matrix:** Systematically evaluate and log model performance with:
   - (A) Full CausalTSF-Repair.
   - (B) Removing Hypernetwork (fixed \$\\beta=1.0\$).
   - (C) Removing CRIB consistency loss (\$\\gamma=0\$).
   - (D) Removing PICA null-space initialization.
   - (E) Removing Null-Space Projection (\$P_{\\text{inv}}\$).
2. **Interactive Streamlit Diagnostic Dashboard (**dashboard/app.py**):**
3. - Build multi-page interface:
     - _Page 1: Telemetry Stream & Live Forecasts:_ Interactive Plotly charts showing historical window, true future, and predicted mean with \$\\pm 2\\sigma\$ uncertainty envelopes.
     - _Page 2: Causal Guardrail Monitor:_ Real-time gauges for \$\\sigma^2_{\\text{prelim}}\$, Hutchinson divergence, and the dynamic safety threshold \$\\tau_{\\text{MC}}(t)\$.
     - _Page 3: Latent Manifold Visualizer:_ 2D/3D PCA and t-SNE projections of \$Z_{\\text{inv}}\$ vs. \$Z_e\$ trajectories.
     - _Page 4: Surgical Weight Repair Audit:_ Live table logging intervention timestamps, singular value spectra of \$Z_e\$, and sub-2ms latency measurements.

#### **Mathematical Formulations**

- Continuous Ranked Probability Score:

- \$\$\\text{CRPS}(F, y) = \\int_{-\\infty}^\\infty \\left(F(z) - \\mathbb{I}(z \\ge y)\\right)^2 dz\$\$

#### **Deliverables & Code Artifacts**

- src/evaluation/metrics.py
- src/evaluation/benchmark_runner.py
- dashboard/app.py
- run_pipeline.py

#### **Verification Criteria & Milestone Output**

- CausalTSF-Repair achieves a **15–20% lower MSE/MAE** than LatentTSF and FOIL on shortcut-shifted OOD test sets.
- Causal Consistency Rate (CCR) reaches \$\\ge 98.5\\%\$.
- Streamlit dashboard renders live inference and updates intervention logs in \$<50\\text{ms}\$ refresh cycles.

## **5\. Summary Implementation Verification Matrix**

The following table serves as the primary verification checklist for tracking implementation progress:

| **Phase**   | **Core Component**              | **Primary Code Target**                      | **Unit Test / Verification Script** | **Success Metric Threshold**                                                                         |
| ----------- | ------------------------------- | -------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Phase 1** | Data Pipeline & Shortcut Engine | src/data_engine/shortcut_injector.py         | tests/test_shortcut_injector.py, tests/test_dataset_loader.py | Target-aligned corr \$\\ge 90\\%\$ ID; domain_id for CRIB; `features=M/MS` |
| ---         | ---                             | ---                                          | ---                                 | ---                                                                                                  |
| **Phase 2** | PICA & Active VIAE              | src/stage1_state_constructor/viae_module.py  | tests/test_pica_and_viae.py         | No collapse: \$\\text{Var}(Z_{\\text{inv}}) > 0.1\$; smooth t-SNE                                    |
| ---         | ---                             | ---                                          | ---                                 | ---                                                                                                  |
| **Phase 2** | Hypernetwork Controller         | src/stage1_state_constructor/hypernetwork.py | tests/test_hypernetwork.py          | Continuous weights generated in \$<1.0\\text{ms}\$ for any \$\\beta\$                                |
| ---         | ---                             | ---                                          | ---                                 | ---                                                                                                  |
| **Phase 3** | Latent PatchTST & Caching       | src/stage2_forecaster/patchtst_module.py     | tests/test_mc_dropout_gating.py     | Channel-independent forward pass in full FP32                                                        |
| ---         | ---                             | ---                                          | ---                                 | ---                                                                                                  |
| **Phase 3** | Adaptive MC-Dropout & Radar     | src/stage2_forecaster/mc_dropout_engine.py   | tests/test_mc_dropout_gating.py     | Gate 1 latency \$&lt;1.0\\text{ms}\$; \$\\sigma^2\$ spikes \$&gt;300\\%\$ on OOD                     |
| ---         | ---                             | ---                                          | ---                                 | ---                                                                                                  |
| **Phase 4** | Null-Space Patcher              | src/stage3_causal_guard/nullspace_patcher.py | tests/test_nullspace_repair.py      | \$\\Vert{} W_{\\text{repaired}} \\cdot Z_e \\Vert{} = 0.0\$; total repair latency \$<2.0\\text{ms}\$ |
| ---         | ---                             | ---                                          | ---                                 | ---                                                                                                  |
| **Phase 5** | Benchmarking & Ablation         | src/evaluation/benchmark_runner.py           | run_pipeline.py --mode benchmark    | 15–20% MSE improvement over FOIL/LatentTSF                                                           |
| ---         | ---                             | ---                                          | ---                                 | ---                                                                                                  |
| **Phase 5** | Streamlit UI Dashboard          | dashboard/app.py                             | streamlit run dashboard/app.py      | Interactive multi-page UI updating in \$<50\\text{ms}\$                                              |
| ---         | ---                             | ---                                          | ---                                 | ---                                                                                                  |

## **6\. Implementation Reference Literature**

- **PatchTST Backbone:** Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." _ICLR 2023_.
- **Latent Forecasting & Chaos:** Yang, J., Hu, Y., Li, Y., Zhang, K., Ding, Q., & Yu, P. S. (2026). "From Observations to States: Latent Time Series Forecasting." _ICML 2026_.
- **CRIB Consistency Loss:** Yang, J., Hu, Y., Zhang, K., Luyang, N., Yu, P. S., & Ding, K. (2025/2026). "Revisiting Multivariate Time Series Forecasting with Missing Values." _arXiv:2509.23494_.
- **Hypernetwork Information Bottleneck:** Peng, J., Deng, C., Deng, Y., Ren, B., & Yang, L. (2025). "Hyper-VIB: A Hypernetwork-Enhanced Information Bottleneck Approach." _arXiv:2511.15041_.
- **PICA & Invariant Disentanglement:** Norman, Y., & Meir, R. (2025/2026). "Unsupervised Representation Learning - an Invariant Risk Minimization Perspective." _ICLR 2026_.
- **Topological Flow Divergence:** Wu, R., Xie, X., & Li, Y. J. (2026). "Geometry-Aware Causal Flow: Topological Limits and Entropic Tunneling." _arXiv:2603.17385_.
- **Null-Space Weight Editing:** Xiong, Y., & Xie, X. (2025/2026). "OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting." _AAAI-26_.
- **Causal Repair Foundations:** Vares, F., & Johnson, B. (2025). "Causality-Driven Neural Network Repair: Challenges and Opportunities." _CauSE Workshop at ESEC/FSE 2025_.
- **Adaptive Monte Carlo Calibration:** Bethell, D. (2024). "Robust Uncertainty Quantification using Conformalised Monte Carlo." _arXiv:2308.09647_.