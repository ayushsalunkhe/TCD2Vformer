# Novelty and Contribution Boundary Audit

**Status:** **ACADEMICALLY BOUNDED & VERIFIED**

## Delineation of Project Contributions

| Category | Component / Feature | Provenance | Detailed Description |
| :--- | :--- | :--- | :--- |
| **INHERITED FROM D2VFORMER** | Date2Vec Formulation | Wang et al. (IEEE TNNLS / arXiv:2409.11024) | Trigonometric frequency embedding of calendar timestamps using sin/cos functions. |
| | Flexible Forecasting Concept | Wang et al. (2024) | Conceptual goal of arbitrary-length forecasting using continuous temporal query coordinates. |
| | RevIN Normalization | Kim et al. (ICLR 2022) | Reversible instance normalization applied to time series input channels. |
| | Benchmark Datasets | Zhou et al. (AAAI 2021) | ETTh1, ETTh2, ETTm1, and Exchange Rate benchmarks. |
| **PROJECT RECONSTRUCTION** | Parameter-Free PureD2Vformer | **Our Reconstruction (Phase 1)** | Elimination of horizon-dependent linear projection layers (`Linear(d_model, O)`), achieving genuine mathematical parameter independence: $\partial N_{\text{params}} / \partial O \equiv 0$. |
| | Chronological Data Split Audit | **Our Audit (Phase 1)** | Identification and correction of data split leakage and inconsistent test windowing in open implementations. |
| **DIAGNOSTIC CONTRIBUTION** | Attention Entropy Audit | **Our Diagnostic (Phase 2)** | Discovery that cross-temporal Date2Vec attention operates near maximum entropy ($H_{\text{norm}} \approx 0.97$), behaving as an unweighted uniform smoother. |
| | Counterfactual Control Tests | **Our Diagnostic (Phase 2)** | Uniform substitution and position-shuffling controls quantifying the exact informational utility of cross-temporal attention. |
| **EMPIRICAL CONTRIBUTION** | Validation-Selected Temperature | **Our Contribution (Phases 3–4)** | Formal proof that temperature $\tau$ must be selected exclusively via validation loss at $O_{\text{train}}=48$, rejecting test-set lookahead and disproving universal $\tau=4.0$. |
| | Multi-Horizon Cross-Dataset Matrix | **Our Contribution (Phases 3–6)** | Pre-registered evaluation across 4 datasets, 3 seeds, and 6 horizons ($O \in [24, 720]$) with locked test evaluations and checksum verification. |
| **ARCHITECTURAL CONTRIBUTION** | TCD2Vformer Architecture | **Our Contribution (Phase 6)** | Introduction of dynamic, input-conditioned and query-conditioned temperature modulation: $\tau(X) = \text{MLP}(\bar{d}_x)$ and $\tau(t) = \text{MLP}(\bar{d}_{y, t})$. |
| | Horizon-Invariant Temperature MLP | **Our Contribution (Phase 6)** | Mathematical formulation ensuring dynamic temperature generation adds exactly 305 parameters regardless of forecast horizon. |
| **ENGINEERING / IMPLEMENTATION** | Reproducibility & Automation Suite | **Our Engineering (Phases 5–6)** | SHA-256 state-dict checksum verification, automated evaluation runners, interactive standalone browser demo (`demo/index.html`), and turnkey Google Colab GPU suite. |
