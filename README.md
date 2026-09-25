# Temporal-Conditioned Date2Vecformer (TCD2Vformer)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Audit Status](https://img.shields.io/badge/Audit-Classification%20A%20(Frozen)-brightgreen.svg)](results/FINAL_AUDIT_SUMMARY.md)
[![Upstream Bug Report](https://img.shields.io/badge/Upstream%20Issue-TeamofHaoWang%2FD2Vformer%20%232-orange.svg)](https://github.com/TeamofHaoWang/D2Vformer/issues/2)

> **Empirical Audit, Mathematical Correction, and Dynamic Attention Temperature Conditioning for Truly Arbitrary-Length Time Series Forecasting.**  
> *BE Computer Engineering Major Project (2025–2026)*  
> **Author:** Ayush Harish Salunkhe & Team

---

## 🌟 Executive Summary

[D2Vformer (Wang et al., IEEE TNNLS / arXiv:2409.11024)](https://github.com/TeamofHaoWang/D2Vformer) proposed using continuous trigonometric date representations (**Date2Vec**) for flexible, arbitrary-length time-series forecasting.

During an exhaustive empirical audit for our BE Major Project, we discovered two fundamental limitations in the original architecture:

1. **Horizon-Dependent Parameter Leakage (Architectural Flaw):**  
   The reference implementation employed an output projection layer `nn.Linear(d_model, pred_len)`. As a result, trainable parameter counts scale linearly with the forecast horizon $O$ (44,021 parameters at $O=24$ up to 132,608 at $O=720$). This breaks the core zero-shot claim of arbitrary-length forecasting, as a checkpoint trained at $O=48$ cannot physically execute at $O=720$ without altering the network head.
   - **Reported upstream to original authors:** [GitHub Issue #2 on TeamofHaoWang/D2Vformer](https://github.com/TeamofHaoWang/D2Vformer/issues/2).

2. **Attention Diffusion & Uniform Entropy:**  
   Diagnostic evaluation revealed that cross-temporal Date2Vec attention operates near maximum entropy ($H_{\text{norm}} \approx 0.97$), behaving effectively as an unweighted uniform moving average.

### Our Solution: TCD2Vformer

- **PureD2Vformer (Horizon-Independent Reconstruction):** Replaced the output head with a fixed position-wise FFN ($H \to d_{\text{ff}} \to D$), mathematically enforcing $\frac{\partial N_{\text{params}}}{\partial O} \equiv 0$ (constant 44,021 parameters for all $O \in [24, 720]$).
- **Dynamic Temperature Conditioning:** Introduced input-conditioned ($\tau(X) = \text{MLP}(\bar{D}_x)$) and query-conditioned ($\tau(t) = \text{MLP}(\bar{D}_{y,t})$) temperature scaling. This dynamically sharpens or broadens attention weights based on temporal phase embeddings, adding only 305 parameters total with strict horizon invariance.
- **Empirical Validation:** 288 evaluations across 4 benchmark datasets (ETTh1, ETTh2, ETTm1, Exchange Rate), 6 horizons ($O \in \{24, 48, 96, 192, 336, 720\}$), and 3 random seeds (42, 43, 44), demonstrating up to **+3.12% MSE gain on ETTh2 at $O=720$ (Cohen's $d = 1.67$)**.

---

## 🔬 Parameter Scaling Comparison ($\frac{\partial N}{\partial O}$)

| Forecast Horizon ($O$) | Original D2Vformer | PureD2Vformer (Ours) | TCD2Vformer (Ours) | Horizon Invariant? |
| :---: | :---: | :---: | :---: | :---: |
| **$O = 24$** | 44,021 | 44,021 | **44,326** | ✅ Yes ($\Delta = 0$) |
| **$O = 48$** | 47,093 | 44,021 | **44,326** | ✅ Yes ($\Delta = 0$) |
| **$O = 96$** | 53,237 | 44,021 | **44,326** | ✅ Yes ($\Delta = 0$) |
| **$O = 192$** | 65,525 | 44,021 | **44,326** | ✅ Yes ($\Delta = 0$) |
| **$O = 336$** | 83,957 | 44,021 | **44,326** | ✅ Yes ($\Delta = 0$) |
| **$O = 720$** | 133,109 | 44,021 | **44,326** | ✅ Yes ($\Delta = 0$) |
| **Scaling Property** | $\mathcal{O}(O)$ (Linear Growth) | $\mathcal{O}(1)$ (Strictly Constant) | **$\mathcal{O}(1)$ (Strictly Constant)** | — |

*All 96 audit checks in [`parameter_invariance_audit.csv`](results/final_audit/parameter_invariance_audit.csv) passed with 100% invariance.*

---

## 📊 Key Experimental Findings

### 1. Monotonic Zero-Shot Horizon Gains on ETTh2
Models trained strictly at $O_{\text{train}} = 48$ and evaluated zero-shot at extended horizons:

| Evaluation Horizon ($O$) | Baseline ($\tau=1.0$) MSE | TCD2Vformer MSE | Improvement (%) | Effect Size (Cohen's $d$) |
| :---: | :---: | :---: | :---: | :---: |
| **$O = 48$** | 0.2868 | 0.2868 | +0.00% | — |
| **$O = 96$** | 0.3541 | 0.3529 | **+0.34%** | 0.42 |
| **$O = 192$** | 0.4190 | 0.4151 | **+0.93%** | 0.78 |
| **$O = 336$** | 0.4632 | 0.4531 | **+2.18%** | 1.24 |
| **$O = 720$** | 0.5098 | 0.4939 | **+3.12%** | **1.67** (Large) |

### 2. Scientific Honesty & Boundary Conditions (ETTm1 Failure Analysis)
On high-frequency 15-minute data (ETTm1), sharpening attention ($\tau \approx 0.18$) caused error degradation at $O=720$. Our diagnostic identified that Date2Vec trigonometric encodings overfit high-frequency intra-day harmonics when extrapolated to 720 steps (7.5 days ahead). In noisy, high-frequency regimes, uniform smoothing is mathematically optimal.

---

## 🛠️ Repository Architecture

```text
TCD2Vformer/
├── models/
│   ├── __init__.py
│   ├── pure_d2vformer.py         # Horizon-independent baseline (Eq. 1-10)
│   ├── temperature_d2vformer.py  # Static/Learnable temperature scaling
│   └── tcd2vformer.py            # Adaptive context/query conditioned TCD2Vformer
├── layers/
│   ├── Date2Vec.py               # Trigonometric temporal harmonic embedding
│   ├── Fusion_Block.py           # Cross-temporal attention module
│   └── Revin.py                  # Reversible Instance Normalization (RevIN)
├── utils/
│   ├── data.py                   # Strict 60/20/20 chronological data splits
│   ├── metrics.py                # MSE, MAE, and Shannon Attention Entropy
│   └── reproducibility.py        # Seed locks & SHA-256 parameter checksums
├── demo/
│   └── index.html                # Interactive single-page Viva Defense Demo
├── results/
│   ├── FINAL_AUDIT_SUMMARY.md    # Complete classification audit
│   └── contribution_boundary.md  # Formal novelty delineation
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/ayushsalunkhe/TCD2Vformer.git
cd TCD2Vformer
pip install -r requirements.txt
```

### 2. Verify Horizon Independence ($\frac{\partial N}{\partial O} \equiv 0$)
```python
import torch
from models.tcd2vformer import TCD2Vformer

# Initialize model for 7-variable dataset (e.g., ETTh2)
model = TCD2Vformer(c_in=7, seq_len=96, temperature_mode='query_conditioned')

# Verify parameter counts across all horizons
param_counts = model.assert_horizon_independence(eval_horizons=[24, 48, 96, 192, 336, 720])
for O, count in param_counts.items():
    print(f"Horizon O = {O:3d} -> Trainable Parameters = {count}")
# Output: Exactly 44,326 for all horizons!
```

### 3. Run the Interactive Viva Demo
Simply open `demo/index.html` in any modern web browser to access:
- Interactive Attention Entropy vs Horizon visualizer
- Parameter Scaling Calculator ($\mathcal{O}(1)$ vs $\mathcal{O}(O)$)
- Viva Defense Cheat-Sheet & Examiner Question Bank
- Full audited benchmark matrices across all 4 datasets

---

## 🤝 Open Source Contribution

- **Upstream Project:** [D2Vformer (TeamofHaoWang/D2Vformer)](https://github.com/TeamofHaoWang/D2Vformer)
- **Upstream Bug Report:** [Issue #2: Output Linear(d_model, pred_len) creates horizon-dependent parameter growth](https://github.com/TeamofHaoWang/D2Vformer/issues/2)
- **Status:** Open upstream bug report filed by `@ayushsalunkhe` documenting the architectural inconsistency and proposing the FFN drop-in replacement.

---

## 📖 Citation

If you find this work or our empirical audit helpful in your research, please consider citing:

```bibtex
@misc{salunkhe2026tcd2vformer,
  author = {Ayush Harish Salunkhe},
  title = {TCD2Vformer: Temporal-Conditioned Date2Vecformer for Horizon-Independent Time-Series Forecasting},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ayushsalunkhe/TCD2Vformer}}
}
```
