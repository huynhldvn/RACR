# Reliability-Aware Consistency Regularization for Semi-Supervised Medical Image Segmentation

[![Release](https://img.shields.io/badge/Release-v1.0.1-blue.svg)](https://github.com/huynhldvn/RACR/releases/tag/v1.0.1)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation and reproducible notebooks for the paper:

> **Reliability-Aware Consistency Regularization for Semi-Supervised Medical Image Segmentation**  
> Le Dinh Huynh, Truong Cong Doan, and Phan Duy Hung  
> *Manuscript under review; venue information to be updated.*  
> Official Repository: [https://github.com/huynhldvn/RACR](https://github.com/huynhldvn/RACR) (Release v1.0.1)

---

## 📌 Overview

Semi-supervised consistency regularization methods (e.g., Mean Teacher) enforce prediction agreement under stochastic perturbations. However, conventional models rely on rigid, monotonically increasing ramp-up schedules that overlook instantaneous teacher reliability—forcing the student network to fit uncalibrated, noisy pseudo-labels during early training iterations.

**Reliability-Aware Consistency Regularization (RACR)** dynamically modulates consistency supervision according to instantaneous teacher confidence:
* **Zero Parameter Overhead:** Derives batch reliability directly from the single existing EMA teacher forward pass via lightweight tensor reduction ($\max_c p_t$), introducing $<0.008\,\text{ms}$ latency ($<0.8\%$ computational overhead).
* **Dynamic Supervisory Pacing:** Replaces rigid schedules with an empirical pacing signal:
  $$\lambda_{\text{adapt}}(t) = (\alpha_0 + (1 - \alpha_0)R_t) \cdot \lambda_{\text{base}}(t)$$
  where $\alpha_0 = 0.5$ preserves baseline optimization stability. As $R_t \in (0, 1]$ aggregates instantaneous maximum softmax probabilities across the unlabeled batch, the effective supervisory pacing multiplier $(\alpha_0 + (1 - \alpha_0)R_t)$ smoothly modulates consistency loss within $[\alpha_0, 1.0]$ in proportion to instantaneous teacher prediction maturity, providing robust, self-paced regularization without requiring manual schedule tuning.
* **Variance & Boundary Stabilization:** Suppresses confirmation bias and stabilizes boundary localization across independent runs without requiring stochastic Monte Carlo sampling passes.

![Framework](figures/framework.png)

---

## 📊 Overview of Empirical Findings

The framework was evaluated on the **ACDC 2017 benchmark** under the standardized **10% labeled data regime** (7 labeled, 63 unlabeled patient scans) across both Mean Teacher and UAMT consistency paradigms:
* **Enhanced Optimization Stability:** Incorporating dynamic reliability pacing stabilizes training trajectories and substantially suppresses cross-run variance across diverse random initializations.
* **Improved Boundary & Overlap Fidelity:** Pacing consistency supervision dynamically against instantaneous teacher confidence mitigates early error accumulation, improving anatomical contour delineation and regional overlap.
* **Efficient Single-Pass Regularization:** Achieves robust consistency pacing in a single forward pass without requiring expensive multi-pass Monte Carlo sampling overhead (<1% runtime latency).

> For detailed quantitative benchmark comparisons, multi-seed ablation tables, and per-run experimental records, please refer to Section 3 and the Supplementary Material of our manuscript.

---

## 📁 Repository Structure

```text
.
├── README.md                          # Documentation and reproduction guide
├── LICENSE                            # MIT License
├── figures/
│   └── framework.png                  # Architectural diagram of RACR
├── codebase/                          # Reference baseline implementations
│   ├── MT/
│   │   ├── MT-10000eps.ipynb          # Original baseline MT notebook
│   │   └── README.md
│   └── UAMT/
│       └── README.md
├── reproduce/                         # Reproduction verification notebooks
│   ├── MT.ipynb                       # Baseline MT verification
│   └── UAMT.ipynb                     # Baseline UAMT verification
└── our/                               # Proposed RACR implementation
    ├── Reliability_MT_soft.ipynb      # RACR Soft-Weighting on Mean Teacher (without mask)
    ├── Reliability_MT_mask.ipynb      # RACR Class-Aware Masking on Mean Teacher (with mask)
    └── Reliability_UAMT.ipynb         # RACR integrated with UAMT
```

---

## 🛠️ Environment and Requirements

The experiments were conducted in Python 3.10 and PyTorch 2.1.2 on dual NVIDIA Tesla T4 / RTX GPUs.

### Dependencies
Install the required packages via `pip`:

```bash
pip install torch torchvision numpy scipy h5py medpy SimpleITK tqdm
```

Recommended versions:
* `torch >= 2.0.0`
* `torchvision >= 0.15.0`
* `h5py >= 3.8.0`
* `medpy >= 0.4.0`
* `SimpleITK >= 2.2.0`
* `numpy >= 1.24.0`

---

## 🗂️ Dataset Preparation

Experiments are benchmarked on the **Automated Cardiac Diagnosis Challenge (ACDC 2017)** dataset:
* **Official Website:** [https://www.creatis.insa-lyon.fr/Challenge/acdc/](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
* **Preprocessing Protocol:** We follow the standardized patient partitioning and slice preprocessing established by [SSL4MIS](https://github.com/HiLab-git/SSL4MIS/blob/master/data/ACDC/README.md):
  * **Training Split:** 70 patient volumes (10% regime: 7 labeled, 63 unlabeled patient scans).
  * **Validation Split:** 10 patient volumes.
  * **Testing Split:** 20 patient subjects (40 3D test volumes across ED and ES cardiac phases).
  * All 2D cine-MRI slices are standardized to $256 \times 256$ pixels with zero-mean, unit-variance intensity normalization.

---

## 🚀 Running the Notebooks

The primary implementation is organized into clean, self-contained interactive notebooks inside the `our/` folder:

1. **RACR Soft-Weighting on Mean Teacher (without mask):**
   * Open `our/Reliability_MT_soft.ipynb` in Google Colab, Kaggle, or your local Jupyter environment.
   * Implements the soft reliability pacing formulation without spatial truncation, providing smooth, continuous gradient flow across all spatial regions.

2. **RACR with Class-Aware Masking on Mean Teacher (with mask):**
   * Open `our/Reliability_MT_mask.ipynb`.
   * Integrates class-aware confidence thresholds ($\tau_{\text{bg}} = 0.90, \tau_{\text{fg}} = 0.60$) and spatial downweighting ($w_{\text{bg}} = 0.20, w_{\text{fg}} = 1.0$) to prevent dominant background saturation and filter ambiguous boundary noise.

3. **RACR on Uncertainty-Aware Mean Teacher (UAMT):**
   * Open `our/Reliability_UAMT.ipynb`.
   * Implements single-pass reliability pacing on top of active MC-dropout uncertainty estimation, delivering consistent boundary error reduction without introducing multi-pass latency.

### Key Hyperparameters:
* **Total Iterations:** 10,000 steps
* **Batch Size:** 24 slices per iteration (12 labeled and 12 unlabeled slices)
* **Optimizer:** SGD ($\text{lr} = 0.01$, momentum $= 0.9$, weight decay $= 0.0001$)
* **EMA Decay:** $\alpha_{\text{ema}} = 0.99$
* **Reliability Factor:** $\alpha_0 = 0.5$ (multiplier dynamic range in $[0.5, 1.0]$)
* **Max Consistency Weight:** $\lambda_{\text{base,max}} = 0.1$ (Mean Teacher), $\lambda_{\text{base,max}} = 1.0$ (UAMT)

---

## 📜 Citation

If you find this repository or methodology useful for your research, please cite our manuscript:

```bibtex
@article{huynh2026reliability,
  title   = {Reliability-Aware Consistency Regularization for Semi-Supervised Medical Image Segmentation},
  author  = {Huynh, Le Dinh and Doan, Truong Cong and Hung, Phan Duy},
  journal = {Under Review},
  year    = {2026}
}
```

---

## 🙏 Acknowledgments

* We thank Luo et al. for the [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) framework and standardized benchmark data split protocol.
* We acknowledge the organizers of the **ACDC 2017 Challenge** for providing the cardiac MRI dataset.
* We also thank Huynh et al. (IUKM 2025) for their reference baseline codebase.
