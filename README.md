# Excitation Design Interpretability for Ship Maneuvering

A toolkit for analyzing **how training signal (excitation) design affects neural network learning quality** in physics-guided ship maneuvering models. Provides multi-perspective interpretability analysis including NTK kernel analysis, manifold collapse detection, shortcut diagnosis, and Fisher Information sufficiency assessment.

## Background

When training neural networks to learn residual dynamics from a physics simulator, the choice of excitation signal (training maneuver) critically affects generalization. This toolkit provides six analysis methods to understand *why* certain excitation designs lead to better out-of-distribution (OOD) performance.

## Analysis Methods

| Script | Method | Key Question |
|--------|--------|-------------|
| `compute_empirical_kernel.py` | Empirical Jacobian Kernel | How does the learned representation structure change with excitation design? |
| `compute_effective_rank.py` | Effective Rank (eRank) | Does poor excitation cause manifold collapse in feature space? |
| `compute_saliency.py` | Gradient Saliency + Permutation Importance | Does the network learn physics-relevant features or shortcuts? |
| `compute_fi_sufficiency.py` | Fisher Information Sufficiency Ratio | How much excitation is "enough" for parameter identifiability? |
| `train_multiseed.py` | Multi-Seed Training | Are findings reproducible across random initializations? |
| `evaluate_multiseed.py` | Multi-Seed Evaluation with CI | Statistical significance of OOD performance differences |

## Repository Structure

```
├── analysis/
│   ├── compute_empirical_kernel.py   # K(x,x') = J(x) @ J(x')^T spectral analysis
│   ├── compute_effective_rank.py     # eRank = exp(H(σ)) manifold collapse
│   ├── compute_saliency.py           # Gradient saliency + permutation importance
│   ├── compute_fi_sufficiency.py     # R_FI = det(F_train) / det(F_ideal)
│   ├── train_multiseed.py            # Parallel multi-seed training launcher
│   └── evaluate_multiseed.py         # OOD RMSE aggregation with bootstrap CI
├── README.md
├── LICENSE
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/xzz2022hust-oss/excitation-design-interpretability.git
cd excitation-design-interpretability
pip install -r requirements.txt
```

## Quick Start

### 1. Empirical Jacobian Kernel

Computes `K(x, x') = J(x) @ J(x')^T` where `J` is the parameter Jacobian. Analyzes kernel eigenvalue spectrum to reveal representation quality.

```bash
python analysis/compute_empirical_kernel.py
```

**Key metrics**: Kernel effective rank, condition number, spectral decay rate

### 2. Effective Rank (Manifold Collapse Detection)

Quantifies the effective dimensionality of the feature/output manifold using SVD:

`eRank(X) = exp(H(σ))` where `H(σ) = -Σ pᵢ log(pᵢ)` (Roy & Vetterli 2007)

```bash
python analysis/compute_effective_rank.py
```

**Key insight**: Low eRank → the network "sees" a low-dimensional projection of the data, missing important dynamics.

### 3. Shortcut Learning Diagnosis

Two complementary methods to detect if the network relies on physics-relevant or shortcut features:

```bash
python analysis/compute_saliency.py
```

- **Gradient Saliency**: `E[|∂ŷ/∂xᵢ|]` — which input dimensions drive predictions?
- **Permutation Importance**: `ΔLoss` when shuffling feature `xᵢ` — which features are functionally important?

### 4. Fisher Information Sufficiency

`R_FI = det(F_train) / det(F_ideal)` with automatic elbow detection:

```bash
python analysis/compute_fi_sufficiency.py
```

**Key insight**: Identifies the "minimum sufficient" excitation level for identifiability.

### 5. Multi-Seed Reproducibility

```bash
# Train with multiple seeds
python analysis/train_multiseed.py --seeds 43 44 45

# Evaluate with confidence intervals
python analysis/evaluate_multiseed.py --seeds 42 43 44 45
```

## Mathematical Background

### Empirical Kernel
```
J(x) = flatten(∂f(x; θ) / ∂θ)     # Parameter Jacobian
K = J @ J^T                         # Empirical kernel (NTK approximation)
Eigenvalues of K → representation quality
```

### Effective Rank
```
X = U Σ V^T           # SVD
pᵢ = σᵢ / Σⱼ σⱼ      # Normalized singular values
eRank = exp(-Σ pᵢ log pᵢ)   # ∈ [1, min(m,n)]
```

### FI Sufficiency Ratio
```
F = (1/T) Σ_t J_t^T R⁻¹ J_t    # Fisher Information Matrix
R_FI = det(F_train) / det(F_ideal)  # ∈ [0, 1]
```

## References

1. **Roy, O., & Vetterli, M.** (2007). The effective rank: A measure of effective dimensionality. *European Signal Processing Conference (EUSIPCO)*.

2. **Jacot, A., Gabriel, F., & Hongler, C.** (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS*.

3. **Yasukawa, H., & Yoshimura, Y.** (2015). Introduction of MMG standard method for ship maneuvering predictions. *J. Marine Science and Technology*, 20(1), 37–52.

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{excitation_interpretability,
  title  = {Excitation Design Interpretability for Ship Maneuvering},
  url    = {https://github.com/xzz2022hust-oss/excitation-design-interpretability},
  year   = {2026}
}
```
