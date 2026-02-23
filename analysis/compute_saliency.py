# compute_saliency.py
# Strategy 4: Shortcut Learning Diagnosis
#
# Two complementary methods:
#   1. Gradient Saliency: E[|∂ŷ/∂x_i|] per input dimension (fast, visual)
#   2. Permutation Importance: ΔLoss when shuffling x_i (robust, quantitative)
#
# Diagnoses whether mix_000 model uses shortcuts (δ only) vs
# mix_100 model learning full physics (u, v, r, δ coupling)

from __future__ import annotations
import sys, os, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_kvlcc2_residual_v2 import ResidualMLP, TrainConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use step2-a checkpoints (seed42) and datasets
CKPT_BASE = Path(__file__).parent.parent / "step2-a" / "checkpoints"
DATA_DIR = Path(__file__).parent.parent / "step2-a" / "datasets"
OUT_DIR = Path(__file__).parent / "results"

MIXES = ["mix_000", "mix_025", "mix_050", "mix_075", "mix_100"]
OPTEXC_PCTS = [0, 25, 50, 75, 100]
SEED = 42

# Feature names (SOP v2.00 input format)
FEATURE_NAMES = ["u", "v", "r", "δ_stu", "n", "δ̇_stu", "δ_ref", "δ̇_ref"]
# Short names for plotting
FEATURE_SHORT = ["u", "v", "r", "δ", "n", "δ̇", "δ_ref", "δ̇_ref"]


def load_model(mix: str, seed: int = SEED):
    """Load trained model checkpoint."""
    ckpt_path = CKPT_BASE / mix / f"seed{seed}" / "m00" / "stage1_model.pt"
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ResidualMLP(cfg.input_dim, cfg.hidden_dims, cfg.output_dim, 0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["x_mean"], ckpt["x_std"], ckpt["y_mean"], ckpt["y_std"]


def compute_gradient_saliency(model, x_mean, x_std, y_mean, y_std, X_ood, n_samples=2000):
    """
    Compute average absolute gradient saliency per input dimension.
    
    Saliency_i = E_{x ~ D_OOD} [|∂ŷ/∂x_i|]
    
    Returns: (n_input,) array of saliency values
    """
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_ood), min(n_samples, len(X_ood)), replace=False)
    X_sub = X_ood[idx]
    
    X_t = torch.tensor(X_sub, dtype=torch.float32, requires_grad=True)
    X_norm = (X_t - x_mean) / x_std
    pred_norm = model(X_norm)
    pred = pred_norm * y_std + y_mean
    
    # Sum over all outputs and samples to get gradients
    pred.sum().backward()
    
    # Average absolute gradient per input dimension
    saliency = X_t.grad.abs().mean(dim=0).detach().numpy()
    
    # Normalize to sum to 1 for interpretability
    saliency_normalized = saliency / (saliency.sum() + 1e-30)
    
    return saliency, saliency_normalized


def compute_permutation_importance(model, x_mean, x_std, y_mean, y_std, 
                                     X_ood, Y_ood, n_repeats=10):
    """
    Compute permutation feature importance.
    
    For each feature i:
      1. Shuffle column i of X_ood
      2. Compute loss increase ΔLoss = Loss(shuffled) - Loss(original)
      3. Repeat n_repeats times, report mean and std
    
    Returns: dict with mean and std importance per feature
    """
    # Baseline loss
    with torch.no_grad():
        X_t = torch.tensor(X_ood, dtype=torch.float32)
        X_norm = (X_t - x_mean) / x_std
        pred_norm = model(X_norm)
        pred = (pred_norm * y_std + y_mean).numpy()
        baseline_mse = np.mean((pred - Y_ood.astype(np.float32)) ** 2)
    
    importances = {}
    rng = np.random.RandomState(42)
    
    for feat_idx in range(X_ood.shape[1]):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_ood.copy()
            X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
            
            with torch.no_grad():
                X_t = torch.tensor(X_perm, dtype=torch.float32)
                X_norm = (X_t - x_mean) / x_std
                pred_norm = model(X_norm)
                pred = (pred_norm * y_std + y_mean).numpy()
                perm_mse = np.mean((pred - Y_ood.astype(np.float32)) ** 2)
            
            deltas.append(perm_mse - baseline_mse)
        
        importances[feat_idx] = {
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
            "relative": float(np.mean(deltas) / (baseline_mse + 1e-30)),
        }
    
    return importances, baseline_mse


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load OOD test data (use all OOD sets combined for robust estimate)
    ood_names = ["ood_prbs", "ood_spiral", "ood_multistep", "ood_highfreq"]
    X_ood_all, Y_ood_all = [], []
    for name in ood_names:
        npz = np.load(DATA_DIR / f"{name}.npz")
        X_ood_all.append(npz["X"])
        Y_ood_all.append(npz["Y"])
    X_ood = np.concatenate(X_ood_all, axis=0)
    Y_ood = np.concatenate(Y_ood_all, axis=0)
    print(f"Combined OOD data: {X_ood.shape}")
    
    results = {}
    
    print("=" * 70)
    print("SHORTCUT LEARNING DIAGNOSIS")
    print("=" * 70)
    
    for mix, pct in zip(MIXES, OPTEXC_PCTS):
        print(f"\n--- {mix} ({pct}% OptExc) ---")
        
        model, xm, xs, ym, ys = load_model(mix)
        
        # 1. Gradient saliency
        sal_raw, sal_norm = compute_gradient_saliency(model, xm, xs, ym, ys, X_ood)
        print(f"  Gradient saliency (normalized):")
        for i, (name, val) in enumerate(zip(FEATURE_SHORT, sal_norm)):
            bar = "█" * int(val * 50)
            print(f"    {name:>6}: {val:.3f} {bar}")
        
        # 2. Permutation importance
        perm_imp, base_mse = compute_permutation_importance(
            model, xm, xs, ym, ys, X_ood, Y_ood, n_repeats=10)
        print(f"  Permutation importance (relative ΔLoss/Loss):")
        for i, name in enumerate(FEATURE_SHORT):
            val = perm_imp[i]["relative"]
            bar = "█" * int(min(val, 1.0) * 30)
            print(f"    {name:>6}: {val:.4f} {bar}")
        
        # Compute "shortcut score": fraction of saliency on δ-related features
        delta_indices = [3, 5, 6, 7]  # δ_stu, δ̇_stu, δ_ref, δ̇_ref
        physics_indices = [0, 1, 2]   # u, v, r
        
        shortcut_score_sal = sum(sal_norm[i] for i in delta_indices)
        physics_score_sal = sum(sal_norm[i] for i in physics_indices)
        
        shortcut_score_perm = sum(perm_imp[i]["relative"] for i in delta_indices)
        physics_score_perm = sum(perm_imp[i]["relative"] for i in physics_indices)
        
        results[mix] = {
            "pct": pct,
            "saliency_raw": sal_raw.tolist(),
            "saliency_normalized": sal_norm.tolist(),
            "permutation_importance": {
                FEATURE_SHORT[i]: perm_imp[i] for i in range(len(FEATURE_SHORT))
            },
            "baseline_mse": float(base_mse),
            "shortcut_score_saliency": float(shortcut_score_sal),
            "physics_score_saliency": float(physics_score_sal),
            "shortcut_score_permutation": float(shortcut_score_perm),
            "physics_score_permutation": float(physics_score_perm),
        }
        
        print(f"  → Shortcut Score (saliency): {shortcut_score_sal:.3f} (δ-features)")
        print(f"  → Physics Score (saliency):  {physics_score_sal:.3f} (u,v,r)")
    
    # Save
    with open(OUT_DIR / "saliency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved saliency_results.json")
    
    # ==================== PLOTS ====================
    plot_saliency_heatmap(results)
    plot_permutation_bars(results)
    plot_shortcut_evolution(results)
    
    print(f"\n[OK] All plots saved to {OUT_DIR}")


def plot_saliency_heatmap(results):
    """Heatmap: saliency per feature × training mix."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    n_mixes = len(MIXES)
    n_features = len(FEATURE_SHORT)
    
    # Build matrix
    matrix = np.zeros((n_mixes, n_features))
    for i, mix in enumerate(MIXES):
        matrix[i] = results[mix]["saliency_normalized"]
    
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(FEATURE_SHORT, fontsize=9)
    ax.set_yticks(range(n_mixes))
    ax.set_yticklabels([f"{OPTEXC_PCTS[i]}% OptExc" for i in range(n_mixes)], fontsize=9)
    
    # Add text annotations
    for i in range(n_mixes):
        for j in range(n_features):
            val = matrix[i, j]
            color = "white" if val > 0.2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    
    ax.set_title("Gradient Saliency: Feature Attribution per Training Mix")
    plt.colorbar(im, ax=ax, label="Normalized Saliency", shrink=0.8)
    
    # Mark physics vs shortcut regions
    ax.axvline(x=2.5, color="white", linewidth=2, linestyle="--")
    ax.text(1, -0.7, "Physics (u,v,r)", ha="center", fontsize=8, fontweight="bold", color="#27ae60")
    ax.text(5, -0.7, "Control (δ, n, ...)", ha="center", fontsize=8, fontweight="bold", color="#e74c3c")
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "saliency_heatmap.png", dpi=200)
    plt.savefig(OUT_DIR / "saliency_heatmap.pdf")
    plt.close()
    print("[OK] Saved saliency_heatmap.png/pdf")


def plot_permutation_bars(results):
    """Grouped bar: permutation importance per feature for mix_000 vs mix_100."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for panel_idx, mix in enumerate(["mix_000", "mix_100"]):
        ax = axes[panel_idx]
        pct = results[mix]["pct"]
        
        means = [results[mix]["permutation_importance"][f]["mean"] for f in FEATURE_SHORT]
        stds = [results[mix]["permutation_importance"][f]["std"] for f in FEATURE_SHORT]
        
        # Color by physics vs control
        colors = ["#27ae60"] * 3 + ["#e74c3c"] * 2 + ["#e74c3c"] * 3
        
        x = np.arange(len(FEATURE_SHORT))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(FEATURE_SHORT, fontsize=9)
        ax.set_ylabel("Permutation Importance (ΔMSE)")
        ax.set_title(f"({'a' if panel_idx==0 else 'b'}) {mix} ({pct}% OptExc)")
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "permutation_importance.png", dpi=200)
    plt.savefig(OUT_DIR / "permutation_importance.pdf")
    plt.close()
    print("[OK] Saved permutation_importance.png/pdf")


def plot_shortcut_evolution(results):
    """Line plot: shortcut vs physics score evolution across mixes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    pcts = OPTEXC_PCTS
    
    for panel_idx, (method, title) in enumerate([
        ("saliency", "Gradient Saliency"),
        ("permutation", "Permutation Importance"),
    ]):
        ax = axes[panel_idx]
        
        shortcut = [results[m][f"shortcut_score_{method}"] for m in MIXES]
        physics = [results[m][f"physics_score_{method}"] for m in MIXES]
        
        ax.plot(pcts, shortcut, "o-", color="#e74c3c", label="Control features (shortcut)", 
                markersize=6, linewidth=2)
        ax.plot(pcts, physics, "s-", color="#27ae60", label="State features (physics)",
                markersize=6, linewidth=2)
        
        # Fill between to show transition
        ax.fill_between(pcts, shortcut, physics, alpha=0.1, color="gray")
        
        ax.set_xlabel("OptExc Proportion (%)")
        ax.set_ylabel("Aggregate Attribution Score")
        ax.set_title(f"({chr(97+panel_idx)}) {title}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pcts)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shortcut_evolution.png", dpi=200)
    plt.savefig(OUT_DIR / "shortcut_evolution.pdf")
    plt.close()
    print("[OK] Saved shortcut_evolution.png/pdf")


if __name__ == "__main__":
    main()
