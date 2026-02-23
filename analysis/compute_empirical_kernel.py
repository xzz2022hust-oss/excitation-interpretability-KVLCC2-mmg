# compute_empirical_kernel.py
# Strategy 1 (Light Version): Empirical Jacobian Kernel Analysis
#
# Computes the empirical tangent feature kernel K(x,x') = J(x) @ J(x')^T
# where J(x) = ∂f_θ(x)/∂θ is the Jacobian of the network output w.r.t. parameters.
#
# IMPORTANT: This uses the TRAINED model's Jacobian, not the initial one.
# Correct terminology: "empirical Jacobian kernel" or "empirical tangent feature kernel"
# (NOT "Neural Tangent Kernel" which is defined at initialization)
#
# Reports: log-det(K + λI) and cond(K + λI) vs OptExc%

from __future__ import annotations
import sys, os, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.func import functional_call, vmap, jacrev

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_kvlcc2_residual_v2 import ResidualMLP, TrainConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CKPT_BASE = Path(__file__).parent.parent / "step2-a" / "checkpoints"
DATA_DIR = Path(__file__).parent.parent / "step2-a" / "datasets"
OUT_DIR = Path(__file__).parent / "results"

MIXES = ["mix_000", "mix_025", "mix_050", "mix_075", "mix_100"]
OPTEXC_PCTS = [0, 25, 50, 75, 100]
SEED = 42

N_KERNEL = 256
REG_LAMBDA = 1e-6


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


def compute_jacobian_features(model, X_norm):
    """
    Compute Jacobian features using torch.func for correctness.
    
    J(x_i) = flatten(∂f(x_i)/∂θ)  for each sample x_i
    
    Returns: J of shape (N, output_dim * n_params)
    """
    n_samples = X_norm.shape[0]
    
    # Get parameter dictionary
    params = {name: p.detach() for name, p in model.named_parameters()}
    n_params = sum(p.numel() for p in params.values())
    output_dim = 3
    
    print(f"    Computing Jacobian features: {n_samples} samples × {n_params} params...")
    
    def fnet_single(params, x):
        """Forward pass with functional params for a single sample."""
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)
    
    # Compute per-sample Jacobian: ∂f(x_i)/∂θ
    # jac_func maps (params, x_single) -> dict of parameter Jacobians
    jac_func = jacrev(fnet_single, argnums=0)
    
    all_jacs = []
    batch_size = 16
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        for i in range(start, end):
            x_i = X_norm[i]
            jac_dict = jac_func(params, x_i)
            
            # Flatten jacobian: for each output j, concat all param grads
            jac_flat = []
            for name in sorted(params.keys()):
                # jac_dict[name] has shape (output_dim, *param_shape)
                j = jac_dict[name]  # (3, *param_shape)
                jac_flat.append(j.reshape(output_dim, -1))  # (3, n_param_i)
            
            # Concatenate across params: (3, n_params_total)
            jac_full = torch.cat(jac_flat, dim=1)
            # Flatten to (3 * n_params_total,)
            all_jacs.append(jac_full.flatten().detach())
        
        if (start // batch_size) % 4 == 0:
            print(f"      Progress: {min(end, n_samples)}/{n_samples}")
    
    J = torch.stack(all_jacs).numpy().astype(np.float64)
    print(f"    Jacobian matrix shape: {J.shape}")
    print(f"    J norm range: [{np.min(np.abs(J)):.2e}, {np.max(np.abs(J)):.2e}]")
    print(f"    J Frobenius norm: {np.linalg.norm(J, 'fro'):.6e}")
    return J


def compute_kernel_metrics(J, reg_lambda=REG_LAMBDA):
    """
    Compute kernel K = J @ J^T metrics via SVD of J.
    
    Instead of forming the N×N kernel K = J@J^T (which can overflow),
    compute SVD of J: J = U Σ V^T, then eigenvalues of K are σ².
    """
    N = J.shape[0]
    
    # SVD of J: eigenvalues of K = J@J^T are the squared singular values
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    # s has shape (min(N, D),) — squared gives eigenvalues of K
    eigvals_K = s ** 2  # eigenvalues of K = J@J^T
    
    print(f"    SVD singular values range: [{s.min():.2e}, {s.max():.2e}]")
    print(f"    K eigenvalues range: [{eigvals_K.min():.2e}, {eigvals_K.max():.2e}]")
    
    # Add regularization: eigenvalues of K + λI = σ² + λ
    eigvals_reg = eigvals_K + reg_lambda
    # Pad with reg_lambda for any zero eigenvalues (if N > D)
    if len(eigvals_reg) < N:
        eigvals_reg = np.concatenate([eigvals_reg, np.full(N - len(eigvals_reg), reg_lambda)])
    
    eigvals_reg = np.sort(eigvals_reg)[::-1]  # descending
    
    # Metrics
    log_det = np.sum(np.log(np.maximum(eigvals_reg, 1e-300)))
    cond_number = eigvals_reg[0] / (eigvals_reg[-1] + 1e-300)
    trace = np.sum(eigvals_reg)
    
    # Effective rank of kernel
    p = eigvals_reg / (eigvals_reg.sum() + 1e-300)
    erank = np.exp(-np.sum(p * np.log(p + 1e-300)))
    
    return {
        "log_det": float(log_det),
        "cond_number": float(cond_number),
        "log_cond": float(np.log10(cond_number + 1e-300)),
        "trace": float(trace),
        "erank": float(erank),
        "top_eigenvalues": eigvals_reg[:20].tolist(),
        "top_singular_values": s[:20].tolist(),
        "n_samples": int(N),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.RandomState(42)
    
    print("=" * 70)
    print("EMPIRICAL JACOBIAN KERNEL ANALYSIS (torch.func)")
    print(f"N_kernel = {N_KERNEL}, λ_reg = {REG_LAMBDA}")
    print("=" * 70)
    
    results = {}
    
    for mix, pct in zip(MIXES, OPTEXC_PCTS):
        print(f"\n--- {mix} ({pct}% OptExc) ---")
        
        model, xm, xs, ym, ys = load_model(mix)
        
        # Load and subsample training data
        npz = np.load(DATA_DIR / f"train_{mix}.npz")
        X_all = npz["X"].astype(np.float32)
        
        idx = rng.choice(len(X_all), N_KERNEL, replace=False)
        X_sub = X_all[idx]
        
        # Normalize
        X_norm = torch.tensor((X_sub - xm.numpy()) / xs.numpy(), dtype=torch.float32)
        
        # Compute Jacobian features
        J = compute_jacobian_features(model, X_norm)
        
        # Compute kernel metrics
        metrics = compute_kernel_metrics(J)
        metrics["pct"] = pct
        results[mix] = metrics
        
        print(f"  log-det(K+λI)  = {metrics['log_det']:.2f}")
        print(f"  cond(K+λI)     = {metrics['cond_number']:.2e}")
        print(f"  log₁₀(κ)       = {metrics['log_cond']:.2f}")
        print(f"  eRank(K)       = {metrics['erank']:.2f}")
    
    # Save
    with open(OUT_DIR / "empirical_kernel_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved empirical_kernel_results.json")
    
    # Correlations
    print("\n" + "=" * 70)
    print("CORRELATION: Kernel Metrics vs OOD RMSE")
    print("=" * 70)
    
    eval_path = Path(__file__).parent.parent / "step2-a" / "results" / "eval_results.json"
    with open(eval_path) as f:
        eval_data = json.load(f)
    
    ood_tests = ["ood_prbs", "ood_spiral", "ood_multistep", "ood_highfreq"]
    avg_ood_rmse = []
    for mix in MIXES:
        rmses = [eval_data["results"][mix][t]["rmse_total"] for t in ood_tests]
        avg_ood_rmse.append(np.mean(rmses))
    
    for metric_name in ["log_det", "cond_number", "erank", "trace"]:
        vals = [results[m][metric_name] for m in MIXES]
        if len(set(vals)) > 1:  # skip if all values are the same
            rho, p = stats.spearmanr(vals, avg_ood_rmse)
            r_p, p_p = stats.pearsonr(vals, avg_ood_rmse)
            print(f"  {metric_name:>15}: Spearman ρ = {rho:+.3f} (p={p:.3f}), Pearson r = {r_p:+.3f}")
        else:
            print(f"  {metric_name:>15}: [SKIP] all values identical")
    
    # FIM correlation
    fim_path = Path(__file__).parent.parent / "step2-a" / "results" / "fim_results.json"
    if fim_path.exists():
        with open(fim_path) as f:
            fim = json.load(f)
        fim_log_dets = [fim[m]["log_det"] for m in MIXES]
        kernel_log_dets = [results[m]["log_det"] for m in MIXES]
        if len(set(kernel_log_dets)) > 1:
            rho, p = stats.spearmanr(fim_log_dets, kernel_log_dets)
            print(f"\n  FIM vs Kernel log-det: ρ = {rho:+.3f} (p={p:.3f})")
    
    # Plot
    plot_kernel_analysis(results, avg_ood_rmse)
    print(f"\n[OK] All plots saved to {OUT_DIR}")


def plot_kernel_analysis(results, avg_ood_rmse):
    """Kernel metrics plots."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 11,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pcts = OPTEXC_PCTS
    
    # (a) log-det(K) vs OptExc%
    ax = axes[0]
    log_dets = [results[m]["log_det"] for m in MIXES]
    ax.plot(pcts, log_dets, "o-", color="#3498db", markersize=7, linewidth=2)
    ax.set_xlabel("OptExc Proportion (%)")
    ax.set_ylabel(r"$\log\det(\mathbf{K} + \lambda\mathbf{I})$")
    ax.set_title("(a) Empirical Kernel Log-Determinant")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pcts)
    
    # (b) Effective rank of K vs OptExc%
    ax = axes[1]
    eranks = [results[m]["erank"] for m in MIXES]
    ax.plot(pcts, eranks, "s-", color="#e74c3c", markersize=7, linewidth=2)
    ax.set_xlabel("OptExc Proportion (%)")
    ax.set_ylabel("Effective Rank of K")
    ax.set_title("(b) Kernel Effective Rank")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pcts)
    
    # (c) log-det(K) vs OOD RMSE scatter
    ax = axes[2]
    sc = ax.scatter(log_dets, [r * 1e4 for r in avg_ood_rmse], c=pcts, cmap="RdYlGn",
                    s=100, edgecolors="black", zorder=5)
    for i, mix in enumerate(MIXES):
        ax.annotate(f"{pcts[i]}%", (log_dets[i], avg_ood_rmse[i] * 1e4),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    
    vals = [results[m]["log_det"] for m in MIXES]
    if len(set(vals)) > 1:
        rho, _ = stats.spearmanr(vals, avg_ood_rmse)
        title_suffix = f" (ρ={rho:.3f})"
    else:
        title_suffix = " (insufficient variation)"
    
    ax.set_xlabel(r"$\log\det(\mathbf{K} + \lambda\mathbf{I})$")
    ax.set_ylabel("Avg OOD RMSE (×10⁻⁴)")
    ax.set_title(f"(c) Kernel vs OOD Error{title_suffix}")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="OptExc %")
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "empirical_kernel.png", dpi=200)
    plt.savefig(OUT_DIR / "empirical_kernel.pdf")
    plt.close()
    print("[OK] Saved empirical_kernel.png/pdf")


if __name__ == "__main__":
    main()
