# evaluate_multiseed.py
# Aggregate OOD RMSE across multiple seeds for each mix
# Produces: mean/std/CI RMSE tables + publication-quality figures
#
# Usage: python evaluate_multiseed.py
#        python evaluate_multiseed.py --seeds 42 43 44 45

from __future__ import annotations
import sys, os, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_kvlcc2_residual_v2 import ResidualMLP, TrainConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import argparse

CKPT_BASE = Path(__file__).parent / "checkpoints"
DATA_DIR = Path(__file__).parent.parent / "step2-a" / "datasets"
OUT_DIR = Path(__file__).parent / "results"

MIXES = ["mix_000", "mix_025", "mix_050", "mix_075", "mix_100"]
OPTEXC_PCTS = [0, 25, 50, 75, 100]
OOD_TESTS = ["ood_prbs", "ood_spiral", "ood_multistep", "ood_highfreq"]
ID_TEST = "ood_id_zigzag"


def load_model(mix: str, seed: int):
    """Load trained model checkpoint."""
    ckpt_path = CKPT_BASE / mix / f"seed{seed}" / "m00" / "stage1_model.pt"
    if not ckpt_path.exists():
        return None
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ResidualMLP(cfg.input_dim, cfg.hidden_dims, cfg.output_dim, 0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["x_mean"], ckpt["x_std"], ckpt["y_mean"], ckpt["y_std"]


@torch.no_grad()
def eval_rmse(model, x_mean, x_std, y_mean, y_std, X: np.ndarray, Y: np.ndarray) -> float:
    """Evaluate total RMSE (L2 norm of per-channel RMSE)."""
    X_t = torch.from_numpy(X.astype(np.float32))
    X_norm = (X_t - x_mean) / x_std
    pred_norm = model(X_norm)
    pred = (pred_norm * y_std + y_mean).numpy()
    Y_f = Y.astype(np.float32)
    mse_per_ch = np.mean((pred - Y_f) ** 2, axis=0)
    rmse_per_ch = np.sqrt(mse_per_ch)
    return float(np.sqrt(np.mean(rmse_per_ch ** 2)))  # total RMSE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    args = parser.parse_args()
    seeds = args.seeds
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all OOD test data
    ood_data = {}
    for name in OOD_TESTS + [ID_TEST]:
        npz = np.load(DATA_DIR / f"{name}.npz")
        ood_data[name] = (npz["X"], npz["Y"])
        print(f"Loaded {name}: {npz['X'].shape}")
    
    # Evaluate each mix × seed
    print("\n" + "=" * 80)
    print(f"Evaluating {len(MIXES)} mixes × {len(seeds)} seeds")
    print("=" * 80)
    
    # results[mix][test_name] = list of RMSE values (one per seed)
    all_results = {}
    
    for mix in MIXES:
        all_results[mix] = {"seeds_available": [], "rmse": {}}
        for test_name in OOD_TESTS + [ID_TEST]:
            all_results[mix]["rmse"][test_name] = []
        
        for seed in seeds:
            loaded = load_model(mix, seed)
            if loaded is None:
                print(f"  [SKIP] {mix}/seed{seed} (not found)")
                continue
            
            model, xm, xs, ym, ys = loaded
            all_results[mix]["seeds_available"].append(seed)
            
            for test_name in OOD_TESTS + [ID_TEST]:
                X_test, Y_test = ood_data[test_name]
                rmse = eval_rmse(model, xm, xs, ym, ys, X_test, Y_test)
                all_results[mix]["rmse"][test_name].append(rmse)
        
        # Compute stats
        n_seeds = len(all_results[mix]["seeds_available"])
        print(f"\n  {mix} ({n_seeds} seeds):")
        for test_name in OOD_TESTS:
            vals = all_results[mix]["rmse"][test_name]
            if vals:
                print(f"    {test_name}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")
    
    # Compute aggregated statistics
    summary = {}
    for mix in MIXES:
        summary[mix] = {"pct": OPTEXC_PCTS[MIXES.index(mix)]}
        for test_name in OOD_TESTS + [ID_TEST]:
            vals = all_results[mix]["rmse"][test_name]
            if len(vals) >= 2:
                summary[mix][test_name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "ci95_low": float(np.mean(vals) - 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))),
                    "ci95_high": float(np.mean(vals) + 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))),
                    "n_seeds": len(vals),
                    "values": vals,
                }
            elif len(vals) == 1:
                summary[mix][test_name] = {
                    "mean": vals[0],
                    "std": 0.0,
                    "n_seeds": 1,
                    "values": vals,
                }
    
    # Save
    results_path = OUT_DIR / "multiseed_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Saved: {results_path}")
    
    # Compute inter-seed vs inter-mix variance ratio
    print("\n" + "=" * 80)
    print("VARIANCE DECOMPOSITION")
    print("=" * 80)
    
    for test_name in OOD_TESTS:
        seed_vars = []
        mix_means = []
        for mix in MIXES:
            vals = summary[mix].get(test_name, {}).get("values", [])
            if len(vals) >= 2:
                seed_vars.append(np.var(vals, ddof=1))
            if vals:
                mix_means.append(np.mean(vals))
        
        avg_seed_var = np.mean(seed_vars) if seed_vars else 0
        mix_var = np.var(mix_means, ddof=1) if len(mix_means) >= 2 else 0
        ratio = mix_var / (avg_seed_var + 1e-30)
        
        print(f"  {test_name}:")
        print(f"    Inter-mix variance:  {mix_var:.2e}")
        print(f"    Avg inter-seed var:  {avg_seed_var:.2e}")
        print(f"    Ratio (mix/seed):    {ratio:.1f}×  {'✓ Signal dominates' if ratio > 5 else '⚠ Need more seeds'}")
    
    # ==================== PLOTS ====================
    plot_rmse_with_ci(summary)
    plot_variance_decomposition(summary)
    
    # Compute Spearman correlation with FIM (load from step2-a)
    compute_correlation_with_ci(summary)
    
    print(f"\n[OK] All plots saved to {OUT_DIR}")


def plot_rmse_with_ci(summary):
    """OOD RMSE vs OptExc% with confidence intervals from multi-seed."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 11,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    pcts = [summary[m]["pct"] for m in MIXES]
    colors = {"ood_prbs": "#2c3e50", "ood_spiral": "#e67e22",
              "ood_multistep": "#27ae60", "ood_highfreq": "#8e44ad"}
    labels = {"ood_prbs": "PRBS", "ood_spiral": "Spiral",
              "ood_multistep": "Multi-step", "ood_highfreq": "High-freq"}
    
    # (a) Raw RMSE with error bars
    ax = axes[0]
    for test_name in OOD_TESTS:
        means = [summary[m].get(test_name, {}).get("mean", 0) for m in MIXES]
        stds = [summary[m].get(test_name, {}).get("std", 0) for m in MIXES]
        ax.errorbar(pcts, means, yerr=stds, fmt="o-", color=colors[test_name],
                    label=labels[test_name], capsize=3, markersize=4, alpha=0.85)
    
    # InD baseline
    id_means = [summary[m].get(ID_TEST, {}).get("mean", 0) for m in MIXES]
    id_stds = [summary[m].get(ID_TEST, {}).get("std", 0) for m in MIXES]
    ax.errorbar(pcts, id_means, yerr=id_stds, fmt="s--", color="gray",
                label="InD (Zigzag)", capsize=3, markersize=4, alpha=0.7)
    
    ax.set_xlabel("OptExc Proportion (%)")
    ax.set_ylabel("RMSE")
    ax.set_title("(a) OOD RMSE with Multi-Seed CI")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pcts)
    
    # (b) OOD Gap with error bars
    ax = axes[1]
    for test_name in OOD_TESTS:
        gaps_mean = []
        gaps_std = []
        for mix in MIXES:
            ood_vals = summary[mix].get(test_name, {}).get("values", [])
            id_vals = summary[mix].get(ID_TEST, {}).get("values", [])
            if ood_vals and id_vals:
                # Compute gap for each seed, then stats
                gaps = [ood / id_v if id_v > 0 else 0 
                        for ood, id_v in zip(ood_vals, id_vals)]
                gaps_mean.append(np.mean(gaps))
                gaps_std.append(np.std(gaps, ddof=1) if len(gaps) > 1 else 0)
            else:
                gaps_mean.append(0)
                gaps_std.append(0)
        
        ax.errorbar(pcts, gaps_mean, yerr=gaps_std, fmt="o-", color=colors[test_name],
                    label=labels[test_name], capsize=3, markersize=4, alpha=0.85)
    
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("OptExc Proportion (%)")
    ax.set_ylabel("OOD Gap (RMSE_OOD / RMSE_InD)")
    ax.set_title("(b) OOD Gap with Multi-Seed CI")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pcts)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rmse_multiseed.png", dpi=200)
    plt.savefig(OUT_DIR / "rmse_multiseed.pdf")
    plt.close()
    print("[OK] Saved rmse_multiseed.png/pdf")


def plot_variance_decomposition(summary):
    """Stacked bar showing inter-mix vs inter-seed variance."""
    test_short = [t.replace("ood_", "") for t in OOD_TESTS]
    
    mix_vars = []
    seed_vars = []
    
    for test_name in OOD_TESTS:
        sv_list = []
        means = []
        for mix in MIXES:
            vals = summary[mix].get(test_name, {}).get("values", [])
            if len(vals) >= 2:
                sv_list.append(np.var(vals, ddof=1))
            if vals:
                means.append(np.mean(vals))
        mix_vars.append(np.var(means, ddof=1) if len(means) >= 2 else 0)
        seed_vars.append(np.mean(sv_list) if sv_list else 0)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(OOD_TESTS))
    w = 0.6
    
    # Normalize to percentage
    totals = [m + s for m, s in zip(mix_vars, seed_vars)]
    mix_pcts = [m / (t + 1e-30) * 100 for m, t in zip(mix_vars, totals)]
    seed_pcts = [s / (t + 1e-30) * 100 for s, t in zip(seed_vars, totals)]
    
    ax.bar(x, mix_pcts, w, label="Inter-mix (PE effect)", color="#3498db", alpha=0.85)
    ax.bar(x, seed_pcts, w, bottom=mix_pcts, label="Inter-seed (stochastic)", color="#e74c3c", alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(test_short)
    ax.set_ylabel("Variance Contribution (%)")
    ax.set_title("Variance Decomposition: PE Signal vs Random Seed")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    
    # Add ratio annotation
    for i, (mv, sv) in enumerate(zip(mix_vars, seed_vars)):
        ratio = mv / (sv + 1e-30)
        ax.text(i, 102, f"{ratio:.0f}×", ha="center", fontsize=8, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "variance_decomposition.png", dpi=200)
    plt.savefig(OUT_DIR / "variance_decomposition.pdf")
    plt.close()
    print("[OK] Saved variance_decomposition.png/pdf")


def compute_correlation_with_ci(summary):
    """Compute FIM-RMSE correlation using multi-seed mean RMSE."""
    # Load FIM results from step2-a
    fim_path = Path(__file__).parent.parent / "step2-a" / "results" / "fim_results.json"
    if not fim_path.exists():
        print("[SKIP] FIM results not found, skipping correlation")
        return
    
    with open(fim_path) as f:
        fim = json.load(f)
    
    log_dets = [fim[m]["log_det"] for m in MIXES]
    
    # Average OOD RMSE across all test sets (mean of means)
    ood_rmse_means = []
    for mix in MIXES:
        mix_means = []
        for t in OOD_TESTS:
            val = summary[mix].get(t, {}).get("mean", None)
            if val is not None:
                mix_means.append(val)
        ood_rmse_means.append(np.mean(mix_means) if mix_means else 0)
    
    # Spearman correlation
    rho, p_val = stats.spearmanr(log_dets, ood_rmse_means)
    r, p_r = stats.pearsonr(log_dets, ood_rmse_means)
    
    print("\n" + "=" * 80)
    print("FIM-RMSE CORRELATION (Multi-Seed)")
    print("=" * 80)
    print(f"  Spearman ρ = {rho:.3f}  (p = {p_val:.4f})")
    print(f"  Pearson  r = {r:.3f}  (p = {p_r:.4f})")
    
    # Save correlation summary
    corr_results = {
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "log_dets": log_dets,
        "ood_rmse_means": ood_rmse_means,
    }
    with open(OUT_DIR / "correlation_multiseed.json", "w") as f:
        json.dump(corr_results, f, indent=2)


if __name__ == "__main__":
    main()
