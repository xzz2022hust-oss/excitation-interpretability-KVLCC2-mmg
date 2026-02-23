# compute_effective_rank.py
# Strategy 2: Quantify manifold collapse via Effective Rank
#
# Computes eRank for each training mix on:
#   1. Raw state-control features X (NN input space)
#   2. Normalized features X_norm
#   3. Subspace per dynamic channel (u,v,r) via output-weighted input
#
# Reference: Roy & Vetterli (2007), "The effective rank"

from __future__ import annotations
import sys, os, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "step2-a" / "datasets"
OUT_DIR = Path(__file__).parent / "results"
MIXES = ["mix_000", "mix_025", "mix_050", "mix_075", "mix_100"]
OPTEXC_PCTS = [0, 25, 50, 75, 100]

# NN input feature names (SOP v2.00)
FEATURE_NAMES = ["u", "v", "r", "δ_stu", "n", "δ̇_stu", "δ_ref", "δ̇_ref"]


def effective_rank(X: np.ndarray) -> float:
    """
    Compute the effective rank of matrix X.
    
    eRank(X) = exp(H(σ)) where H(σ) = -Σ p_i log(p_i)
    and p_i = σ_i / Σ σ_j (normalized singular values).
    
    Roy & Vetterli (2007): "The effective rank: A measure of 
    effective dimensionality"
    
    Returns a value in [1, min(m, n)] — higher means more spread 
    across dimensions.
    """
    # Center the data
    X_centered = X - X.mean(axis=0)
    s = np.linalg.svd(X_centered, compute_uv=False)
    # Normalize
    p = s / (s.sum() + 1e-30)
    # Shannon entropy of singular value distribution
    H = -np.sum(p * np.log(p + 1e-30))
    return np.exp(H)


def singular_value_spectrum(X: np.ndarray) -> np.ndarray:
    """Return normalized singular values for spectrum analysis."""
    X_centered = X - X.mean(axis=0)
    s = np.linalg.svd(X_centered, compute_uv=False)
    return s / (s[0] + 1e-30)  # normalized by largest


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    all_sv_spectra_raw = {}
    all_sv_spectra_norm = {}
    
    print("=" * 70)
    print("EFFECTIVE RANK ANALYSIS")
    print("=" * 70)
    
    for mix, pct in zip(MIXES, OPTEXC_PCTS):
        npz = np.load(DATA_DIR / f"train_{mix}.npz")
        X = npz["X"].astype(np.float64)
        Y = npz["Y"].astype(np.float64)
        
        # 1. Raw feature space eRank
        erank_raw = effective_rank(X)
        sv_raw = singular_value_spectrum(X)
        all_sv_spectra_raw[mix] = sv_raw.tolist()
        
        # 2. Normalized feature space eRank (what the NN actually sees)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-8] = 1.0  # avoid div by zero
        X_norm = (X - X_mean) / X_std
        erank_norm = effective_rank(X_norm)
        sv_norm = singular_value_spectrum(X_norm)
        all_sv_spectra_norm[mix] = sv_norm.tolist()
        
        # 3. Per-output-channel: input subspace weighted by output variance
        # This tells us which input directions contribute to each output
        erank_per_channel = {}
        for ch_idx, ch_name in enumerate(["u_dot", "v_dot", "r_dot"]):
            # Weight inputs by absolute output correlation
            y_ch = Y[:, ch_idx]
            if np.std(y_ch) < 1e-12:
                erank_per_channel[ch_name] = float("nan")
                continue
            # Compute correlation of each input with this output
            corr = np.array([np.abs(np.corrcoef(X_norm[:, j], y_ch)[0, 1]) 
                             for j in range(X_norm.shape[1])])
            corr[np.isnan(corr)] = 0
            # Weighted input: scale columns by correlation importance
            X_weighted = X_norm * corr[np.newaxis, :]
            erank_per_channel[ch_name] = effective_rank(X_weighted)
        
        # 4. Output space eRank (label diversity)
        erank_output = effective_rank(Y)
        
        results[mix] = {
            "pct": pct,
            "erank_raw": float(erank_raw),
            "erank_normalized": float(erank_norm),
            "erank_per_channel": erank_per_channel,
            "erank_output": float(erank_output),
            "sv_spectrum_raw": sv_raw.tolist(),
            "sv_spectrum_norm": sv_norm.tolist(),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        }
        
        print(f"\n  {mix} ({pct}% OptExc):")
        print(f"    eRank(raw):  {erank_raw:.3f} / {X.shape[1]}")
        print(f"    eRank(norm): {erank_norm:.3f} / {X.shape[1]}")
        print(f"    eRank(Y):    {erank_output:.3f} / {Y.shape[1]}")
        for ch, val in erank_per_channel.items():
            print(f"    eRank({ch}-weighted): {val:.3f}")
    
    # Save results
    results_path = OUT_DIR / "effective_rank_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved: {results_path}")
    
    # ======================== PLOT ========================
    plot_effective_rank(results)
    plot_sv_spectrum(results)
    
    print(f"\n[OK] All plots saved to {OUT_DIR}")


def plot_effective_rank(results):
    """Bar chart: eRank vs OptExc% for raw, normalized, and output spaces."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    pcts = [results[m]["pct"] for m in MIXES]
    
    # (a) Raw vs Normalized eRank
    ax = axes[0]
    erank_raw = [results[m]["erank_raw"] for m in MIXES]
    erank_norm = [results[m]["erank_normalized"] for m in MIXES]
    
    x = np.arange(len(pcts))
    w = 0.35
    ax.bar(x - w/2, erank_raw, w, label="Raw features", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, erank_norm, w, label="Normalized features", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.set_xlabel("OptExc Proportion")
    ax.set_ylabel("Effective Rank")
    ax.set_title("(a) Data Manifold Dimensionality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)
    # Annotate max dimensionality
    ax.axhline(y=8, color="gray", linestyle="--", alpha=0.5, label=f"Max = {8}")
    
    # (b) Output space eRank
    ax = axes[1]
    erank_out = [results[m]["erank_output"] for m in MIXES]
    ax.bar(pcts, erank_out, width=15, color="#27ae60", alpha=0.85)
    ax.set_xlabel("OptExc Proportion (%)")
    ax.set_ylabel("Effective Rank")
    ax.set_title("(b) Label Space Dimensionality")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0, top=3.5)
    
    # (c) Per-channel weighted eRank
    ax = axes[2]
    channels = ["u_dot", "v_dot", "r_dot"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for ch, color in zip(channels, colors):
        vals = [results[m]["erank_per_channel"].get(ch, 0) for m in MIXES]
        ax.plot(pcts, vals, "o-", color=color, label=ch, markersize=5)
    ax.set_xlabel("OptExc Proportion (%)")
    ax.set_ylabel("Effective Rank")
    ax.set_title("(c) Output-Weighted Input Diversity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "effective_rank.png", dpi=200)
    plt.savefig(OUT_DIR / "effective_rank.pdf")
    plt.close()
    print("[OK] Saved effective_rank.png/pdf")


def plot_sv_spectrum(results):
    """Singular value spectrum for each mix — visualizes manifold collapse."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    colors = ["#2c3e50", "#e74c3c", "#e67e22", "#27ae60", "#3498db"]
    
    for panel_idx, (space, title) in enumerate([
        ("sv_spectrum_raw", "Raw Feature Space"),
        ("sv_spectrum_norm", "Normalized Feature Space"),
    ]):
        ax = axes[panel_idx]
        for i, (mix, color) in enumerate(zip(MIXES, colors)):
            sv = np.array(results[mix][space])
            ax.semilogy(range(1, len(sv) + 1), sv, "o-", color=color,
                        label=f"{OPTEXC_PCTS[i]}% OptExc", markersize=4, alpha=0.8)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Normalized σ / σ₁")
        ax.set_title(f"({chr(97+panel_idx)}) SV Spectrum: {title}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 8.5)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sv_spectrum.png", dpi=200)
    plt.savefig(OUT_DIR / "sv_spectrum.pdf")
    plt.close()
    print("[OK] Saved sv_spectrum.png/pdf")


if __name__ == "__main__":
    main()
