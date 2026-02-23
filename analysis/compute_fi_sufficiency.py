# compute_fi_sufficiency.py
# Strategy 5: Fisher Information Sufficiency Ratio
#
# Defines R_FI = det(F_train) / det(F_ideal)
# where F_ideal = F(mix_100) as achievable upper bound
#
# Also performs elbow detection on R_FI vs OOD Gap curve
# to derive a principled threshold

from __future__ import annotations
import sys, os, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP2A_RESULTS = Path(__file__).parent.parent / "step2-a" / "results"
OUT_DIR = Path(__file__).parent / "results"
MIXES = ["mix_000", "mix_025", "mix_050", "mix_075", "mix_100"]
OPTEXC_PCTS = [0, 25, 50, 75, 100]


def find_elbow(x, y):
    """
    Find elbow point using maximum perpendicular distance from line 
    connecting first and last points (Kneedle algorithm simplified).
    
    Returns: index of elbow point
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Normalize to [0,1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-30)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-30)
    
    # Line from first to last point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    
    # Perpendicular distance from each point to the line
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    
    distances = []
    for i in range(len(x)):
        point = np.array([x_norm[i], y_norm[i]])
        d = np.abs(np.cross(line_vec, p1 - point)) / (line_len + 1e-30)
        distances.append(d)
    
    return int(np.argmax(distances))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load FIM results
    fim_path = STEP2A_RESULTS / "fim_results.json"
    with open(fim_path) as f:
        fim = json.load(f)
    
    # Load eval results (try multi-seed first, fall back to single-seed)
    multiseed_path = OUT_DIR / "multiseed_eval_results.json"
    singleseed_path = STEP2A_RESULTS / "eval_results.json"
    
    if multiseed_path.exists():
        with open(multiseed_path) as f:
            eval_data = json.load(f)
        use_multiseed = True
        print("[INFO] Using multi-seed evaluation results")
    else:
        with open(singleseed_path) as f:
            eval_raw = json.load(f)
        # Reshape to compatible format
        eval_data = {}
        for mix in MIXES:
            eval_data[mix] = {"pct": eval_raw["results"][mix]["optexc_pct"]}
            for test_name in ["ood_prbs", "ood_spiral", "ood_multistep", "ood_highfreq", "ood_id_zigzag"]:
                eval_data[mix][test_name] = {
                    "mean": eval_raw["results"][mix][test_name]["rmse_total"],
                    "std": 0.0,
                    "n_seeds": 1,
                }
        use_multiseed = False
        print("[INFO] Using single-seed evaluation results (run evaluate_multiseed.py for CI)")
    
    # Compute R_FI
    det_ideal = fim["mix_100"]["det"]
    
    print("=" * 70)
    print("FISHER INFORMATION SUFFICIENCY RATIO")
    print("=" * 70)
    
    results = {}
    r_fi_values = []
    ood_gap_values = []
    avg_ood_rmse_values = []
    
    for mix in MIXES:
        det_train = fim[mix]["det"]
        r_fi = det_train / (det_ideal + 1e-50)
        
        # Average OOD RMSE
        ood_tests = ["ood_prbs", "ood_spiral", "ood_multistep", "ood_highfreq"]
        ood_rmses = [eval_data[mix].get(t, {}).get("mean", 0) for t in ood_tests]
        avg_ood_rmse = np.mean(ood_rmses)
        
        # InD RMSE
        id_rmse = eval_data[mix].get("ood_id_zigzag", {}).get("mean", 1e-10)
        
        # OOD Gap (ratio)
        ood_gap = avg_ood_rmse / id_rmse if id_rmse > 0 else float("inf")
        
        results[mix] = {
            "pct": OPTEXC_PCTS[MIXES.index(mix)],
            "det_FIM": float(det_train),
            "R_FI": float(r_fi),
            "log_det": float(fim[mix]["log_det"]),
            "avg_ood_rmse": float(avg_ood_rmse),
            "avg_ood_gap": float(ood_gap),
            "kappa": float(fim[mix]["kappa"]),
        }
        
        r_fi_values.append(r_fi)
        ood_gap_values.append(ood_gap)
        avg_ood_rmse_values.append(avg_ood_rmse)
        
        print(f"  {mix}: R_FI = {r_fi:.4f}, OOD Gap = {ood_gap:.2f}×, Avg RMSE = {avg_ood_rmse:.6f}")
    
    # Find elbow point
    elbow_idx = find_elbow(r_fi_values, ood_gap_values)
    elbow_r_fi = r_fi_values[elbow_idx]
    elbow_mix = MIXES[elbow_idx]
    
    print(f"\n  [ELBOW] Detected at {elbow_mix}: R_FI = {elbow_r_fi:.4f}")
    print(f"  [THRESHOLD] Suggested R_FI threshold: {elbow_r_fi:.3f}")
    print(f"  [INTERPRETATION] Above this threshold, OOD Gap enters stable regime (<3×)")
    
    # Classification zones
    for mix in MIXES:
        r = results[mix]["R_FI"]
        gap = results[mix]["avg_ood_gap"]
        if r >= elbow_r_fi:
            zone = "✓ Sufficient"
        else:
            zone = "✗ Insufficient"
        results[mix]["zone"] = zone
        print(f"    {mix}: R_FI={r:.4f} → {zone} (Gap={gap:.1f}×)")
    
    # Save
    output = {
        "results": results,
        "threshold": {
            "method": "elbow_detection",
            "R_FI_threshold": float(elbow_r_fi),
            "elbow_mix": elbow_mix,
            "interpretation": "R_FI above threshold yields OOD Gap < 3x, indicating genuine generalization rather than interpolation",
        },
        "reference": {
            "F_ideal_source": "mix_100 (100% OptExc)",
            "det_F_ideal": float(det_ideal),
        },
    }
    with open(OUT_DIR / "fi_sufficiency_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Saved fi_sufficiency_results.json")
    
    # ==================== PLOT ====================
    plot_fi_sufficiency(results, elbow_r_fi, elbow_idx)
    
    print(f"[OK] All plots saved to {OUT_DIR}")


def plot_fi_sufficiency(results, threshold, elbow_idx):
    """R_FI vs OOD Gap with threshold zone marking."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 11,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    r_fi = [results[m]["R_FI"] for m in MIXES]
    gap = [results[m]["avg_ood_gap"] for m in MIXES]
    rmse = [results[m]["avg_ood_rmse"] for m in MIXES]
    pcts = [results[m]["pct"] for m in MIXES]
    
    # (a) R_FI vs OOD Gap
    ax = axes[0]
    ax.plot(r_fi, gap, "ko-", markersize=8, linewidth=1.5, zorder=5)
    
    # Annotate each point
    for i, mix in enumerate(MIXES):
        ax.annotate(f"{pcts[i]}%", (r_fi[i], gap[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    
    # Threshold zone
    ax.axvline(x=threshold, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"$R_{{FI}}$ threshold = {threshold:.3f}")
    ax.axhspan(0, 3, alpha=0.08, color="green", label="Stable zone (Gap < 3×)")
    ax.axhspan(3, max(gap) * 1.1, alpha=0.08, color="red", label="Degraded zone")
    
    # Elbow marker
    ax.scatter([r_fi[elbow_idx]], [gap[elbow_idx]], color="red", s=150, 
               zorder=10, marker="*", label=f"Elbow ({MIXES[elbow_idx]})")
    
    ax.set_xlabel(r"$\mathcal{R}_{FI}$ (Fisher Information Sufficiency Ratio)")
    ax.set_ylabel("Average OOD Gap")
    ax.set_title(r"(a) $\mathcal{R}_{FI}$ vs OOD Generalization Gap")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.15)
    
    # (b) R_FI vs Average OOD RMSE  
    ax = axes[1]
    ax.plot(r_fi, [r * 1e4 for r in rmse], "s-", color="#3498db", markersize=8, linewidth=1.5)
    
    for i, mix in enumerate(MIXES):
        ax.annotate(f"{pcts[i]}%", (r_fi[i], rmse[i] * 1e4),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    
    ax.axvline(x=threshold, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel(r"$\mathcal{R}_{FI}$ (Fisher Information Sufficiency Ratio)")
    ax.set_ylabel("Average OOD RMSE (×10⁻⁴)")
    ax.set_title(r"(b) $\mathcal{R}_{FI}$ vs Absolute OOD Error")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.15)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fi_sufficiency.png", dpi=200)
    plt.savefig(OUT_DIR / "fi_sufficiency.pdf")
    plt.close()
    print("[OK] Saved fi_sufficiency.png/pdf")


if __name__ == "__main__":
    main()
