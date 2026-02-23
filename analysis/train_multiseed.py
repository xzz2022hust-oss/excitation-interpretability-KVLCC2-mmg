# train_multiseed.py
# Multi-seed training for Step 2 Extension
# Launches 5 mix × N seeds = 5N training jobs
#
# Reuses step2-a datasets (read-only), writes checkpoints to step2_extension/checkpoints
#
# Usage:
#   python train_multiseed.py                     # default: seeds 43,44,45 (seed42 reuse from step2-a)
#   python train_multiseed.py --seeds 43 44 45 46 47

from __future__ import annotations
import subprocess, sys, os, time, argparse, shutil
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).parent.parent / "train_kvlcc2_residual_v2.py"
DATA_DIR = Path(__file__).parent.parent / "step2-a" / "datasets"
OUT_BASE = Path(__file__).parent / "checkpoints"

# Reuse existing step2-a checkpoints for seed42
STEP2A_CKPT = Path(__file__).parent.parent / "step2-a" / "checkpoints"

MIXES = ["mix_000", "mix_025", "mix_050", "mix_075", "mix_100"]


def seed_torch_env(seed: int) -> dict:
    """Return environment variables that inject a seed into PyTorch/NumPy."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    return env


def build_seed_preamble(seed: int) -> str:
    """Python code to inject at start of training to control all randomness."""
    return (
        f"import torch, numpy as np, random; "
        f"random.seed({seed}); np.random.seed({seed}); "
        f"torch.manual_seed({seed}); "
        f"torch.cuda.manual_seed_all({seed}); "
        f"torch.backends.cudnn.deterministic=True; "
        f"torch.backends.cudnn.benchmark=False"
    )


def create_seed_wrapper(seed: int, dataset: Path, out_dir: Path, wrapper_path: Path):
    """Create a thin wrapper script that sets seed then invokes training."""
    # Use absolute path to project root for reliable imports
    project_root = str(Path(__file__).parent.parent.resolve()).replace("\\", "\\\\")
    code = f'''# Auto-generated seed wrapper for seed={seed}
import random, numpy as np, torch
random.seed({seed})
np.random.seed({seed})
torch.manual_seed({seed})
torch.cuda.manual_seed_all({seed})
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import sys, os
sys.path.insert(0, r"{project_root}")

from train_kvlcc2_residual_v2 import TrainConfig, Trainer

cfg = TrainConfig(
    dataset_path=r"{dataset}",
    output_dir=r"{out_dir}",
    member_id=0,
    stage=1,
    batch_size=64,
    lr=1e-3,
    max_epochs=200,
    patience=20,
)

print("=" * 60)
print(f"Training seed={seed}, dataset={{cfg.dataset_path}}")
print(f"Output: {{cfg.output_dir}}")
print("=" * 60)

trainer = Trainer(cfg)
trainer.train()
'''
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    with open(wrapper_path, "w") as f:
        f.write(code)


def link_seed42(mix: str):
    """Symlink or copy seed42 checkpoint from step2-a."""
    src = STEP2A_CKPT / mix / "seed42" / "m00"
    dst = OUT_BASE / mix / "seed42" / "m00"
    if dst.exists():
        return  # already exists
    dst.mkdir(parents=True, exist_ok=True)
    
    # Copy model and training log (avoid symlink issues on Windows)
    for fname in ["stage1_model.pt", "stage1_training_log.json", "stage1_curve.png"]:
        s = src / fname
        d = dst / fname
        if s.exists() and not d.exists():
            shutil.copy2(s, d)
            print(f"  [COPY] {s} -> {d}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed ablation training")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45],
                        help="Seeds to train (seed42 reused from step2-a)")
    parser.add_argument("--max_parallel", type=int, default=5,
                        help="Max parallel processes (GPU memory limited)")
    args = parser.parse_args()
    
    seeds = args.seeds
    print("=" * 70)
    print(f"Multi-Seed Training: {len(MIXES)} mixes × {len(seeds)} new seeds")
    print(f"Seeds: {seeds} (+ seed42 from step2-a)")
    print("=" * 70)
    
    # Step 1: Link seed42 from step2-a (DISABLED — retrain on new datasets)
    # print("\n[Step 1] Linking seed42 checkpoints from step2-a...")
    # for mix in MIXES:
    #     link_seed42(mix)
    
    # Step 2: Generate wrapper scripts and launch training
    print(f"\n[Step 2] Launching {len(MIXES) * len(seeds)} training jobs...")
    
    all_jobs = []
    for mix in MIXES:
        dataset = DATA_DIR / f"train_{mix}.npz"
        for seed in seeds:
            out_dir = OUT_BASE / mix / f"seed{seed}" / "m00"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already trained
            model_path = out_dir / "stage1_model.pt"
            if model_path.exists():
                print(f"  [SKIP] {mix}/seed{seed} (already exists)")
                continue
            
            wrapper = OUT_BASE / mix / f"seed{seed}" / "train_wrapper.py"
            create_seed_wrapper(seed, dataset, out_dir, wrapper)
            all_jobs.append((mix, seed, wrapper, out_dir))
    
    if not all_jobs:
        print("\n[OK] All models already trained!")
        return
    
    print(f"\n  {len(all_jobs)} jobs to run (max {args.max_parallel} parallel)")
    
    # Launch in batches
    active = []
    completed = []
    job_idx = 0
    
    while job_idx < len(all_jobs) or active:
        # Launch new jobs up to max_parallel
        while len(active) < args.max_parallel and job_idx < len(all_jobs):
            mix, seed, wrapper, out_dir = all_jobs[job_idx]
            log_path = out_dir.parent / "train.log"
            log_f = open(log_path, "w")
            
            cmd = [sys.executable, str(wrapper)]
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                 cwd=str(Path(__file__).parent.parent))
            active.append((mix, seed, p, log_f, log_path))
            print(f"  [LAUNCH] {mix}/seed{seed}")
            job_idx += 1
        
        # Check for completions
        still_active = []
        for mix, seed, p, lf, lp in active:
            rc = p.poll()
            if rc is not None:
                lf.close()
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  [DONE] {mix}/seed{seed}: {status}")
                completed.append((mix, seed, rc))
            else:
                still_active.append((mix, seed, p, lf, lp))
        active = still_active
        
        if active:
            time.sleep(3)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    n_ok = sum(1 for _, _, rc in completed if rc == 0)
    n_fail = sum(1 for _, _, rc in completed if rc != 0)
    print(f"  OK: {n_ok}  FAIL: {n_fail}")
    for mix, seed, rc in completed:
        if rc != 0:
            log_path = OUT_BASE / mix / f"seed{seed}" / "train.log"
            print(f"  FAILED: {mix}/seed{seed} — see {log_path}")


if __name__ == "__main__":
    main()
