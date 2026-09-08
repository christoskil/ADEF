"""
Comparison Experiment: ADEF vs Baselines
===========================================
Runs ADEF (full integrated pipeline) and all four baselines on the
same dataset and prints a comparison table matching Table 2 of the
paper (storage ratio, reconstruction MSE, anomaly recall, fault F1,
mean detection delay).

Usage:
    python experiments/run_comparison.py [--N 20] [--T 2016] [--seed 42]
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_integrated import run_integrated, compute_all_metrics
from components.baselines import (
    StoreAllBaseline, LEACHBaseline, ThresholdBaseline,
    DistributedAutoencoderBaseline,
)


def run_comparison(N: int = 20, T: int = 2016, seed: int = 42):
    print(f"[1/5] Generating data and running ADEF (N={N}, T={T})...")
    r = run_integrated(N=N, T=T, seed=seed, verbose=False)
    adef_metrics = compute_all_metrics(r)

    data, ground_truth = r["data"], r["ground_truth"]
    positions = r["positions"]
    fault_map = {n.node_id: (n.fault_start, n.fault_end, n.fault_type)
                 for n in r["nodes"] if n.fault_type is not None}

    print("[2/5] Running Store-All baseline...")
    store_all = StoreAllBaseline().run(data, ground_truth, fault_map)

    print("[3/5] Running LEACH baseline...")
    leach = LEACHBaseline().run(data, ground_truth, positions, fault_map, seed=seed)

    print("[4/5] Running Threshold-Based baseline...")
    threshold = ThresholdBaseline().run(data, ground_truth, fault_map)

    print("[5/5] Running Distributed Autoencoder baseline...")
    autoenc = DistributedAutoencoderBaseline().run(data, ground_truth, fault_map, seed=seed)

    rows = [
        ("ADEF (Proposed)", adef_metrics["storage_ratio"], adef_metrics["mean_mse"],
         adef_metrics["anomaly_recall"], adef_metrics["fault_f1"],
         adef_metrics["mean_detection_delay"]),
        (store_all.name, store_all.storage_ratio, store_all.mean_mse,
         store_all.anomaly_recall, store_all.fault_f1, store_all.mean_detection_delay),
        (leach.name, leach.storage_ratio, leach.mean_mse,
         leach.anomaly_recall, leach.fault_f1, leach.mean_detection_delay),
        (threshold.name, threshold.storage_ratio, threshold.mean_mse,
         threshold.anomaly_recall, threshold.fault_f1, threshold.mean_detection_delay),
        (autoenc.name, autoenc.storage_ratio, autoenc.mean_mse,
         autoenc.anomaly_recall, autoenc.fault_f1, autoenc.mean_detection_delay),
    ]

    print("\n" + "=" * 88)
    print(f"{'Method':<32}{'Storage':>10}{'MSE':>10}{'Recall':>10}{'F1':>10}{'Delay':>10}")
    print("-" * 88)
    for name, sr, mse, rec, f1, delay in rows:
        delay_str = f"{delay:.1f}" if delay == delay else "--"  # NaN check
        print(f"{name:<32}{sr*100:>9.1f}%{mse:>10.3f}{rec:>10.3f}{f1:>10.3f}{delay_str:>10}")
    print("=" * 88)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADEF vs baselines comparison")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--T", type=int, default=2016)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_comparison(N=args.N, T=args.T, seed=args.seed)
