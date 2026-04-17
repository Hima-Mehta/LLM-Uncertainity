# src/viz.py
import json
from typing import Dict, List

import matplotlib.pyplot as plt

from eval_metrics import (
    load_jsonl,
    compute_coverage_accuracy,
    compute_reliability_bins,
)


def plot_coverage_accuracy(records: List[Dict], key: str, title: str, out_path: str):
    data = compute_coverage_accuracy(records, key)
    xs = [d["coverage"] for d in data]
    ys = [d["accuracy"] for d in data]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Coverage (fraction of answered queries)")
    plt.ylabel("Accuracy on answered subset")
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_reliability(records: List[Dict], conf_key: str, title: str, out_path: str):
    bins, ece = compute_reliability_bins(records, conf_key)
    xs = []
    ys = []
    for b in bins:
        if b["mean_conf"] is None:
            continue
        xs.append(b["mean_conf"])
        ys.append(b["accuracy"])

    plt.figure()
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.scatter(xs, ys, label="Model bins")
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"{title}\nECE={ece:.3f}")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--singlepass", type=str, required=True,
                        help="Path to singlepass.jsonl")
    parser.add_argument("--outdir", type=str, default="experiments/plots")

    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    sp = load_jsonl(args.singlepass)

    for task in ["qa", "math"]:
        task_records = [r for r in sp if r["task"] == task]

        plot_coverage_accuracy(
            task_records,
            key="seq_avg_nll",
            title=f"Coverage vs Accuracy (seq_avg_nll) - {task}",
            out_path=os.path.join(args.outdir, f"covacc_seqnll_{task}.png"),
        )

        plot_reliability(
            task_records,
            conf_key="seq_confidence",
            title=f"Reliability (seq_confidence) - {task}",
            out_path=os.path.join(args.outdir, f"reliability_{task}.png"),
        )
