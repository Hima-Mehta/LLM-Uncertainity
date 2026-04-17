# src/eval_metrics.py
import json
from typing import List, Dict
from collections import defaultdict
import math

from sklearn.metrics import roc_auc_score


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def compute_auroc(records: List[Dict], key: str) -> float:
    """
    key: uncertainty key, e.g. 'seq_avg_nll' or 'token_entropy_max'
    label: 1 if incorrect, 0 if correct
    """
    y_true = []
    y_score = []

    for r in records:
        val = r.get(key, None)
        if val is None:
            continue
        # Skip NaN / inf
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            continue

        y_true.append(0 if r["correct"] else 1)
        y_score.append(val)

    # Need at least one positive and one negative label
    if len(set(y_true)) < 2:
        return float("nan")

    return roc_auc_score(y_true, y_score)

def compute_coverage_accuracy(records: List[Dict], key: str, num_points: int = 10):
    """
    Sort by uncertainty ascending (most confident first)
    and compute accuracy for different coverage levels.
    """
    filtered = [r for r in records if r.get(key) is not None]
    filtered.sort(key=lambda r: r[key])  # ascending: low uncertainty first

    n = len(filtered)
    results = []
    for i in range(1, num_points + 1):
        frac = i / num_points
        k = max(1, int(n * frac))
        subset = filtered[:k]
        acc = sum(1 for r in subset if r["correct"]) / len(subset)
        results.append({"coverage": frac, "accuracy": acc})
    return results


def compute_reliability_bins(records: List[Dict], conf_key: str, num_bins: int = 10):
    """
    For calibration: use confidence key (e.g. 'seq_confidence')
    and compute mean confidence and accuracy per bin.
    """
    bins = defaultdict(list)
    for r in records:
        conf = r.get(conf_key, None)
        if conf is None:
            continue
        conf_clipped = max(0.0, min(1.0, float(conf)))
        bin_idx = min(num_bins - 1, int(conf_clipped * num_bins))
        bins[bin_idx].append(r)

    bin_stats = []
    total_n = sum(len(v) for v in bins.values())
    ece = 0.0

    for b in range(num_bins):
        bucket = bins.get(b, [])
        if not bucket:
            bin_stats.append({
                "bin": b,
                "count": 0,
                "mean_conf": None,
                "accuracy": None,
            })
            continue
        confs = [max(0.0, min(1.0, float(r[conf_key]))) for r in bucket]
        acc = sum(1 for r in bucket if r["correct"]) / len(bucket)
        mean_conf = sum(confs) / len(confs)
        bin_frac = len(bucket) / total_n
        ece += bin_frac * abs(acc - mean_conf)
        bin_stats.append({
            "bin": b,
            "count": len(bucket),
            "mean_conf": mean_conf,
            "accuracy": acc,
        })

    return bin_stats, ece


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--singlepass", type=str, required=True,
                        help="Path to singlepass.jsonl")
    parser.add_argument("--selfcons", type=str, required=False,
                        help="Path to selfconsistency.jsonl")

    args = parser.parse_args()

    sp = load_jsonl(args.singlepass)

    for task in ["qa", "math"]:
        task_records = [r for r in sp if r["task"] == task]
        print(f"\n=== Single-pass metrics for task: {task} ===")
        for key in ["seq_avg_nll", "token_entropy_max"]:
            au = compute_auroc(task_records, key)
            print(f"AUROC (unc={key}): {au:.3f}")
        bins, ece = compute_reliability_bins(task_records, "seq_confidence")
        print(f"ECE (seq_confidence): {ece:.3f}")

    if args.selfcons:
        sc = load_jsonl(args.selfcons)
        for task in ["qa", "math"]:
            task_records = [r for r in sc if r["task"] == task]
            print(f"\n=== Self-consistency metrics for task: {task} ===")
            au = compute_auroc(task_records, "self_consistency_uncertainty")
            print(f"AUROC (unc=self_consistency_uncertainty): {au:.3f}")
