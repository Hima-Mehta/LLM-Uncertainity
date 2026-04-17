# src/uncertainty.py
from typing import List, Dict
import math

import torch
import torch.nn.functional as F
from collections import Counter


def token_metrics_from_scores(
    scores: List[torch.Tensor],
    gen_ids_only: torch.Tensor,
) -> Dict[str, List[float]]:
    """
    scores: list of [1, vocab] logits for each generated step (length T)
    gen_ids_only: [T] tensor of generated token ids (excluding prompt)

    Returns dict with:
      - token_entropies
      - token_log_probs
      - token_max_probs
    """
    entropies = []
    log_probs = []
    max_probs = []

    for t, logits in enumerate(scores):
        probs = F.softmax(logits[0], dim=-1)       # [vocab]
        logp = F.log_softmax(logits[0], dim=-1)    # [vocab]

        token_id = gen_ids_only[t]

        entropy = -(probs * logp).sum().item()
        log_prob = logp[token_id].item()
        max_prob = probs.max().item()

        entropies.append(entropy)
        log_probs.append(log_prob)
        max_probs.append(max_prob)

    return {
        "token_entropies": entropies,
        "token_log_probs": log_probs,
        "token_max_probs": max_probs,
    }


def sequence_uncertainty(log_probs: List[float]) -> Dict[str, float]:
    """
    Compute sequence-level NLL and derived "confidence".
    """
    if len(log_probs) == 0:
        return {"seq_avg_nll": float("inf"), "seq_confidence": 0.0}

    nll = -sum(log_probs)
    avg_nll = nll / len(log_probs)
    confidence = math.exp(-avg_nll)  # monotonic w.r.t. avg_nll

    return {
        "seq_avg_nll": float(avg_nll),
        "seq_confidence": float(confidence),
    }


def aggregate_token_uncertainty(entropies: List[float]) -> Dict[str, float]:
    if not entropies:
        return {
            "token_entropy_mean": None,
            "token_entropy_max": None,
        }
    mean_ent = sum(entropies) / len(entropies)
    max_ent = max(entropies)
    return {
        "token_entropy_mean": float(mean_ent),
        "token_entropy_max": float(max_ent),
    }


def self_consistency_uncertainty(normalized_answers: List[str]) -> Dict[str, float]:
    """
    Given K normalized answers for one prompt, compute:
      - self_consistency_conf
      - self_consistency_uncertainty = 1 - conf
    """
    if not normalized_answers:
        return {
            "self_consistency_conf": 0.0,
            "self_consistency_uncertainty": 1.0,
        }

    counts = Counter(normalized_answers)
    majority_answer, majority_count = counts.most_common(1)[0]
    k = len(normalized_answers)
    conf = majority_count / k
    return {
        "majority_answer": majority_answer,
        "self_consistency_conf": float(conf),
        "self_consistency_uncertainty": float(1.0 - conf),
    }
