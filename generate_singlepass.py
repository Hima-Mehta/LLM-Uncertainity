# src/generate_singlepass.py
import json
from pathlib import Path

from tqdm import tqdm

from models import LLMWrapper
from datasets import load_qa_examples, load_math_examples, is_correct
from uncertainty import (
    token_metrics_from_scores,
    sequence_uncertainty,
    aggregate_token_uncertainty,
)


def run_singlepass(
    model_name: str,
    out_path: str,
    n_qa: int = 50,
    n_math: int = 50,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
):
    llm = LLMWrapper(model_name)
    examples = []
    examples += load_qa_examples(n_qa)
    examples += load_math_examples(n_math)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in tqdm(examples, desc="Single-pass generation"):
            pred_text, sequences, scores, input_ids = llm.generate_with_scores(
                ex.prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            prompt_len = input_ids.shape[-1]
            gen_ids_only = sequences[prompt_len:]

            token_metrics = token_metrics_from_scores(scores, gen_ids_only)
            seq_unc = sequence_uncertainty(token_metrics["token_log_probs"])
            tok_agg = aggregate_token_uncertainty(token_metrics["token_entropies"])

            correct_flag = is_correct(ex.gold_answer, pred_text, ex.task)

            rec = {
                "task": ex.task,
                "prompt": ex.prompt,
                "gold": ex.gold_answer,
                "pred": pred_text,
                "correct": bool(correct_flag),
                "token_entropies": token_metrics["token_entropies"],
                "token_max_probs": token_metrics["token_max_probs"],
                **seq_unc,
                **tok_agg,
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Saved single-pass records to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--out", type=str, default="experiments/singlepass.jsonl")
    parser.add_argument("--n_qa", type=int, default=50)
    parser.add_argument("--n_math", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()

    run_singlepass(
        model_name=args.model_name,
        out_path=args.out,
        n_qa=args.n_qa,
        n_math=args.n_math,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
