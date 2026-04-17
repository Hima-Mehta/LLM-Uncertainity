# src/generate_selfconsistency.py
import json
from pathlib import Path

from tqdm import tqdm
from models import LLMWrapper
from datasets import load_qa_examples, load_math_examples, is_correct, normalize_answer, extract_number
from uncertainty import self_consistency_uncertainty


def normalize_for_task(answer: str, task: str) -> str:
    if task == "math":
        num = extract_number(answer)
        return str(num) if num is not None else ""
    else:
        return normalize_answer(answer)


def run_selfconsistency(
    model_name: str,
    out_path: str,
    n_qa: int = 50,
    n_math: int = 50,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    k_samples: int = 5,
):
    llm = LLMWrapper(model_name)
    examples = []
    examples += load_qa_examples(n_qa)
    examples += load_math_examples(n_math)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in tqdm(examples, desc="Self-consistency generation"):
            raw_answers = []
            norm_answers = []

            for _ in range(k_samples):
                ans = llm.generate_text_only(
                    ex.prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                raw_answers.append(ans)
                norm_answers.append(normalize_for_task(ans, ex.task))

            sc = self_consistency_uncertainty(norm_answers)
            majority_answer = sc["majority_answer"]
            correct_flag = is_correct(ex.gold_answer, majority_answer, ex.task)

            rec = {
                "task": ex.task,
                "prompt": ex.prompt,
                "gold": ex.gold_answer,
                "answers": raw_answers,
                "norm_answers": norm_answers,
                "majority_answer": majority_answer,
                "correct": bool(correct_flag),
                "self_consistency_conf": sc["self_consistency_conf"],
                "self_consistency_uncertainty": sc["self_consistency_uncertainty"],
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Saved self-consistency records to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--out", type=str, default="experiments/selfconsistency.jsonl")
    parser.add_argument("--n_qa", type=int, default=50)
    parser.add_argument("--n_math", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--k_samples", type=int, default=5)

    args = parser.parse_args()

    run_selfconsistency(
        model_name=args.model_name,
        out_path=args.out,
        n_qa=args.n_qa,
        n_math=args.n_math,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        k_samples=args.k_samples,
    )
