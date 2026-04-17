from dataclasses import dataclass
from typing import List
import re
import string


@dataclass
class Example:
    task: str          # "qa" or "math"
    prompt: str
    gold_answer: str


def load_qa_examples(n: int = 50) -> List[Example]:
    """
    For now: small hardcoded list (you can replace with TriviaQA subset later).
    """
    base = [
        Example("qa", "Q: What is the capital of France?\nA:", "Paris"),
        Example("qa", "Q: Who wrote the play 'Hamlet'?\nA:", "William Shakespeare"),
        Example("qa", "Q: What is the largest planet in our solar system?\nA:", "Jupiter"),
        Example("qa", "Q: In which country is the city of Tokyo located?\nA:", "Japan"),
        Example("qa", "Q: What is the chemical symbol for water?\nA:", "H2O"),
    ]
    # Repeat / truncate to reach n if needed
    out = (base * (n // len(base) + 1))[:n]
    return out


def load_math_examples(n: int = 50) -> List[Example]:
    """
    Simple arithmetic / word problems.
    """
    base = [
        Example("math", "Q: What is 7 + 5?\nA:", "12"),
        Example("math", "Q: If you have 15 apples and give away 7, how many are left?\nA:", "8"),
        Example("math", "Q: What is 9 * 6?\nA:", "54"),
        Example("math", "Q: What is 21 - 13?\nA:", "8"),
        Example("math", "Q: A box has 4 rows of 3 chocolates each. How many chocolates are there?\nA:", "12"),
    ]
    out = (base * (n // len(base) + 1))[:n]
    return out


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_number(t: str):
    m = re.search(r"-?\d+(\.\d+)?", t)
    return float(m.group(0)) if m else None


def is_correct(gold: str, pred: str, task: str) -> bool:
    """
    Simple correctness:
      - math: compare first number
      - qa: normalized gold substring in pred
    """
    if task == "math":
        g = extract_number(gold)
        p = extract_number(pred)
        return g is not None and p is not None and g == p
    else:
        gold_n = normalize_answer(gold)
        pred_n = normalize_answer(pred)
        return gold_n in pred_n
