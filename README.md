# LLM-Uncertainty

**Uncertainty Quantification in Language Models — token-level and sentence-level, no retraining required.**

LLMs can generate confident-sounding text even when incorrect. This project extracts uncertainty signals directly from model logits during inference, helping determine when a model "knows that it doesn't know."

---

## Features

- Token-level uncertainty (entropy, max probability gap)
- Sentence-level uncertainty (avg NLL, sequence confidence)
- AUROC evaluation of uncertainty quality
- Coverage–accuracy selective answering simulation
- Calibration (reliability) and Expected Calibration Error (ECE)
- Visualizations for uncertainty interpretation

---

## Methods

At each decoding step `t`, logits are converted to probabilities via softmax:

```
p_t(i) = exp(z_t(i)) / Σ_j exp(z_t(j))
```

### Token-level uncertainty

```
H_t     = - Σ ( p_t(i) * log(p_t(i)) )   # token entropy
U_t     = 1 - max_i p_t(i)                # low max-prob = high uncertainty
```

### Sentence-level uncertainty

```
NLL         = - Σ ( log( p_t(y_t) ) )
avg_NLL     = (1 / T) * Σ ( - log(p_t(y_t)) )
confidence  = exp( - avg_NLL )
mean_entropy = (1 / T) * Σ ( H_t )
max_entropy  = max_t ( H_t )
```

---

## Code Structure

```
├── datasets.py              # QA + Math sample generation
├── models.py                # LLM + logits wrapper
├── generate_singlepass.py   # Outputs + uncertainty metrics
├── eval_metrics.py          # AUROC, calibration, selective answering
├── viz.py                   # Plots for results interpretation
└── experiments/             # Saved model outputs + plots
```

---

## Tasks & Evaluation

| Task Type | Assessment |
|---|---|
| Factual QA | Normalised text match |
| Math Word Problems | Numeric match |

Evaluation metrics:

- **AUROC** — how well uncertainty predicts incorrect answers
- **Coverage vs Accuracy** — does selective answering improve safety?
- **Calibration + ECE** — does confidence match actual accuracy?

---

## Requirements

```
python 3.9+
torch
transformers
accelerate
scikit-learn
matplotlib
tqdm
```

---

## Usage

### Step 1 — Generate model outputs and uncertainty metrics

```bash
python generate_singlepass.py \
  --model_name microsoft/Phi-2 \
  --out experiments/singlepass.jsonl \
  --n_qa 40 --n_math 40 \
  --max_new_tokens 32
```

Output (`singlepass.jsonl`) contains: prompt, output, correctness, token-level entropy values, avg_NLL, sequence confidence.

### Step 2 — Compute metrics

```bash
python eval_metrics.py \
  --singlepass experiments/singlepass.jsonl
```

Example output:

```
AUROC(avg_NLL) = 0.81
ECE = 0.12
```

### Step 3 — Generate plots

```bash
python viz.py \
  --singlepass experiments/singlepass.jsonl \
  --outdir experiments/plots
```

| Plot | Filename |
|---|---|
| Coverage vs Accuracy | `coverage_accuracy.png` |
| Calibration (Reliability) Diagram | `calibration.png` |

---

## Key Results

- `avg_NLL` was the strongest predictor of incorrect answers
- Token entropy spikes identified confusion during reasoning steps
- Selective answering (removing high-uncertainty responses) significantly improved accuracy
- Calibration plots revealed slight overconfidence in model outputs

---

## Interpretation Guide

| Metric / Plot | What it tells you |
|---|---|
| AUROC | How well uncertainty predicts wrong answers |
| Coverage–Accuracy | Does selective answering improve safety? |
| Calibration | Does confidence match actual accuracy? |
| Token entropy over time | Where the model became confused |

---

## Practical Applications

- **Safe abstention** — refuse to answer when uncertain
- **Human-in-the-loop** — route low-confidence outputs to humans
- **Confidence-aware decisions** — use uncertainty as a signal in autonomous systems
- **RAG quality control** — filter unreliable retrieved-context responses

---

## Notes

- All uncertainty metrics are computed during inference — no fine-tuning or retraining required
- Compatible with any open-source LLM that exposes logits
- GPU strongly recommended for faster experiment execution
