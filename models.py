from typing import Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMWrapper:
    """
    Simple wrapper around a causal LLM that can return logits (scores)
    for each generated token.
    """

    def __init__(self, model_name: str, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device is None else None,
            torch_dtype="auto",
        )
        if device is not None:
            self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_with_scores(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> Tuple[str, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Returns:
          pred_text: decoded new text (excluding prompt)
          sequences: tensor [total_len]
          scores: list of logits [1, vocab_size] per new token
          input_ids: tensor [1, prompt_len] for the prompt
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        sequences = output.sequences[0]   # [total_len]
        scores = output.scores            # list length T, each [1, vocab]

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids_only = sequences[prompt_len:]
        pred_text = self.tokenizer.decode(gen_ids_only, skip_special_tokens=True)

        return pred_text, sequences, scores, inputs["input_ids"]

    @torch.no_grad()
    def generate_text_only(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> str:
        """
        For self-consistency: just sample text, no scores.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            output_scores=False,
            return_dict_in_generate=False,
        )
        sequences = output[0]
        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids_only = sequences[prompt_len:]
        pred_text = self.tokenizer.decode(gen_ids_only, skip_special_tokens=True)
        return pred_text