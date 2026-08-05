from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from typing import Any


class Generator:
    """Wrapper around a causal language model used to produce answers from prompts.

    Args:
        model_name: Hugging Face model identifier for the tokenizer and model.
    """

    def __init__(self, model_name: str):
        """Load the tokenizer and model for inference.

        Args:
            model_name: Name of the model to load from the Hugging Face hub.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )

    def generate(self, prompt: str) -> Any:
        """Generate a text completion for the supplied prompt.

        Args:
            prompt: User question or RAG prompt text sent to the model.

        Returns:
            Any: Generated completion text decoded from the model output.
        """
        messages = [
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                # max_new_tokens=256,
                max_new_tokens=128,
                do_sample=False
            )

        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
