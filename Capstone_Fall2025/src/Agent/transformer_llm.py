import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenAgent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", load_4bit: bool = True):

        self.system_prompt = ""
        self.max_tokens = 1000
        self.temperature = 0.0

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        kwargs = {"device_map": "auto"}
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        if load_4bit:
            kwargs.update(dict(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ))

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt or ""

    def set_parameters(self, max_tokens=None, temperature=None):
        if max_tokens is not None:
            self.max_tokens = int(max_tokens)
        if temperature is not None:
            self.temperature = float(temperature)

    def ask(self, question: str) -> str:
        """
        Uses Qwen's chat template so user can pass system/user like with Bedrock.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})

        # Build the prompt using the tokenizer’s chat template (handles Qwen formatting)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # temperature==0 → deterministic (greedy). Else sampling.
        do_sample = self.temperature > 0
        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            do_sample=do_sample,
            temperature=max(self.temperature, 1e-6) if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(gen_out[0], skip_special_tokens=True)

        # Extract only the assistant’s new text (strip the prompt part)
        prompt_len = inputs.input_ids.shape[-1]
        new_tokens = gen_out[0][prompt_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer if answer else text.strip()

