from typing import Optional, List, Dict
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dotenv import load_dotenv
load_dotenv()

# from huggingface_hub import login
# login(token=os.getenv("hf_hub_token"), add_to_git_credential=True)
token=os.getenv("hf_hub_token")

class MetaLlamaAgent:
    """
    Local HF agent
      - set_system_prompt(str)
      - set_parameters(max_tokens: int = None, temperature: float = None)
      - ask(question: str) -> str

    Works with Meta Llama, Mistral, and other chat-instruct models from Hugging Face.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,     # allow override or use HF_MODEL from .env
        dtype: Optional[str] = None,          # "float16" | "bfloat16" | None
        device: Optional[str] = None,         # "cuda" | "cpu"
        trust_remote_code: bool = False
    ):
        # Resolve model/device/dtype
        model_name = model_name or os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
        self.device = device or os.getenv("HF_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_str = dtype or os.getenv("HF_DTYPE")
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_str, None)

        # float16 on CPU is not supported — fall back cleanly
        if self.device == "cpu" and self.dtype == torch.float16:
            self.dtype = None

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )

        # Some models don't set a pad token; use EOS as pad to avoid warnings
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        # )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.dtype or torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=bnb_config,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            token=token
        )
        self.model.eval()

        # Generation params (can be updated via set_parameters)
        self.system_prompt = ""
        self.max_tokens = 512
        self.temperature = 0.2

    # --- Public API  ---

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt or ""

    def set_parameters(self, max_tokens: Optional[int] = None, temperature: Optional[float] = None):
        if max_tokens is not None:
            self.max_tokens = int(max_tokens)
        if temperature is not None:
            self.temperature = float(temperature)

    def _msgs(self, user_text: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    # --- Main call ---

    def ask(self, question: str) -> str:
        messages = self._msgs(question)

        # Use chat template if available; otherwise simple "User/Assistant" fallback
        if getattr(self.tokenizer, "chat_template", None):
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            sys = f"{self.system_prompt}\n\n" if self.system_prompt else ""
            prompt_text = f"{sys}User: {question}\nAssistant:"

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=(self.temperature > 0.0),
            temperature=self.temperature if self.temperature > 0.0 else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during generation. Try smaller max_tokens, lower HF_DTYPE (e.g., float16), "
                "or a smaller/quantized model."
            )

        # Return only newly generated tokens (strip the prompt)
        gen_ids = output[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text

