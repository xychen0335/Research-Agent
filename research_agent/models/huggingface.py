"""Local Hugging Face causal LM policy.

Token IDs and sampled logprobs are returned so GRPO can use the behaviour policy
probabilities. Requires a Qwen3.5-capable transformers build; the import is lazy.
"""

from __future__ import annotations

import asyncio
from typing import Any

from research_agent.contracts import Generation
from research_agent.models.base import encode_text, estimate_tokens

STOP_STRING = "</tool_call>"


class HuggingFacePolicy:
    def __init__(
        self,
        *,
        model_name: str,
        policy_version: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        device: str | None = None,
        model: Any = None,
        tokenizer: Any = None,
        adapter: str | None = None,
        local_files_only: bool = False,
        trainable_adapter: bool = False,
    ):
        self.model_name = model_name
        self.policy_version = policy_version or model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._requested_device = device
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._adapter = adapter
        self._local_files_only = local_files_only
        self._trainable_adapter = trainable_adapter

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, local_files_only=self._local_files_only
        )
        if self._requested_device:
            device = self._requested_device
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self._device = device
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(
                torch.bfloat16
                if device == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                else (torch.float16 if device != "cpu" else torch.float32)
            ),
            trust_remote_code=True,
            local_files_only=self._local_files_only,
        )
        if self._adapter:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(
                self._model,
                self._adapter,
                is_trainable=self._trainable_adapter,
            )
            self.policy_version = f"{self.policy_version}+lora"
        elif self._trainable_adapter:
            from peft import LoraConfig, TaskType, get_peft_model

            if hasattr(self._model, "enable_input_require_grads"):
                self._model.enable_input_require_grads()
            self._model = get_peft_model(
                self._model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                ),
            )
            self.policy_version = f"{self.policy_version}+lora"
        self._model.to(device)
        self._model.eval()

    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        self._ensure_loaded()
        encoded = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, dict):
            return list(encoded["input_ids"])
        return list(encoded)

    def encode_text(self, text: str) -> list[int]:
        if self._tokenizer is None:
            return encode_text(text)
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def count_text(self, text: str) -> int:
        if self._tokenizer is None:
            return estimate_tokens(text)
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _generate_sync(self, messages: list[dict[str, str]]) -> Generation:
        import torch

        self._ensure_loaded()
        prompt_ids = self.encode_messages(messages)
        input_ids = torch.tensor([prompt_ids], device=self._device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self.temperature and self.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
        else:
            generate_kwargs["do_sample"] = False
        try:
            generate_kwargs["stop_strings"] = [STOP_STRING]
            generate_kwargs["tokenizer"] = self._tokenizer
            outputs = self._model.generate(input_ids, **generate_kwargs)
        except TypeError:
            generate_kwargs.pop("stop_strings", None)
            generate_kwargs.pop("tokenizer", None)
            outputs = self._model.generate(input_ids, **generate_kwargs)
        sequences = outputs.sequences[0]
        completion_ids = sequences[len(prompt_ids) :].tolist()
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        if text and not text.rstrip().endswith(STOP_STRING) and "<tool_call>" in text:
            text = text.rstrip() + f"\n{STOP_STRING}"
        logprobs: list[float] = []
        scores = getattr(outputs, "scores", None) or ()
        for step, score in enumerate(scores):
            if step >= len(completion_ids):
                break
            token_id = completion_ids[step]
            log_softmax = torch.nn.functional.log_softmax(score[0], dim=-1)
            logprobs.append(float(log_softmax[token_id].item()))
        return Generation(
            text=text.strip(),
            token_ids=completion_ids,
            logprobs=logprobs or None,
            prompt_token_ids=prompt_ids,
            finish_reason="stop",
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(completion_ids),
            usable_for_rl=bool(completion_ids) and bool(logprobs),
        )

    async def generate(self, messages: list[dict[str, str]]) -> Generation:
        return await asyncio.to_thread(self._generate_sync, messages)
