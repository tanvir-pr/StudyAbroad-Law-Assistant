"""Load Qwen for cited answers. 4-bit on CUDA; full precision on CPU."""

from dataclasses import dataclass
from typing import Protocol

import torch


class Generator(Protocol):
    def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str: ...


@dataclass
class GenerationConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    load_in_4bit: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.1


class QwenGenerator:
    def __init__(self, config: GenerationConfig | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.config = config or GenerationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        use_4bit = bool(self.config.load_in_4bit and torch.cuda.is_available())
        if use_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quant,
                device_map="auto",
            )
            self.device = None
            self.mode = "4bit-cuda"
        else:
            dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.model.to("cpu")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.mode = f"full-{self.device}"
        self.model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class EchoGenerator:
    """Deterministic stand-in for tests. Cites the first hit or refuses."""

    def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        content = "\n".join(message["content"] for message in messages)
        if "(no passages retrieved)" in content:
            return (
                "REFUSE: I cannot cite an official source from the locked allow-list "
                "for this question. Check OIF, Tempus Public Foundation, or the university."
            )
        source_id = None
        source_url = None
        for line in content.splitlines():
            stripped = line.strip()
            if "source_id=" in stripped and source_id is None:
                source_id = stripped.split("source_id=", 1)[1].strip()
            if stripped.startswith("source_url="):
                source_url = stripped.split("=", 1)[1].strip()
            if source_id and source_url:
                break
        if not source_id or not source_url:
            return (
                "REFUSE: I cannot cite an official source from the locked allow-list "
                "for this question. Check OIF, Tempus Public Foundation, or the university."
            )
        return (
            f"According to the retrieved passage, see the official text "
            f"[{source_id}]({source_url}). Not legal advice."
        )


class ExtractiveGenerator:
    """Quote the highest-overlap retrieved passage. Useful for local RAGAS tables."""

    def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        content = "\n".join(message["content"] for message in messages)
        if "(no passages retrieved)" in content:
            return (
                "REFUSE: I cannot cite an official source from the locked allow-list "
                "for this question. Check OIF, Tempus Public Foundation, or the university."
            )
        question = ""
        for line in content.splitlines():
            if line.startswith("QUESTION:"):
                question = line.split(":", 1)[1].strip().lower()
        tokens = [token for token in question.split() if len(token) > 2]
        blocks = content.split("\n\n")
        best = None
        best_score = -1.0
        for block in blocks:
            if "source_id=" not in block or "text=" not in block:
                continue
            source_id = None
            source_url = None
            text = ""
            for line in block.splitlines():
                if "source_id=" in line and source_id is None:
                    source_id = line.split("source_id=", 1)[1].strip()
                if line.startswith("source_url="):
                    source_url = line.split("=", 1)[1].strip()
                if line.startswith("text="):
                    text = line.split("=", 1)[1].strip()
            if not (source_id and source_url and text):
                continue
            lower = text.lower()
            score = sum(1.0 for token in tokens if token in lower) / max(len(tokens), 1)
            if score > best_score:
                best_score = score
                best = (source_id, source_url, text)
        if best is None:
            return (
                "REFUSE: I cannot cite an official source from the locked allow-list "
                "for this question. Check OIF, Tempus Public Foundation, or the university."
            )
        source_id, source_url, text = best
        snippet = " ".join(text.split())[:280]
        return f"{snippet} [{source_id}]({source_url}). Not legal advice."
