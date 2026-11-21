"""Load base or LoRA-adapted Qwen for day-18 4-way generation."""

from __future__ import annotations

from pathlib import Path

from rag.generate import EchoGenerator, GenerationConfig, QwenGenerator


class TaggedEchoGenerator(EchoGenerator):
    """Echo stand-in that labels which arm produced the text (CPU / no adapter)."""

    def __init__(self, tag: str):
        self.tag = tag
        self.mode = f"echo-{tag}"

    def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        content = "\n".join(message["content"] for message in messages)
        # No-RAG arms pass a user message without CONTEXT blocks.
        if "CONTEXT:" not in content and "(no passages retrieved)" not in content:
            if self.tag.startswith("lora"):
                return (
                    f"[{self.tag}] Without retrieved sources I should not invent rules. "
                    "REFUSE: I cannot cite an official source from the locked allow-list "
                    "for this question. Check OIF, Tempus Public Foundation, or the university."
                )
            return (
                f"[{self.tag}] Students may work many hours. Not legal advice."
            )
        answer = super().generate(messages, max_new_tokens, temperature)
        if answer.startswith("REFUSE:"):
            return answer
        return f"[{self.tag}] {answer}"


def load_generator(
    *,
    use_lora: bool,
    echo: bool,
    adapter_dir: Path | None,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
):
    if echo:
        tag = "lora" if use_lora else "base"
        return TaggedEchoGenerator(tag), f"echo-{tag}", f"echo-{tag}"

    if use_lora:
        adapter = adapter_dir
        if adapter is None or not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter}. Train with train_qlora on a T4 first."
            )
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        use_4bit = torch.cuda.is_available()
        kwargs = {"device_map": "auto"}
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        base = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model = PeftModel.from_pretrained(base, str(adapter))
        model.eval()

        class PeftGenerator:
            def __init__(self):
                self.tokenizer = tok
                self.model = model
                self.mode = "lora-4bit" if use_4bit else "lora-full"
                self.config = GenerationConfig(model_name=model_name)

            def generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()} if hasattr(self.model, "device") else inputs
                # device_map=auto: move to first param device
                try:
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except StopIteration:
                    pass
                do_sample = temperature > 0
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                new = out[0, inputs["input_ids"].shape[-1] :]
                return self.tokenizer.decode(new, skip_special_tokens=True).strip()

        gen = PeftGenerator()
        return gen, model_name + "+lora", gen.mode

    gen = QwenGenerator(GenerationConfig(model_name=model_name))
    return gen, model_name, gen.mode
