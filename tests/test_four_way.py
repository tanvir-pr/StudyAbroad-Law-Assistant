from pathlib import Path

from finetune.generators import TaggedEchoGenerator, load_generator
from finetune.train_qlora import load_config
from ingestion.sources import ROOT


def test_second_qlora_config_differs():
    primary = load_config(ROOT / "finetune" / "qlora.yaml")
    alt = load_config(ROOT / "finetune" / "qlora_r8_seed7.yaml")
    assert primary["lora"]["r"] == 16
    assert primary["seed"] == 42
    assert alt["lora"]["r"] == 8
    assert alt["seed"] == 7
    assert alt["output_dir"].endswith("qlora_r8_seed7")


def test_tagged_echo_base_vs_lora_no_rag():
    base = TaggedEchoGenerator("base")
    lora = TaggedEchoGenerator("lora")
    messages = [{"role": "user", "content": "QUESTION: work hours?\n\nWrite the answer now."}]
    base_ans = base.generate(messages, 64, 0.0)
    lora_ans = lora.generate(messages, 64, 0.0)
    assert "REFUSE:" not in base_ans  # ungrounded base may invent (demo failure mode)
    assert "REFUSE:" in lora_ans


def test_load_generator_echo_arms():
    g0, _, mode0 = load_generator(use_lora=False, echo=True, adapter_dir=None)
    g1, _, mode1 = load_generator(use_lora=True, echo=True, adapter_dir=Path("missing"))
    assert mode0.startswith("echo")
    assert mode1.startswith("echo")
    assert g0.tag == "base"
    assert g1.tag == "lora"
