"""Caricamento modello e inferenza greedy deterministica."""

import os
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_determinism(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(hf_id: str, revision: str, device: str | None = None):
    if device is None:
        device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 4096) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
