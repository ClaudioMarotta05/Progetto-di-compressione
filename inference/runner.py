"""Caricamento modello e inferenza greedy deterministica con chunking + prompt lookup."""
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
    # Su Linux con GPU usa CUDA; su Mac usa CPU float32 (più veloce di MPS e senza OOM/NaN)
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(hf_id: str, revision: str, device: str | None = None):
    if device is None:
        device = pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, revision=revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return model, tokenizer, device


def _format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def chunk_text(text: str, max_words: int = 200) -> list[str]:
    """Spezza il testo in blocchi da ~max_words parole rispettando i paragrafi."""
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in text.split("\n"):
        words = para.split()
        if len(words) > max_words:
            if current:
                chunks.append("\n".join(current))
                current, current_words = [], 0
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
        elif current_words + len(words) > max_words and current:
            chunks.append("\n".join(current))
            current = [para]
            current_words = len(words)
        else:
            current.append(para)
            current_words += len(words)

    if current:
        chunks.append("\n".join(current))
    return chunks


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    prompt_lookup_num_tokens: int = 10,
) -> str:
    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    if prompt_lookup_num_tokens:
        kwargs["prompt_lookup_num_tokens"] = prompt_lookup_num_tokens

    outputs = model.generate(**inputs, **kwargs)
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_chunked(
    model,
    tokenizer,
    strategy: str,
    skeleton: str,
    build_prompt_fn,
    ratio: float = 0.30,
    max_words_per_chunk: int = 200,
    prompt_lookup_num_tokens: int = 10,
    verbose: bool = True,
) -> str:
    """Inferenza blocco per blocco con prompt lookup (equivalente a greedy puro)."""
    chunks = chunk_text(skeleton, max_words=max_words_per_chunk)
    parts: list[str] = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        if not chunk.strip():
            parts.append(chunk)
            continue
        prompt = build_prompt_fn(strategy, chunk)
        chunk_toks = len(tokenizer.encode(chunk))
        expected = int(chunk_toks / max(0.40, 1.0 - ratio) * 1.25) + 40
        max_new = max(128, min(900, expected))

        out = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new,
            prompt_lookup_num_tokens=prompt_lookup_num_tokens,
        )
        parts.append(out)
        if verbose and (idx % 10 == 0 or idx == total):
            print(f"    chunk {idx}/{total}", flush=True)

    return "\n".join(parts)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
