#!/usr/bin/env python3
"""Qwen inference on the Colab VM. Resumable: skips ids already in the answers file.

Environment:
  ITEMS_PATH   default /content/items.json
  ANSWERS_PATH default /content/answers.jsonl
  BATCH_SIZE   default 40 (0 means all remaining)
  MODEL_REPO   default Qwen/Qwen3-4B
  MAX_NEW      default 160
"""
import json
import os
import subprocess
import sys
import time

ITEMS_PATH = os.environ.get("ITEMS_PATH", "/content/items.json")
ANSWERS_PATH = os.environ.get("ANSWERS_PATH", "/content/answers.jsonl")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "40"))
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen3-4B")
MAX_NEW = int(os.environ.get("MAX_NEW", "160"))


def emit(kind, payload):
    print(f"{kind} " + json.dumps(payload, ensure_ascii=False), flush=True)


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("id") and rec.get("text") and rec.get("exit", 0) == 0:
                    ids.add(rec["id"])
    return ids


def _ver_at_least(version, major, minor):
    parts = []
    for p in version.split("."):
        num = ""
        for ch in p:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            parts.append(int(num))
    while len(parts) < 2:
        parts.append(0)
    return (parts[0], parts[1]) >= (major, minor)


def install():
    import torch
    need = False
    try:
        import transformers
        if not _ver_at_least(transformers.__version__, 4, 51):
            need = True
    except ImportError:
        need = True
    if need:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "transformers>=4.51.0", "accelerate>=1.0.0",
        ])
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.45.0"])


def gpu_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        return out
    except Exception as e:
        return f"none ({e})"


def load_model(repo):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(repo)
    cuda = torch.cuda.is_available()
    if cuda:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo, quantization_config=qconf, device_map="auto",
        )
        quant = "nf4"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True,
        )
        quant = "fp16-cpu"
    model.eval()
    return tok, model, quant, cuda


def generate(tok, model, prompt):
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    enc = tok(rendered, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new = gen[0, enc["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True).strip()
    return text, time.time() - t0, int(new.shape[0])


def main():
    print("BOOT remote_infer", flush=True)
    install()
    import torch
    name = gpu_name()
    print("GPU", name, "cuda", torch.cuda.is_available(), flush=True)
    items = json.load(open(ITEMS_PATH))
    have = done_ids(ANSWERS_PATH)
    todo = [it for it in items if it["id"] not in have]
    if BATCH_SIZE > 0:
        todo = todo[:BATCH_SIZE]
    print(f"TODO {len(todo)} already {len(have)}", flush=True)
    if not todo:
        emit("META", {"gpu": name, "repo": MODEL_REPO, "skipped": True, "already": len(have)})
        emit("BATCH_DONE", {"n": 0})
        return 0

    repo = MODEL_REPO
    model_key = "qwen3_4b_q4"
    try:
        tok, model, quant, cuda = load_model(repo)
    except Exception as e:
        print("LOAD_FAIL", repo, repr(e), flush=True)
        if repo != "Qwen/Qwen3-1.7B":
            repo = "Qwen/Qwen3-1.7B"
            print("FALLBACK", repo, flush=True)
            tok, model, quant, cuda = load_model(repo)
            model_key = "qwen3_1.7b"
        else:
            raise
    if not cuda and repo.endswith("4B"):
        model_key = "qwen3_4b_fp16cpu"
    elif not cuda:
        model_key = "qwen3_1.7b"
    elif quant == "nf4" and "4B" in repo:
        model_key = "qwen3_4b_q4"
    meta = {
        "gpu": name, "repo": repo, "quant": quant, "cuda": bool(cuda),
        "model_key": model_key, "n_todo": len(todo),
    }
    emit("META", meta)
    os.makedirs(os.path.dirname(ANSWERS_PATH) or ".", exist_ok=True)
    for it in todo:
        try:
            text, wall, ntok = generate(tok, model, it["prompt"])
            rec = dict(
                model=model_key, id=it["id"], task=it["task"], text=text,
                wall_s=round(wall, 3), new_tokens=ntok, repo=repo, quant=quant,
                exit=0,
            )
        except Exception as e:
            rec = dict(
                model=model_key, id=it["id"], task=it["task"], text="",
                wall_s=None, exit=1, error=repr(e), repo=repo, quant=quant,
            )
        with open(ANSWERS_PATH, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        emit("RESULT", rec)
        print(f"ITEM {it['id']} exit={rec['exit']} chars={len(rec.get('text') or '')} {rec.get('wall_s')}", flush=True)
    emit("BATCH_DONE", {"n": len(todo), "model_key": model_key, "repo": repo, "quant": quant})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
