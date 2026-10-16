# Local CPU models on the 20 prompts: greedy decoding, chat template, max 120 new tokens. Resumable (jsonl).
# MODEL_KEY=qwen3_0.6b | smollm2_360m
import json, os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__)); from tasks import items
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = "/workspace/projects/4e0e7b68-80db-40d3-a6d5-b21cf99d82b6/work"
PATHS = {"qwen3_0.6b": f"{W}/Qwen3-0.6B", "smollm2_360m": f"{W}/SmolLM2-360M-Instruct"}  # Qwen/Qwen3-0.6B, HuggingFaceTB/SmolLM2-360M-Instruct
KEY = os.environ.get("MODEL_KEY", "smollm2_360m"); OUT = f"{D}/results/local_{KEY}.jsonl"
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(PATHS[KEY]); m = AutoModelForCausalLM.from_pretrained(PATHS[KEY], dtype=torch.float32).eval()
done = {json.loads(l)["id"] for l in open(OUT)} if os.path.exists(OUT) else set()
for it in items():
    if it["id"] in done: continue
    kw = {"enable_thinking": False} if KEY.startswith("qwen3") else {}
    p = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], add_generation_prompt=True, tokenize=False, **kw)
    enc = tok(p, return_tensors="pt"); t0 = time.time()
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = o[0, enc["input_ids"].shape[1]:]
    rec = dict(id=it["id"], task=it["task"], text=tok.decode(gen, skip_special_tokens=True).strip(), gen_tokens=int(len(gen)), wall_s=round(time.time() - t0, 2))
    open(OUT, "a").write(json.dumps(rec) + "\n"); print(KEY, it["id"], rec["wall_s"], repr(rec["text"][:60]), flush=True)
