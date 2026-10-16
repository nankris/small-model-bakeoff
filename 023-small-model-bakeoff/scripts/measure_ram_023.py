# Peak RAM (max RSS) of each local model: load fp32 on CPU + one greedy generation (first prompt, 120 new tokens).
# Each model runs in its own child process so peaks don't mix. Writes results/ram.json.
# On the shared sandbox the all-in-one parent mode exceeded the 120 s shell limit, so the children were run one at a time:
#   python scripts/measure_ram_023.py smollm2_360m ; python scripts/measure_ram_023.py qwen3_0.6b   (outputs collected into results/ram.json)
import json, os, sys, subprocess, resource
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) > 1:
    import time, torch; from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path.insert(0, os.path.dirname(__file__)); from tasks import items; 
    W = "/workspace/projects/4e0e7b68-80db-40d3-a6d5-b21cf99d82b6/work"; PATHS = {"qwen3_0.6b": f"{W}/Qwen3-0.6B", "smollm2_360m": f"{W}/SmolLM2-360M-Instruct"}  # same as run_023_local.py
    key = sys.argv[1]; torch.set_num_threads(2); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(PATHS[key]); m = AutoModelForCausalLM.from_pretrained(PATHS[key], dtype=torch.float32).eval()
    load_s = time.time() - t0; after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    kw = {"enable_thinking": False} if key.startswith("qwen3") else {}
    p = tok.apply_chat_template([{"role": "user", "content": items()[0]["prompt"]}], add_generation_prompt=True, tokenize=False, **kw)
    with torch.no_grad(): m.generate(**tok(p, return_tensors="pt"), max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
    print(json.dumps(dict(model=key, params_m=round(sum(x.numel() for x in m.parameters()) / 1e6, 1), load_s=round(load_s, 1),
                          rss_after_load_mb=round(after_load), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))))
else:
    out = [json.loads(subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
           for k in ["smollm2_360m", "qwen3_0.6b"]]
    json.dump(out, open(f"{D}/results/ram.json", "w"), indent=1); print(out)
