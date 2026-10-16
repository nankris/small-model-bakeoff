# STEP=answer: Grok answers the same 20 prompts (1 call each, lean flags, reasoning effort low) -> results/grok_answers.json
# STEP=judge:  Grok judges summary + tweet outputs of all 3 systems, blind (shuffled, labelled A/B/C), in one call per item -> results/judge.json
import json, os, sys, random, re, tempfile, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(__file__))
from grok_log import run_grok, QuotaHit
from tasks import items
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CSV = f"{D}/results/grok_calls.csv"
LEAN = ["--system-prompt-override", "You are a helpful assistant. Follow the output format exactly.", "--disable-web-search", "--no-subagents", "--tools", "read_file", "--reasoning-effort", "low"]
IT = items(); STEP = os.environ.get("STEP", "answer")
def answer(it):
    r = run_grok(it["prompt"], f"answer_{it['id']}", CSV, "023-answer", 1, cwd=tempfile.mkdtemp(), extra=LEAN)
    return it["id"], dict(text=r["text"].strip(), cost=r["cost_usd"], latency=r["latency_s"], exit=r["exit_code"])
def load_local(k):
    return {json.loads(l)["id"]: json.loads(l)["text"] for l in open(f"{D}/results/local_{k}.jsonl")}
def judge(it, outs):
    keys = list(outs); random.Random(it["id"]).shuffle(keys); lab = "ABC"[:len(keys)]
    src = it["source"] if it["task"] == "summary" else "\n- " + "\n- ".join(it["source"])
    crit = ("faithful to the text, one sentence, at most 25 words, clear" if it["task"] == "summary"
            else "accurate to the notes, at most 280 characters, exactly one hashtag, engaging, no invented facts")
    body = "\n\n".join(f"Candidate {l}:\n<<<{outs[k]}>>>" for l, k in zip(lab, keys))
    p = (f"You are grading {it['task']} outputs. Source:\n{src}\n\nCriteria: {crit}.\n\n{body}\n\n"
         f"Score each candidate from 1 (bad) to 5 (excellent). Reply only with lines 'A: <score>' etc.")
    r = run_grok(p, f"judge_{it['id']}", CSV, "023-judge", 1, cwd=tempfile.mkdtemp(), extra=LEAN)
    sc = {m.group(1): int(m.group(2)) for m in re.finditer(r"^\W*([ABC])\W*[:=-]\s*([1-5])", r["text"], re.M)}
    return it["id"], {k: sc.get(l) for l, k in zip(lab, keys)} | {"_raw": r["text"]}
if __name__ == "__main__":
    hit = False; res = {}
    with cf.ThreadPoolExecutor(5) as ex:
        if STEP == "answer":
            futs = [ex.submit(answer, it) for it in IT]
        else:
            G = json.load(open(f"{D}/results/grok_answers.json")); Q, S = load_local("qwen3_0.6b"), load_local("smollm2_360m")
            futs = [ex.submit(judge, it, {"grok": G[it["id"]]["text"], "qwen3_0.6b": Q[it["id"]], "smollm2_360m": S[it["id"]]})
                    for it in IT if it["task"] in ("summary", "tweet")]
        for f in cf.as_completed(futs):
            try:
                k, v = f.result(); res[k] = v
            except QuotaHit as e:
                hit = True; print("QUOTA HIT", e)
    json.dump(res, open(f"{D}/results/{'grok_answers' if STEP == 'answer' else 'judge'}.json", "w"), indent=1)
    print(STEP, len(res)); sys.exit(3 if hit else 0)
