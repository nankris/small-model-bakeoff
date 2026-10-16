import json, os, re, sys, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); from tasks import items
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IT = {i["id"]: i for i in items()}
G = {k: v["text"] for k, v in json.load(open(f"{D}/results/grok_answers.json")).items()}
L = {k: {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/results/local_{k}.jsonl")} for k in ["qwen3_0.6b", "smollm2_360m"]}
J = json.load(open(f"{D}/results/judge.json")) if os.path.exists(f"{D}/results/judge.json") else {}
SYS = {"grok": G, "qwen3_0.6b": {k: v["text"] for k, v in L["qwen3_0.6b"].items()}, "smollm2_360m": {k: v["text"] for k, v in L["smollm2_360m"].items()}}
def clean(t): return re.sub(r"^```\w*\s*|\s*```$", "", (t or "").strip()).strip()
def sc_triage(it, t): return float(clean(t).strip(" .'\"").lower() == it["gold"].lower())
def sc_regex(it, t):
    r = clean(t).splitlines()[0].strip() if clean(t) else ""
    r = re.sub(r"^r?['\"/](.*)['\"/]$", r"\1", r)
    try: rx = re.compile(r)
    except Exception: return 0.0
    ok = [bool(rx.fullmatch(s)) for s in it["pos"]] + [not rx.fullmatch(s) for s in it["neg"]]
    return float(all(ok))  # all-or-nothing: a regex that rejects everything would otherwise get 4/7 for free
def sc_json(it, t):
    m = re.search(r"\{.*\}", clean(t), re.S)
    try: o = json.loads(m.group(0)) if m else None
    except Exception: o = None
    if not isinstance(o, dict): return 0.0
    g = it["gold"]
    return sum((str(o.get(k, "")).strip().lower() == str(v).lower()) if k != "min_years" else (str(o.get(k)) == str(v)) for k, v in g.items()) / len(g)
def sc_judged(it, sysname):
    v = (J.get(it["id"]) or {}).get(sysname); return None if v is None else (v - 1) / 4  # 1..5 -> 0..1
def hard_checks(it, t):
    t = clean(t)
    if it["task"] == "summary": return float(len(t.split()) <= 25 and t.count(". ") == 0)
    if it["task"] == "tweet": return float(len(t) <= 280 and len(re.findall(r"#\w+", t)) == 1)
rows = {}
for s in SYS:
    per = {}
    for iid, it in IT.items():
        t = SYS[s].get(iid, "")
        if it["task"] == "triage": v = sc_triage(it, t)
        elif it["task"] == "regex": v = sc_regex(it, t)
        elif it["task"] == "json": v = sc_json(it, t)
        else: v = sc_judged(it, s)
        per[iid] = dict(score=v, format_ok=hard_checks(it, t) if it["task"] in ("summary", "tweet") else None)
    rows[s] = per
TASKS = ["triage", "summary", "tweet", "regex", "json"]
summ = {s: {tk: st.mean([v["score"] for k, v in rows[s].items() if IT[k]["task"] == tk and v["score"] is not None] or [float("nan")]) for tk in TASKS} for s in SYS}
for s in SYS:
    summ[s]["format_ok_summary_tweet"] = st.mean([v["format_ok"] for k, v in rows[s].items() if v["format_ok"] is not None])
    summ[s]["overall"] = st.mean([summ[s][t] for t in TASKS])
for k in L:
    summ[k]["sec_per_item"] = st.mean(v["wall_s"] for v in L[k].values()); summ[k]["tok_per_s"] = sum(v["gen_tokens"] for v in L[k].values()) / sum(v["wall_s"] for v in L[k].values())
import csv
C = list(csv.DictReader(open(f"{D}/results/grok_calls.csv")))
ans = [c for c in C if c["experiment"] == "023-answer"]; jud = [c for c in C if c["experiment"] == "023-judge"]
summ["grok"]["usd_per_item"] = sum(float(c["cost_usd"] or 0) for c in ans) / max(1, len(ans)); summ["grok"]["sec_per_item"] = st.mean(float(c["latency_s"]) for c in ans)
summ["_judge_usd_total"] = sum(float(c["cost_usd"] or 0) for c in jud); summ["_answer_usd_total"] = sum(float(c["cost_usd"] or 0) for c in ans)
json.dump(dict(summary=summ, per_item=rows), open(f"{D}/results/summary.json", "w"), indent=1)
for s in SYS: print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ[s].items()})
print("grok usd answers", summ["_answer_usd_total"], "judge", summ["_judge_usd_total"])
fig, ax = plt.subplots(figsize=(11, 5)); w = .26; col = {"grok": "#4C72B0", "qwen3_0.6b": "#DD8452", "smollm2_360m": "#55A868"}
lab = {"grok": "Grok CLI (reference + judge)", "qwen3_0.6b": "Qwen3-0.6B (CPU)", "smollm2_360m": "SmolLM2-360M (CPU)"}
names = ["newsletter\ntriage", "one-line\nsummary*", "tweet\ndraft*", "regex\n(passes all 7 tests)", "JSON\nextraction", "overall"]
for j, s in enumerate(SYS):
    vals = [summ[s][t] * 100 for t in TASKS] + [summ[s]["overall"] * 100]
    ax.bar([i + (j - 1) * w for i in range(6)], vals, w, color=col[s], label=lab[s])
    for i, v in enumerate(vals): ax.text(i + (j - 1) * w, v + 1, f"{v:.0f}", ha="center", fontsize=7)
ax.set_xticks(range(6)); ax.set_xticklabels(names); ax.set_ylim(0, 125); ax.set_ylabel("score (%)  - 4 items per task")
ax.set_title("#023 Small-model bake-off on daily tasks (*judged 1-5 by Grok, blind, rescaled to 0-100)"); ax.legend(fontsize=8, loc="upper right", ncol=3)
plt.tight_layout(); plt.savefig(f"{D}/chart.png", dpi=125)
