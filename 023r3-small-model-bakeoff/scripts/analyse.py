#!/usr/bin/env python3
"""Score a complete item file and apply the preregistered decision rule.

Usage: analyse.py [items.json]
Incomplete arms do not get a superiority or equivalence call.
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score import (
    TASKS,
    format_ok,
    lenient_regex,
    lenient_triage,
    mcnemar_exact,
    score_json,
    score_regex,
    score_triage,
    stratified_bootstrap,
    verdict_from_intervals,
    wilson,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def arm_text(rows, kind):
    out = {}
    meta = None
    for rec in rows:
        model = str(rec.get("model", ""))
        if kind == "grok" and model != "grok":
            continue
        if kind == "qwen" and not model.startswith("qwen"):
            continue
        if rec.get("exit", 0) != 0 or rec.get("text") is None or rec.get("text") == "":
            continue
        out[rec["id"]] = rec["text"]
        meta = rec
    return out, meta


def judge_map():
    grades = {}
    for rec in load_jsonl(ROOT / "results" / "judge.jsonl"):
        grades[(rec["model"], rec["id"])] = rec
    return grades


def primary(task, text, item, model_key, grades):
    if task == "triage":
        return score_triage(text, item), ""
    if task == "regex":
        return score_regex(text, item), ""
    if task == "json":
        return score_json(text, item), ""
    g = grades.get((model_key, item["id"]))
    if g is None:
        return None, "missing judge"
    ok = int(bool(g["pass"]) and format_ok(task, text))
    return ok, g.get("reason", "")


def lenient(task, text, item, primary_bit):
    if task == "triage":
        return lenient_triage(text, item)
    if task == "regex":
        return lenient_regex(text, item)
    return primary_bit


def pct(x):
    return f"{100 * x:.1f}"


def main():
    items_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "items.json"
    if not items_path.is_absolute():
        items_path = ROOT / items_path
    items = json.loads(items_path.read_text())
    grok, _ = arm_text(load_jsonl(ROOT / "results" / "grok.jsonl"), "grok")
    qwen, qmeta = arm_text(load_jsonl(ROOT / "results" / "qwen.jsonl"), "qwen")
    grades = judge_map()
    qname = (qmeta or {}).get("model", "qwen")
    repo = (qmeta or {}).get("repo", "")
    quant = (qmeta or {}).get("quant", "")

    detailed = []
    missing = []
    for it in items:
        for key, pool in (("grok", grok), ("qwen", qwen)):
            if it["id"] not in pool:
                missing.append(f"{key}:{it['id']}")
                continue
            bit, why = primary(it["task"], pool[it["id"]], it, key, grades)
            if bit is None:
                missing.append(f"judge:{key}:{it['id']}")
                continue
            detailed.append({
                "model": key,
                "id": it["id"],
                "task": it["task"],
                "success": int(bit),
                "lenient": int(lenient(it["task"], pool[it["id"]], it, bit) or 0),
                "reason": why,
                "checkpoint": qname if key == "qwen" else "grok",
            })

    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "scores.json").write_text(json.dumps(detailed, indent=2) + "\n")

    compute = {}
    if (ROOT / "results" / "compute.json").exists():
        compute = json.loads((ROOT / "results" / "compute.json").read_text())

    by = {}
    for row in detailed:
        by[(row["model"], row["id"])] = row

    complete = not missing and len(items) > 0
    per_task = {m: {} for m in ("grok", "qwen")}
    diffs = {}
    mcnemar = {}
    if complete:
        for t in TASKS:
            ids = [it["id"] for it in items if it["task"] == t]
            ids.sort(key=lambda s: int(s[len(t):]))
            for m in ("grok", "qwen"):
                bits = np.array([by[(m, i)]["success"] for i in ids], dtype=float)
                per_task[m][t] = {
                    "n": len(ids),
                    "success": int(bits.sum()),
                    "score": float(bits.mean()),
                    "ci": wilson(int(bits.sum()), len(ids)),
                    "lenient": float(np.mean([by[(m, i)]["lenient"] for i in ids])),
                }
            d = np.array([by[("grok", i)]["success"] - by[("qwen", i)]["success"] for i in ids], dtype=float)
            diffs[t] = d
            b = int(np.sum(d == 1))
            c = int(np.sum(d == -1))
            mcnemar[t] = {"grok_only": b, "qwen_only": c, "p": mcnemar_exact(b, c)}
        boot = stratified_bootstrap(diffs)
        verdict, vreason = verdict_from_intervals(boot["ci95"], boot["ci90"])
        b_all = sum(m["grok_only"] for m in mcnemar.values())
        c_all = sum(m["qwen_only"] for m in mcnemar.values())
        mcnemar_all = {"grok_only": b_all, "qwen_only": c_all, "p": mcnemar_exact(b_all, c_all)}
        overall = {}
        for m in ("grok", "qwen"):
            # Equal-weighted task mean. CI from the same stratified draws is built below for the difference.
            scores = [per_task[m][t]["score"] for t in TASKS]
            overall[m] = {"score": float(np.mean(scores))}
        # Model-level CIs: bootstrap each arm the same way as the difference.
        # Recompute from successes so the writeup can show them.
        rng = np.random.default_rng(23023)
        arm_draws = {m: [] for m in ("grok", "qwen")}
        for t in TASKS:
            ids = [it["id"] for it in items if it["task"] == t]
            ids.sort(key=lambda s: int(s[len(t):]))
            n = len(ids)
            ix = rng.integers(0, n, size=(20000, n))
            for m in ("grok", "qwen"):
                a = np.array([by[(m, i)]["success"] for i in ids], dtype=float)
                arm_draws[m].append(a[ix].mean(axis=1))
        for m in ("grok", "qwen"):
            draws = np.mean(arm_draws[m], axis=0)
            overall[m]["ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
        rule_applied = True
    else:
        boot = None
        verdict = "STILL INCONCLUSIVE"
        vreason = "budget or runtime ended before every item had both answers and, where required, a judge grade"
        mcnemar_all = None
        overall = {}
        rule_applied = False

    # Costs and speed from raw logs.
    g_rows = [r for r in load_jsonl(ROOT / "results" / "grok.jsonl") if r.get("model") == "grok" and r.get("text")]
    q_rows = [r for r in load_jsonl(ROOT / "results" / "qwen.jsonl") if str(r.get("model", "")).startswith("qwen") and r.get("text")]
    g_cost = sum(float(r["cost_usd"]) for r in g_rows if r.get("cost_usd") is not None)
    q_time = [float(r["wall_s"]) for r in q_rows if r.get("wall_s")]
    g_time = [float(r["wall_s"]) for r in g_rows if r.get("wall_s")]

    summary = {
        "items_file": items_path.name,
        "n_items": len(items),
        "n_per_task_intended": len(items) // 5,
        "complete": complete,
        "missing": missing[:40],
        "n_missing": len(missing),
        "rule_applied": rule_applied,
        "verdict": verdict,
        "verdict_reason": vreason,
        "qwen_checkpoint": qname,
        "qwen_repo": repo,
        "qwen_quant": quant,
        "per_task": per_task,
        "overall": overall,
        "comparison": None if boot is None else {
            "difference": boot["difference"],
            "ci95": boot["ci95"],
            "ci90": boot["ci90"],
            "per_task": boot["per_task"],
        },
        "mcnemar_pooled": mcnemar_all,
        "mcnemar_per_task": mcnemar if complete else {},
        "grok_cost_usd": g_cost,
        "grok_n": len(g_rows),
        "qwen_mean_wall_s": float(np.mean(q_time)) if q_time else None,
        "grok_mean_wall_s": float(np.mean(g_time)) if g_time else None,
        "compute": compute,
    }
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_chart(summary)
    write_writeup(summary)
    print(json.dumps({k: summary[k] for k in ("verdict", "verdict_reason", "complete", "n_missing", "qwen_checkpoint")}, indent=2))
    return 0


def write_chart(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    if summary["complete"]:
        x = np.arange(len(TASKS) + 1)
        w = 0.36
        labels = TASKS + ["overall"]
        for j, (m, color) in enumerate((("grok", "#1f4e79"), ("qwen", "#e07a3d"))):
            vals, los, his = [], [], []
            for t in TASKS:
                cell = summary["per_task"][m][t]
                vals.append(100 * cell["score"])
                los.append(100 * cell["ci"][0])
                his.append(100 * cell["ci"][1])
            vals.append(100 * summary["overall"][m]["score"])
            ci = summary["overall"][m]["ci95"]
            los.append(100 * ci[0])
            his.append(100 * ci[1])
            vals = np.array(vals)
            yerr = np.vstack([vals - np.array(los), np.array(his) - vals])
            name = "Grok" if m == "grok" else summary["qwen_checkpoint"]
            ax.bar(x + (j - 0.5) * w, vals, w, label=name, color=color, yerr=yerr, capsize=2, ecolor="#333333")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Accuracy (%)")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Task accuracy, 95% CI")
    else:
        ax.text(0.5, 0.5, "Incomplete sample\nno decision chart", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    if summary["complete"]:
        comp = summary["comparison"]
        rows = [(t, comp["per_task"][t]) for t in TASKS] + [("overall", comp)]
        ys = np.arange(len(rows))[::-1]
        ax.axvspan(-5, 5, color="#d9ead3", alpha=0.9, label="±5 pp equivalence")
        ax.axvline(0, color="#666666", lw=1)
        for y, (name, cell) in zip(ys, rows):
            est = 100 * cell["difference"]
            lo, hi = 100 * cell["ci95"][0], 100 * cell["ci95"][1]
            ax.plot([lo, hi], [y, y], color="#1f4e79", lw=2)
            ax.plot([est], [y], "o", color="#1f4e79")
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.set_xlabel("Grok − Qwen (percentage points)")
        ax.set_title("Paired difference, 95% bootstrap CI")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    else:
        ax.set_axis_off()
    n = summary["n_per_task_intended"]
    fig.suptitle(f"023 round 3 — {summary['verdict']}  ({n} items/task, {summary['qwen_checkpoint']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "chart.png", dpi=150)
    plt.close()


def write_writeup(summary):
    cu = summary.get("compute") or {}
    gpu = cu.get("gpu_allocated")
    meta = cu.get("qwen_meta") or {}
    gpu_name = meta.get("gpu", "")
    lines = []
    lines.append("# 023 round 3 — Grok vs Qwen3-4B on daily text chores")
    lines.append("")
    lines.append(f"**Verdict: {summary['verdict']}.** {summary['verdict_reason']}.")
    lines.append("")
    lines.append(
        "Estimand: equal-weighted mean of the five task accuracies, Grok minus Qwen, "
        "on the preregistered synthetic items. Paired task-stratified bootstrap, 20,000 draws, seed 23023. "
        "Superiority is a 95% CI entirely above 0. Equivalence is a 90% CI entirely inside ±5 percentage points "
        "(TOST at α=0.05), and that result wins if both descriptions fit. "
        "A 95% CI entirely below 0 means Qwen is ahead, which also counts as DISPROVEN for the claim that Grok is better."
    )
    lines.append("")
    if not summary["complete"]:
        lines.append(
            f"The decision rule was not applied. Missing pieces: {summary['n_missing']} "
            f"(showing up to 40): {', '.join(summary['missing'][:12]) or 'n/a'}."
        )
        lines.append("")
    else:
        comp = summary["comparison"]
        lines.append("## Results")
        lines.append("")
        lines.append("| Task | Grok | Qwen | Difference (pp) | 95% CI (pp) |")
        lines.append("|---|---:|---:|---:|---:|")
        for t in TASKS:
            g = summary["per_task"]["grok"][t]
            q = summary["per_task"]["qwen"][t]
            d = comp["per_task"][t]
            lines.append(
                f"| {t} | {g['success']}/{g['n']} = {pct(g['score'])}% "
                f"[{pct(g['ci'][0])}, {pct(g['ci'][1])}] | "
                f"{q['success']}/{q['n']} = {pct(q['score'])}% "
                f"[{pct(q['ci'][0])}, {pct(q['ci'][1])}] | "
                f"{100*d['difference']:.1f} | [{100*d['ci95'][0]:.1f}, {100*d['ci95'][1]:.1f}] |"
            )
        g, q = summary["overall"]["grok"], summary["overall"]["qwen"]
        lines.append(
            f"| **overall** | **{pct(g['score'])}%** [{pct(g['ci95'][0])}, {pct(g['ci95'][1])}] | "
            f"**{pct(q['score'])}%** [{pct(q['ci95'][0])}, {pct(q['ci95'][1])}] | "
            f"**{100*comp['difference']:.1f}** | **[{100*comp['ci95'][0]:.1f}, {100*comp['ci95'][1]:.1f}]** |"
        )
        lines.append("")
        lines.append(
            f"Overall difference 90% CI (the TOST interval): "
            f"[{100*comp['ci90'][0]:.1f}, {100*comp['ci90'][1]:.1f}] percentage points. "
            f"Margin is ±5."
        )
        lines.append("")
        mp = summary["mcnemar_pooled"]
        lines.append(
            f"Pooled exact McNemar (descriptive, not the decision rule): "
            f"Grok-only correct {mp['grok_only']}, Qwen-only correct {mp['qwen_only']}, "
            f"two-sided p={mp['p']:.4g}."
        )
        lines.append("")
        lines.append("| Task | Grok-only | Qwen-only | McNemar p |")
        lines.append("|---|---:|---:|---:|")
        for t in TASKS:
            m = summary["mcnemar_per_task"][t]
            lines.append(f"| {t} | {m['grok_only']} | {m['qwen_only']} | {m['p']:.4g} |")
        lines.append("")
        lines.append("Per-task accuracy brackets are Wilson 95% intervals. Difference intervals are paired bootstrap percentile intervals. The overall accuracy interval is a task-stratified bootstrap of that arm alone.")
        lines.append("")
        lines.append("Lenient scores (descriptive only: triage label contained, regex accepted on any line) are in `results/summary.json` under `lenient` and were not used for the verdict.")
        lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"Qwen checkpoint recorded on the answers: `{summary['qwen_checkpoint']}` "
        f"repo `{summary['qwen_repo']}` quant `{summary['qwen_quant']}`. "
        f"GPU allocation: `{gpu}`. Device string: `{gpu_name}`."
    )
    lines.append("")
    lines.append(
        f"Estimated experiment CU: **{cu.get('cu_estimated_total', 'n/a')}** "
        f"(session elapsed × measured incremental usage rate, capped in the runner at 4.6). "
        f"Account balance before the first session: {cu.get('balance_before_first')}; "
        f"after the last stop: {cu.get('balance_after_last')}. "
        f"The balance delta is not experiment-only CU when other assignments are active."
    )
    lines.append("")
    if summary.get("grok_n"):
        lines.append(
            f"Grok answers: n={summary['grok_n']}, API-equivalent cost ${summary['grok_cost_usd']:.3f} "
            f"(${summary['grok_cost_usd']/max(1, summary['grok_n']):.4f} per answer), "
            f"mean latency {summary['grok_mean_wall_s']:.1f}s."
        )
        if summary.get("qwen_mean_wall_s") is not None:
            lines.append(f"Qwen mean generation time {summary['qwen_mean_wall_s']:.2f}s per item.")
        lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append(
        "Items are a deterministic template family, not a sample of real inboxes. "
        "The bootstrap describes resampling of this bank, so the intervals are tight when a template is solved the same way every time, and they do not by themselves justify a claim about all daily work. "
        "One greedy decode per item. Summary and tweet grades are a blind Codex pass/fail plus a hard length or hashtag check; the judge is independent of both contestants and still subjective. "
        "Objective tasks use a strict extractor (one fence, one wrapping quote or slash, first line). "
        "A lenient pass is reported and does not change the verdict. "
        "Grok is `grok-4.7-build` through the Grok Build CLI with reasoning effort low, not a specially prompted API completion. "
        "If stage 2 was run, the same rule was applied once more on the pooled bank, without an alpha adjustment."
    )
    lines.append("")
    (ROOT / "writeup.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
