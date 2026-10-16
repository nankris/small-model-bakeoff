#!/usr/bin/env python3
"""Preregistered scoring and the two-sided decision rule.

Primary item score is binary. Objective tasks are code-checked. Summary and
tweet require a blind judge pass AND the hard format check from the v2 analyser.
"""
import json
import math
import re

import numpy as np

TASKS = ["triage", "summary", "tweet", "regex", "json"]
MODELS_WANTED = ("grok", "qwen")
DELTA = 0.05  # equivalence margin, proportion points
BOOT_B = 20000
BOOT_SEED = 23023
HASHTAG_RE = re.compile(r"#[A-Za-z][A-Za-z0-9_]*")
LABELS = ("AI research", "Job alert", "Promotion", "Product update")


def normalize(text):
    """Strip a leading think block and one surrounding markdown fence.

    Applied to every model before scoring and before the judge sees the text.
    """
    s = (text or "").strip()
    s = re.sub(r"(?is)^<think>.*?</think>\s*", "", s)
    s = re.sub(r"(?is)^◁think▷.*?◁/think▷\s*", "", s)
    s = s.strip()
    s = re.sub(r"^```[A-Za-z0-9]*[ \t]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    return s.strip()


def score_triage(text, item):
    s = normalize(text)
    line = s.splitlines()[0].strip() if s else ""
    line = line.strip(" .\"'`")
    return int(line == item["gold"])


def score_regex(text, item):
    s = normalize(text)
    line = s.splitlines()[0].strip() if s else ""
    line = re.sub(r"^r?['\"/](.*?)['\"/]$", r"\1", line)
    try:
        ok_pos = all(re.fullmatch(line, x) is not None for x in item["pos"])
        ok_neg = all(re.fullmatch(line, x) is None for x in item["neg"])
        return int(ok_pos and ok_neg)
    except re.error:
        return 0


def score_json(text, item):
    s = normalize(text)
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    return int(obj == item["gold"])


def format_ok(task, text):
    s = normalize(text)
    if task == "summary":
        return int(len(s.split()) <= 25 and s != "")
    if task == "tweet":
        return int(len(s) <= 280 and len(HASHTAG_RE.findall(s)) == 1)
    return 1


def lenient_triage(text, item):
    s = normalize(text)
    hits = [lab for lab in LABELS if lab.lower() in s.lower()]
    return int(hits == [item["gold"]])


def lenient_regex(text, item):
    s = normalize(text)
    candidates = [s] + s.splitlines()
    for cand in candidates:
        line = cand.strip()
        line = re.sub(r"^r?['\"/](.*?)['\"/]$", r"\1", line)
        if not line:
            continue
        try:
            ok = all(re.fullmatch(line, x) is not None for x in item["pos"]) and all(
                re.fullmatch(line, x) is None for x in item["neg"]
            )
        except re.error:
            continue
        if ok:
            return 1
    return 0


def wilson(k, n, z=1.959963984540054):
    if n <= 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def id_num(i):
    return int(re.search(r"(\d+)$", i).group(1))


def verdict_from_intervals(ci95, ci90, delta=DELTA):
    """Return (verdict, reason).

    Priority is fixed:
    1. Equivalence (TOST α=0.05) if the 90% CI lies strictly inside ±delta.
    2. Else superiority of Grok if the 95% CI lies strictly above 0.
    3. Else superiority of Qwen if the 95% CI lies strictly below 0.
    4. Else inconclusive.
    """
    lo95, hi95 = ci95
    lo90, hi90 = ci90
    if lo90 > -delta and hi90 < delta:
        return (
            "DISPROVEN",
            "equivalent within ±5 percentage points (TOST α=0.05: 90% CI inside the margin)",
        )
    if lo95 > 0:
        return ("PROVEN", "Grok superior (95% CI for Grok−Qwen lies entirely above 0)")
    if hi95 < 0:
        return ("DISPROVEN", "Qwen superior (95% CI for Grok−Qwen lies entirely below 0)")
    return (
        "STILL INCONCLUSIVE",
        "95% CI covers 0 or a boundary, and the 90% CI is not inside ±5 points",
    )


def stratified_bootstrap(diffs_by_task, b=BOOT_B, seed=BOOT_SEED):
    """diffs_by_task: task -> 1d array of paired (grok - qwen) item differences.

    Returns overall point, ci95, ci90, per-task point/ci95, and the overall draws.
    """
    rng = np.random.default_rng(seed)
    task_draws = []
    per_task = {}
    for t in TASKS:
        d = np.asarray(diffs_by_task[t], dtype=float)
        n = len(d)
        if n == 0:
            raise ValueError(f"no items for {t}")
        ix = rng.integers(0, n, size=(b, n))
        draws = d[ix].mean(axis=1)
        task_draws.append(draws)
        per_task[t] = {
            "n": int(n),
            "difference": float(d.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "ci90": np.quantile(draws, [0.05, 0.95]).tolist(),
        }
    overall = np.mean(task_draws, axis=0)
    point = float(np.mean([per_task[t]["difference"] for t in TASKS]))
    return {
        "difference": point,
        "ci95": np.quantile(overall, [0.025, 0.975]).tolist(),
        "ci90": np.quantile(overall, [0.05, 0.95]).tolist(),
        "per_task": per_task,
        "draws": overall,
    }


def mcnemar_exact(b, c):
    """Two-sided exact McNemar. b = Grok only correct, c = Qwen only correct."""
    from scipy.stats import binomtest

    n = int(b + c)
    if n == 0:
        return 1.0
    return float(binomtest(int(b), n, 0.5, alternative="two-sided").pvalue)
