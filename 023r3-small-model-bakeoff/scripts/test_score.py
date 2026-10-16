#!/usr/bin/env python3
"""Unit checks for scoring and the decision rule. Run before the experiment."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_items import make
from score import (
    lenient_regex,
    lenient_triage,
    normalize,
    score_json,
    score_regex,
    score_triage,
    verdict_from_intervals,
)

items = {r["id"]: r for r in make(3, 0)}
t0 = items["triage0"]
assert t0["gold"] == "AI research"
assert score_triage("AI research", t0) == 1
assert score_triage("AI research.", t0) == 1
assert score_triage("```\nAI research\n```", t0) == 1
assert score_triage("<think>hmm</think>\nAI research", t0) == 1
assert score_triage("Job alert", t0) == 0
assert score_triage("Category: AI research", t0) == 0
assert lenient_triage("Category: AI research", t0) == 1
assert lenient_triage("AI research or Job alert", t0) == 0

r0 = items["regex0"]
assert score_regex(r"AB-[0-9]{2}", r0) == 1
assert score_regex(r"\AAB-[0-9]{2}\Z", r0) == 1
assert score_regex(r"/AB-[0-9]{2}/", r0) == 1
assert score_regex("```\nAB-[0-9]{2}\n```", r0) == 1
assert score_regex(r"\d{2}", r0) == 0
assert score_regex(r"AB-\d{2}", r0) == 0  # Unicode \d matches Arabic-Indic digits in the negative set
assert lenient_regex("here you go\nAB-[0-9]{2}", r0) == 1
assert score_regex("here you go\nAB-[0-9]{2}", r0) == 0

j0 = items["json0"]
good = json.dumps(j0["gold"])
assert score_json(good, j0) == 1
assert score_json("```json\n" + good + "\n```", j0) == 1
bad = dict(j0["gold"])
bad["city"] = "Mumbai"
assert score_json(json.dumps(bad), j0) == 0

assert normalize("<think>x</think>\n\nhello") == "hello"

# Decision priority.
assert verdict_from_intervals((0.06, 0.20), (0.08, 0.18))[0] == "PROVEN"
assert verdict_from_intervals((-0.02, 0.02), (-0.01, 0.01))[0] == "DISPROVEN"
assert verdict_from_intervals((-0.12, -0.02), (-0.10, -0.03))[0] == "DISPROVEN"
assert verdict_from_intervals((-0.02, 0.09), (-0.01, 0.08))[0] == "STILL INCONCLUSIVE"
# Significant but inside the margin: equivalence wins.
assert verdict_from_intervals((0.01, 0.04), (0.012, 0.038))[0] == "DISPROVEN"
# 90% CI crosses the margin, 95% CI excludes 0: superiority.
assert verdict_from_intervals((0.01, 0.08), (0.02, 0.07))[0] == "PROVEN"

print("score and decision tests passed")
