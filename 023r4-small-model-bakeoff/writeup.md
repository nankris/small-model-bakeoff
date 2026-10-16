# 023 round 4 — Grok vs Qwen3-4B on daily text chores

**Verdict: PROVEN.** Grok superior (95% CI for Grok−Qwen lies entirely above 0).

Estimand: equal-weighted mean of the five task accuracies, Grok minus Qwen, on the preregistered synthetic items. Paired task-stratified bootstrap, 20,000 draws, seed 23023. Superiority is a 95% CI entirely above 0. Equivalence is a 90% CI entirely inside ±5 percentage points (TOST at α=0.05), and that result wins if both descriptions fit. A 95% CI entirely below 0 means Qwen is ahead, which also counts as DISPROVEN for the claim that Grok is better.

## Results

| Task | Grok | Qwen | Difference (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|
| triage | 60/60 = 100.0% [94.0, 100.0] | 45/60 = 75.0% [62.8, 84.2] | 25.0 | [15.0, 36.7] |
| summary | 49/60 = 81.7% [70.1, 89.4] | 34/60 = 56.7% [44.1, 68.4] | 25.0 | [10.0, 40.0] |
| tweet | 60/60 = 100.0% [94.0, 100.0] | 60/60 = 100.0% [94.0, 100.0] | 0.0 | [0.0, 0.0] |
| regex | 60/60 = 100.0% [94.0, 100.0] | 0/60 = 0.0% [0.0, 6.0] | 100.0 | [100.0, 100.0] |
| json | 59/60 = 98.3% [91.1, 99.7] | 26/60 = 43.3% [31.6, 55.9] | 55.0 | [43.3, 66.7] |
| **overall** | **96.0%** [94.0, 98.0] | **55.0%** [51.0, 59.0] | **41.0** | **[36.7, 45.7]** |

Overall difference 90% CI (the TOST interval): [37.3, 44.7] percentage points. Margin is ±5.

Pooled exact McNemar (descriptive, not the decision rule): Grok-only correct 129, Qwen-only correct 6, two-sided p=3.615e-31.

| Task | Grok-only | Qwen-only | McNemar p |
|---|---:|---:|---:|
| triage | 15 | 0 | 6.104e-05 |
| summary | 21 | 6 | 0.005925 |
| tweet | 0 | 0 | 1 |
| regex | 60 | 0 | 1.735e-18 |
| json | 33 | 0 | 2.328e-10 |

Per-task accuracy brackets are Wilson 95% intervals. Difference intervals are paired bootstrap percentile intervals. The overall accuracy interval is a task-stratified bootstrap of that arm alone.

Lenient scores (descriptive only: triage label contained, regex accepted on any line) are in `results/summary.json` under `lenient` and were not used for the verdict.

## Setup

Qwen checkpoint recorded on the answers: `qwen3_4b_q4` repo `Qwen/Qwen3-4B` quant `nf4`. Inherited round-3 GPU allocation: `L4`. Device string: `NVIDIA L4, 23034 MiB`. Round 4 reused every answer and completed only missing blind grades.

Round-4 CU: **0.00** (no new GPU session). Inherited round-3 estimated CU: **0.3808** (session elapsed × measured incremental usage rate, capped in the runner at 4.6). Account balance before the first session: 175.96; after the last stop: 175.43. The balance delta is not experiment-only CU when other assignments are active.

Grok answers: n=300, API-equivalent cost $2.721 ($0.0091 per answer), mean latency 6.2s.
Qwen mean generation time 1.93s per item.

## Limits

Items are a deterministic template family, not a sample of real inboxes. The bootstrap describes resampling of this bank, so the intervals are tight when a template is solved the same way every time, and they do not by themselves justify a claim about all daily work. One greedy decode per item. Summary and tweet grades are a blind Codex pass/fail plus a hard length or hashtag check; the judge is independent of both contestants and still subjective. Objective tasks use a strict extractor (one fence, one wrapping quote or slash, first line). A lenient pass is reported and does not change the verdict. Grok is `grok-4.7-build` through the Grok Build CLI with reasoning effort low, not a specially prompted API completion. If stage 2 was run, the same rule was applied once more on the pooled bank, without an alpha adjustment.

