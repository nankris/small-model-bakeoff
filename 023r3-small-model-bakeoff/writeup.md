# 023 round 3 — Grok vs Qwen3-4B on daily text chores

**Verdict: STILL INCONCLUSIVE.** budget or runtime ended before every item had both answers and, where required, a judge grade.

Estimand: equal-weighted mean of the five task accuracies, Grok minus Qwen, on the preregistered synthetic items. Paired task-stratified bootstrap, 20,000 draws, seed 23023. Superiority is a 95% CI entirely above 0. Equivalence is a 90% CI entirely inside ±5 percentage points (TOST at α=0.05), and that result wins if both descriptions fit. A 95% CI entirely below 0 means Qwen is ahead, which also counts as DISPROVEN for the claim that Grok is better.

The decision rule was not applied. Missing pieces: 240 (showing up to 40): judge:grok:summary0, judge:qwen:summary0, judge:grok:tweet0, judge:qwen:tweet0, judge:grok:summary1, judge:qwen:summary1, judge:grok:tweet1, judge:qwen:tweet1, judge:grok:summary2, judge:qwen:summary2, judge:grok:tweet2, judge:qwen:tweet2.

## Setup

Qwen checkpoint recorded on the answers: `qwen3_4b_q4` repo `Qwen/Qwen3-4B` quant `nf4`. GPU allocation: `L4`. Device string: `NVIDIA L4, 23034 MiB`.

Estimated experiment CU: **0.3808** (session elapsed × measured incremental usage rate, capped in the runner at 4.6). Account balance before the first session: 175.96; after the last stop: 175.43. The balance delta is not experiment-only CU when other assignments are active.

Grok answers: n=300, API-equivalent cost $2.721 ($0.0091 per answer), mean latency 6.2s.
Qwen mean generation time 1.93s per item.

## Limits

Items are a deterministic template family, not a sample of real inboxes. The bootstrap describes resampling of this bank, so the intervals are tight when a template is solved the same way every time, and they do not by themselves justify a claim about all daily work. One greedy decode per item. Summary and tweet grades are a blind Codex pass/fail plus a hard length or hashtag check; the judge is independent of both contestants and still subjective. Objective tasks use a strict extractor (one fence, one wrapping quote or slash, first line). A lenient pass is reported and does not change the verdict. Grok is `grok-4.7-build` through the Grok Build CLI with reasoning effort low, not a specially prompted API completion. If stage 2 was run, the same rule was applied once more on the pooled bank, without an alpha adjustment.

