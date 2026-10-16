# 023r bakeoff v2 — blocked rerun

Verdict: **STILL INCONCLUSIVE**. Colab refused L4 allocation on all three attempts (precondition failed). No local model ran, so the preregistered paired comparison is unavailable. Zero evaluation rounds completed; this is an infrastructure failure, not a two-round null finding.

## Partial diagnostic results

150 synthetic items were prepared (30/task). Only 30 Grok calls completed before cancellation. These partial results do not satisfy the requested >=25 scored items/task and must not be interpreted as a model comparison.

| Task | Grok successes / completed | Accuracy, Wilson 95% CI | SmolLM2-360M | Qwen3-0.6B | Qwen3-4B Q4 |
|---|---:|---|---|---|---|
| triage | 6/6 | 100.0% [61.0, 100.0] | Not run | Not run | Not run |
| summary | 3/6 | 50.0% [18.8, 81.2] | Not run | Not run | Not run |
| tweet | 6/6 | 100.0% [61.0, 100.0] | Not run | Not run | Not run |
| regex | 6/6 | 100.0% [61.0, 100.0] | Not run | Not run | Not run |
| json | 6/6 | 100.0% [61.0, 100.0] | Not run | Not run | Not run |

Primary Grok minus Qwen3-4B difference and 95% CI: **not estimable**. No scores have been imputed for missing arms. Summary/tweet grades come from independent blind Codex exec, not Grok. Raw anonymous judge inputs, grades and identity mapping are retained. The table describes only completed calls and is not a random subsample analysis.

## Compute and artifacts

Requested GPU: L4; actual GPU: none. CU used: **0.00** (account balance 179.84 before / 179.58 at final cleanup; no experiment runtime was allocated). Cleanup called `colab stop -s tig023r-v2`; session not found. Other active sessions left untouched. See results/compute.json and README.md for failure details and resume commands.

The original folder was preserved. New scripts include deterministic synthetic item generation, resumable Grok calls, planned 4-bit NF4 Qwen3-4B and FP16 tiny-model L4 inference, blind Codex judging, per-task Wilson CIs, and paired stratified bootstrap analysis. Remote inference is unvalidated because allocation failed. Figure explicitly marks partial sample sizes and missing arms. No API dollar cost estimate was collected.

## Limits

Template-based synthetic items are not representative of all daily tasks; template correlation makes item-level CIs optimistic for broader generalization. Deterministic generations estimate item variation only. Strict binary rubric is not numerically comparable to the original partial-credit scoring. Independent judging remains subjective. No inference about quantization, speed, RAM, model size or superiority is supported by this incomplete run.
