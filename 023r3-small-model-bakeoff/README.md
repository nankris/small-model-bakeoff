# 023 round 3 — Does Qwen3-4B close the gap with Grok on daily text chores?

Preregistered before any scored generation. GPU type and CU are filled in after the run in the section **Compute (actual)**.

## Hypothesis

On this fixed bank of synthetic daily chores, Grok (`grok-4.7-build` via the Grok Build CLI, reasoning effort low) has a higher equal-weighted task accuracy than Qwen3-4B (NF4, greedy, thinking off). Round 1 already showed a large gap against sub-1B models. The open question is whether a 4B model makes that gap go away.

Primary estimand Δ: the unweighted mean of the five task-level differences (Grok accuracy minus Qwen accuracy). Items are paired. Each task counts the same, regardless of which templates are easy.

## What changed, and why

Round 1 (`../023-small-model-bakeoff`) scored 4 hand-written items per task. Grok was 92%, SmolLM2-360M 44%, Qwen3-0.6B 40%. Grok graded its own free-text answers. There was no interval and no way to call a tie.

Round 2 (`../023r-small-model-bakeoff-v2`) preregistered 30 items per task, four model arms, and a paired bootstrap that could call Grok better or Grok worse, but not "no practical difference". Colab refused an L4 on every attempt. No local model ran. The verdict stayed inconclusive because of the GPU, not because of a small effect.

This round changes three things:

1. **Only Grok vs Qwen3-4B.** The sub-1B models already lost. Dropping them keeps the GPU job inside 5 CU and puts the whole sample on the comparison that was never run.
2. **60 items per task from the start** (the v2 maximum), with one preregistered extension to 90 if the rule below is inconclusive and compute remains. The generator is the v2 generator, indices 0–59, then 60–89 only if needed. Stimuli are frozen in `items.json`, `items_ext.json`, and `items_all.json`.
3. **An equivalence outcome.** A 95% interval that excludes 0 is not the only conclusive result. A difference inside ±5 percentage points is DISPROVEN (equivalent within the margin). v2 could not say that.

Scoring changes, both applied to every model:

- A leading `<think>` block and one surrounding markdown fence are removed before scoring and before the judge sees the text. Qwen3 emits think text unless thinking is disabled; a leaked block should not zero an otherwise right answer.
- Objective tasks follow the executed round-1 extractor more than the never-run v2 strict match: triage is the first line after stripping a trailing period or quote; a regex may sit inside one pair of slashes or quotes; JSON may sit in a fence. The pass is still all-or-nothing.
- Summary and tweet still use the v2 rule: a blind Codex pass/fail (same rubric words) AND the hard check (summary ≤25 words, tweet ≤280 characters and exactly one hashtag). Codex does not see model names. Grok is not the judge.

A lenient score (gold label merely contained; regex accepted on any line) is stored and is not used for the verdict.

## Decision rule (fixed before the run)

Bootstrap: resample items with replacement inside each task, same indices for both models, average the five task differences. 20,000 replicates. Seed 23023.

Let the 95% percentile interval be the superiority interval and the 90% percentile interval be the TOST interval. Margin δ = 0.05 (5 percentage points).

Apply in this order:

1. **DISPROVEN (equivalent within the margin)** if the 90% interval lies strictly inside (−5, +5) percentage points. This is a two one-sided test at α = 0.05. It takes priority when a difference is detectable but smaller than the margin.
2. **PROVEN (Grok superior)** if the 95% interval lies strictly above 0.
3. **DISPROVEN (Qwen superior)** if the 95% interval lies strictly below 0.
4. **STILL INCONCLUSIVE** otherwise.

The 5-point margin is the practical threshold, not a fit to the data. Round 1's gap was about 50 points. Five points is 3 items per 60, smaller than the noise that made the n=4 study swing by 25 points when one item flipped. It is also the point below which the API-equivalent cost from round 1 (about a third of a cent per item, higher in a CLI smoke check here) is hard to justify against a free local 4B model for these chores.

The rule is applied only when every item in the analysed bank has both answers and, for summary and tweet, a judge grade. A partial sample is not used to declare PROVEN or DISPROVEN. If the run stops early because the CU cap or the wall clock is hit, the verdict is STILL INCONCLUSIVE because the budget ran out.

Stage 2, only if stage 1 is inconclusive **and** the rule was applied on the full n=60 bank **and** estimated CU so far is under 3.0 **and** wall time is under 2.5 hours: score indices 60–89 as well (90 per task) and apply the same rule once to the pooled bank. No alpha adjustment. If stage 2 is still inconclusive, stop. That cap is the sample-size budget.

Pooled exact McNemar is reported and is not part of the verdict, so it cannot create a second chance to be significant.

## Power (computed before the run)

`scripts/power.py` draws paired item differences from a known discordant rate and applies this rule (400 trials, 4,000 inner bootstrap draws, 60 items per task).

| Scenario | True Δ | Discordant rate | PROVEN | DISPROVEN | Inconclusive |
|---|---:|---:|---:|---:|---:|
| Tie, models agree | 0 | 6% per task | 0.04 | 0.93 | 0.03 |
| Tie, noisy | 0 | 16% per task | 0.02 | 0.36 | 0.62 |
| Grok ahead | +12 pp | 20% (16 vs 4) | 1.00 | 0.00 | 0.00 |
| Grok ahead, round-1 sized | +30 pp | 38% (34 vs 4) | 1.00 | 0.00 | 0.00 |
| Qwen ahead | −12 pp | 20% (4 vs 16) | 0.00 | 1.00 | 0.00 |
| Mixed tasks, overall +5.2 pp | +5.2 pp | low on objective tasks, ~20% on free text | 0.77 | 0.04 | 0.19 |
| Hard zone, +4 pp | +4 pp | 14% (9 vs 5) | 0.45 | 0.12 | 0.43 |
| Hard zone at the n=90 cap | +4 pp | same | 0.62 | 0.15 | 0.23 |

A gap on the scale of round 1, or a 12-point gap, is essentially certain to resolve. A true tie where the models miss the same items (the likely pattern on a template bank) resolves as equivalence about 93% of the time. The 4% PROVEN rate under that tie is the ordinary tail in which the interval excludes 0 and also sticks out past 5 points. The design does not reliably resolve a true 4-point gap or a tie with 16% discordance; those are the cases stage 2 is for, and they can still end inconclusive when the sample-size cap is reached. That is an accepted limit of 5 CU and a 5-point margin, not a reason to change the margin after seeing scores.

Normal approximation behind the table: with n = 300 paired items, SE(Δ) ≈ √(π_d / 300). At π_d = 0.06, SE ≈ 0.014 and the 90% half-width is about 2.3 points, inside the margin. At a 12-point gap with π_d = 0.20, SE ≈ 0.026 and the 95% interval sits near [7, 17], clear of 0.

## Protocol

- Tasks: newsletter triage (exact category), one-sentence summary, tweet, Python regex (7 fullmatch tests), JSON extraction (exact object). 60 items each. Public synthetic text only.
- Qwen: `Qwen/Qwen3-4B`, bitsandbytes NF4, greedy, `max_new_tokens=160`, `enable_thinking=False`. Colab session `r3023`. Order: L4 up to 15 attempts with a 5-minute wait after a refusal, then T4 (5), then A100 (5), then a CPU runtime and, only if that fails too, local CPU with Qwen3-1.7B. Other account sessions are not stopped.
- Grok: `grok --no-auto-update -p ... --output-format json --disable-web-search --no-subagents --max-turns 1 --reasoning-effort low`, empty tool list, empty working directory. Resumable JSONL.
- Judge: `codex exec` on the default account, blinded aliases, batches of 10 items, schema in `scripts/judge_schema.json`.
- CU cap inside the runner: stop starting a new batch once estimated session CU exceeds 4.6, or 150 minutes, whichever comes first. Estimate is elapsed hours times the incremental `colab usage` rate after allocation (fallback 1.54 CU/h for an L4 if the rate line has not moved). Do not start if the balance is under 50 CU.
- Answers are appended per item and downloaded after every batch. Scripts skip ids already on disk.

Item SHA-256 (bytes of the JSON files):

- `items.json` `8c78fb9b2fe71979a3313fafe9a62a8098564f8c9a3d29568a3965c3422d537c`
- `items_ext.json` `6922dd31747d7a269082b98bf90b7a9d916d9f9c0066283e1dbb3fb203826be5`
- `items_all.json` `17c81ad1b3dad62386fca375395ac812b1eb74a88ff7efafac08a01f462ffde8`

## How to run

```bash
source /workspace/tools/setup.sh
cd /workspace/work/tig/023r3-small-model-bakeoff
/workspace/tools/mlenv/bin/python scripts/test_score.py
/workspace/tools/mlenv/bin/python scripts/run_experiment.py
```

`run_experiment.py` runs Grok and Colab together, then the judge, then `scripts/analyse.py`, then stage 2 only under the rule above.

## Compute (actual)

Pending the run. Target GPU is an L4. Session name `r3023`. Budget 5 CU.
