# Experiment 023 round 4 — complete the frozen paired bakeoff

Written before round-4 scoring. Hypothesis: Grok has higher equal-weighted accuracy than Qwen3-4B across five daily text tasks. The exact round-3 protocol, decision rule, margin justification and power simulation are preserved in README-round3.md and scripts/score.py; they remain binding.

## Changes and fixed decision rule

Resume only: copy all frozen items, all 300 answers per arm, and scripts from round 3. Never regenerate an existing answer. Round 3 failed on Codex account limits before any free-text judge grades. Round 4 completes missing grades using the default-account Codex CLI, blind aliases and the same rubric. Triage, regex and JSON remain programmatically scored; summary word count and tweet character/hashtag constraints remain programmatic, combined with blind semantic grades. Save every completed judge batch locally; reruns skip existing grades. Original folders remain read-only.

Use 60 items per task (300 paired items), all required grades present. Resample paired items within each task, average five task differences, 20,000 bootstrap draws, seed 23023. Delta is Grok minus Qwen accuracy. In priority order: DISPROVEN/equivalent if 90% CI strictly within ±5 percentage points; otherwise PROVEN/Grok superior if 95% CI entirely above zero; otherwise DISPROVEN/Qwen superior if 95% CI entirely below zero. Otherwise apply the original conditional extension to 90/task only under its compute/time conditions. STILL INCONCLUSIVE only upon exhaustion of the fixed budget (including original sample cap).

Five points is the original practical margin, justified against the original roughly 50-point gap and paid per-call cost. The unchanged preregistered power simulation gives 93% equivalence power for a true tie with 6% discordance and approximately 100% superiority power for a 12-point gap with 20% discordance. Normal approximation: n=300, SE=sqrt(discordance/300), giving a 90% half-width of 2.3 points at 6% discordance and 95% half-width of 5.1 points at 20%. A noisy tie or small boundary effect may exhaust the original n=90 cap.

## Compute and execution

Before scoring, `source /workspace/tools/setup.sh; colab usage` reported balance 174.92 CU, rate 0, no assignments. Stage 1 needs no new model generation and hence no new GPU session. Round-4 GPU: none; CU used: 0.00. Inherited Qwen answers used an NVIDIA L4 and estimated 0.3808 CU in round 3 (not charged to round 4). Maximum new compute budget 5 CU, wall time about four hours. Never stop another experiment's sessions. If an extension is necessary use session r3023r4 and follow the requested allocation/retry/cleanup protocol.

Run `/workspace/tools/mlenv/bin/python scripts/judge.py items.json`, then `/workspace/tools/mlenv/bin/python scripts/analyse.py items.json`. Local raw JSONL grades are flushed after each batch. Inherited results are retained; round-4 provenance and outputs are separately recorded. No Colab outputs are missing from stage 1, so no remote copy is needed.

## Completed outcome

PROVEN on the complete stage-1 bank: Grok 288/300 (96.0%), Qwen 165/300 (55.0%), difference +41.0 pp, paired 95% CI [36.7, 45.7], 90% CI [37.3, 44.7]. All 240 missing blind grades were completed in 12 batches, without regenerating any answers. No extension is permitted or needed after this conclusive result. Final balance remains 174.92 CU. Other account assignments appeared during grading; none was started or stopped by this experiment. Round-4 GPU none, CU 0.00; inherited L4 generation estimate 0.3808 CU.

Verification: original scoring/decision tests passed; `scripts/verify_resume.py` checks frozen SHA-256 hashes, 600-answer coverage and 240 unique required grades. It reconstructs the exact blinded judge input content from the stored alias map and unchanged answers. Judge logs retain raw responses. The chart code needed only a numerical clamp of negative floating-point error-bar lengths at 100%; no scoring or statistical rule changed. See writeup.md, chart.png and results/summary.json. The template bank limits generalization beyond these synthetic chores.
