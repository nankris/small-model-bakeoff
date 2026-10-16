# #023 Small-model bake-off on daily tasks: Qwen3-0.6B and SmolLM2-360M (CPU) vs Grok

**Engine:** Grok Build CLI 1.0.52 (SuperGrok, `grok-4.7-build`) as the reference model and the judge. No engine swap. Grok costs are **API-equivalent estimates** from the CLI's `total_cost_usd`.

**Question.** Can a tiny local model on a laptop-class CPU handle the small text chores of a typical AI-blogger/engineer day: triaging newsletters, one-line summaries, tweet drafts, regexes, JSON extraction?

**Answer, short.** Only JSON extraction. Both tiny models got 4/4 job-posting extractions right, the same as Grok. On everything else they fell well short: newsletter triage 25% (Qwen labelled all 4 "Product update", SmolLM2 said "AI research" 3 times), regex 0/4 and 1/4 passing all tests, summaries and tweets judged 38–50% vs Grok's 69–94%. Overall: **Grok 92%, SmolLM2-360M 44%, Qwen3-0.6B 40%**. Grok costs about **0.35¢ per item** (API-equivalent, reasoning effort low) and ~7 s; the local models were 2–4 s per item at $0 marginal cost.

## Setup
- **Tasks:** 5 tasks × 4 items = 20 prompts, all **synthetic text written for this experiment** (no private data): `scripts/tasks.py`.
  - *Newsletter triage:* subject + preview → one of {AI research, Job alert, Promotion, Product update}. Exact match.
  - *One-line summary:* ≤25-word summary of a short news-style paragraph. Judged.
  - *Tweet draft:* 3 notes → tweet ≤280 chars with exactly one hashtag. Judged.
  - *Regex:* description → Python regex, scored **pass/fail on all 7 test strings** (3 must match, 4 must not) with `re.fullmatch`.
  - *JSON extraction:* job posting → `{company, role, city, min_years}`, scored per field (code fences tolerated).
- **Local models (CPU only, 2 threads, fp32, greedy, max 120 new tokens, chat template):** `Qwen/Qwen3-0.6B` (non-thinking mode) and `HuggingFaceTB/SmolLM2-360M-Instruct`, loaded from the project's local weight copies. 2-vCPU arm64 sandbox with 7.9 GB RAM.
- **Grok:** one call per prompt, lean flags `--system-prompt-override "You are a helpful assistant. Follow the output format exactly." --disable-web-search --no-subagents --tools read_file --reasoning-effort low`, 5 in parallel.
- **Judge:** Grok, same lean flags, one call per summary/tweet item. The three outputs are shuffled (seeded by item id) and labelled A/B/C so the judge doesn't know which system wrote which, scored 1–5 against stated criteria, then rescaled to 0–100.
- Run 2026-10-15, ~02:00–02:15 IST.

```bash
source /workspace/tools/setup.sh; PY=/workspace/tools/mlenv/bin/python
MODEL_KEY=smollm2_360m $PY scripts/run_023_local.py; MODEL_KEY=qwen3_0.6b $PY scripts/run_023_local.py
STEP=answer $PY scripts/run_023_grok.py; STEP=judge $PY scripts/run_023_grok.py
$PY scripts/analyse_023.py      # -> results/summary.json, chart.png
```

## Results (4 items per task)

| task | Grok | Qwen3-0.6B | SmolLM2-360M |
|---|---|---|---|
| newsletter triage (exact) | **100%** | 25% | 25% |
| one-line summary (judged) | **69%** | 38% | 50% |
| tweet draft (judged) | **94%** | 38% | 19% |
| regex (all 7 tests pass) | **4/4** | 0/4 | 1/4 |
| JSON extraction (fields) | 100% | 100% | 100% |
| **overall (mean of tasks)** | **92%** | 40% | 44% |
| summary/tweet hard format checks met (length, one sentence, one hashtag) | 8/8 | 2/8 | 1/8 |
| time per item | 7.4 s (API latency) | 3.9 s (6.6 tok/s) | 2.0 s (13.6 tok/s) |
| cost per item | ~0.35¢ API-equiv. | $0 marginal | $0 marginal |
| peak RAM (fp32 load + 1 generation) | n/a (cloud) | 5.1 GB | 2.7 GB |

- **RAM** (added 2026-10-15 ~03:35 IST, `scripts/measure_ram_023.py` → `results/ram.json`): max RSS of a child process that loads the model in fp32 and generates once. SmolLM2-360M (362M params) peaked at **2.7 GB**, Qwen3-0.6B (596M params) at **5.1 GB**; ~0.45 GB of that is the Python+torch+transformers import. These are upper bounds: fp32 is 4 bytes/param and the figure includes safetensors pages mapped during load. A 4-bit GGUF build would need a fraction of this. The first Qwen RAM attempt died with a bus error on the busy shared sandbox; the immediate retry succeeded.
- Grok spend: $0.070 for the 20 answers + $0.045 for the 8 judge calls (API-equivalent).
- Typical failures: Qwen replied `/regex/` to three of the four regex prompts; SmolLM2 wrote `^[1-9]\d{6}$` (7 digits) for a 6-digit PIN. The tiny models mostly ignored the "at most 25 words" and "exactly one hashtag" limits. SmolLM2's tweets included invented @handles and 4 hashtags.
- Per-item outputs: `results/local_*.jsonl`, `results/grok_answers.json`; judge scores and raw judge text: `results/judge.json`; all scores: `results/summary.json`; call log: `results/grok_calls.csv`.

## Caveats
- **Tiny sample:** 4 items per task, so each item is 25 points. Treat per-task numbers as indicative only.
- **Grok judged its own outputs.** The judging was blind (shuffled A/B/C), but self-preference bias is still possible; Grok scored its own summaries 69% (it marked one of them 2/5), so it was not uniformly generous to itself. A second, independent judge was not run.
- The inputs are short, clean and synthetic; real newsletters are longer and messier, which would likely hurt the tiny models more.
- The local models used greedy decoding, a plain prompt and no few-shot examples. Few-shot prompts, constrained decoding (for regex/JSON) or a 1.5–4B model would likely do much better, but those weren't tested here.
- Local timings are from a shared 2-core sandbox while other jobs were running.
