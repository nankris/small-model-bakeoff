# Can a tiny AI model on my laptop do my daily chores? I tested two against Grok

*Draft for Tech in General: real numbers, 15 Oct 2026.*

Every day I do small text jobs: sort newsletters, summarise an article, draft a tweet, write a regex, pull fields out of a job post. Each one is too small for a big model but too annoying to do by hand. So can a *tiny* open model running on my CPU do them for free?

## The test

5 everyday tasks, 4 examples each, all written by me:

1. **Newsletter triage**: is this email AI research, a job alert, a promotion or a product update?
2. **One-line summary** (≤25 words)
3. **Tweet draft** from three bullet notes (≤280 characters, one hashtag)
4. **Regex** from a plain-English description, checked against 7 test strings
5. **JSON extraction** from a job posting

Contenders: **Qwen3-0.6B** and **SmolLM2-360M** on a plain 2-core CPU, against **Grok** (via the Grok Build CLI). Grok also graded the summaries and tweets blind, without knowing who wrote which.

## Results

| | Grok | Qwen3-0.6B | SmolLM2-360M |
|---|---|---|---|
| Newsletter triage | 100% | 25% | 25% |
| Summary | 69% | 38% | 50% |
| Tweet | 94% | 38% | 19% |
| Regex | 4/4 | 0/4 | 1/4 |
| JSON extraction | 100% | 100% | 100% |
| **Overall** | **92%** | **40%** | **44%** |

Grok costs about a third of a cent per task (API-equivalent). The tiny models are free and answer in 2–4 seconds, but they aren't light on memory at full precision: SmolLM2 peaked at 2.7 GB of RAM and Qwen3-0.6B at 5.1 GB (fp32, unquantised; a 4-bit build would need far less).

## What happened

- **JSON extraction: the tiny models nailed it.** All 4 job posts, every field right. When the answer is sitting in the text and the format is clear, 360M parameters is enough.
- **Triage: they guessed.** Qwen labelled every email "Product update". SmolLM2 called a flash-sale email a "Job alert".
- **Regex: mostly nonsense.** Qwen answered `/regex/` three times, the word "regex" between slashes. SmolLM2's PIN-code regex expected 7 digits instead of 6.
- **Limits get ignored.** "At most 25 words" and "exactly one hashtag" were rules Grok followed 8 out of 8 times. The tiny models followed them 2 times and 1 time. SmolLM2 also invented Twitter handles.

## My takeaway

Use a tiny local model for **extraction into a fixed format**, where the answer is literally in the input. For anything that needs judgment, following rules or writing code, a fraction of a cent per call for a frontier model is money well spent. A 4B model or few-shot examples might change that, and that's my next test.

*Caveats: 4 examples per task, so treat this as a sniff test; Grok grading its own work blind could still favour it; my examples are short and clean.*
