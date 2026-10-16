#!/usr/bin/env python3
"""Quick power simulation for the preregistered decision rule. Not an experiment run."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score import TASKS, verdict_from_intervals

B = 4000
N_SIM = 400
N_PER = 60
SEED = 230230


def simulate(name, spec, n_per=N_PER, n_sim=N_SIM, seed=SEED):
    """spec: list of (p_grok_only, p_qwen_only) per task, length 5, same order as TASKS."""
    rng = np.random.default_rng(seed)
    counts = {"PROVEN": 0, "DISPROVEN": 0, "STILL INCONCLUSIVE": 0}
    true_delta = float(np.mean([p10 - p01 for p10, p01 in spec]))
    for _ in range(n_sim):
        diffs = {}
        for t, (p10, p01) in zip(TASKS, spec):
            u = rng.random(n_per)
            d = np.zeros(n_per)
            d[u < p10] = 1.0
            d[(u >= p10) & (u < p10 + p01)] = -1.0
            diffs[t] = d
        # Smaller bootstrap inside the sim. Same quantiles and rule as the analysis.
        rng_b = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        draws = []
        for t in TASKS:
            d = diffs[t]
            ix = rng_b.integers(0, n_per, size=(B, n_per))
            draws.append(d[ix].mean(axis=1))
        overall = np.mean(draws, axis=0)
        ci95 = np.quantile(overall, [0.025, 0.975])
        ci90 = np.quantile(overall, [0.05, 0.95])
        verdict, _ = verdict_from_intervals(ci95, ci90)
        counts[verdict] += 1
    print(
        f"{name}: true Δ={true_delta*100:.1f} pp | "
        + " ".join(f"{k}={v/n_sim:.2f}" for k, v in counts.items())
    )


if __name__ == "__main__":
    # Near-tie, errors mostly shared (discordant 6% per task). Equivalence should land often.
    tie_agree = [(0.03, 0.03)] * 5
    # Near-tie, more discordance (16%). Equivalence power drops.
    tie_noisy = [(0.08, 0.08)] * 5
    # Grok ahead by 12 pp overall, moderate discordance. Superiority should land.
    ahead = [(0.16, 0.04)] * 5
    # The round-1 sized gap, shrunk but still large: +30 pp, Qwen rarely wins an item Grok misses.
    large = [(0.34, 0.04)] * 5
    # Hard zone: +4 pp with 14% discordance. Often neither rule fires. This is why the extension exists.
    hard = [(0.09, 0.05)] * 5
    # Qwen ahead by 12 pp.
    qwen = [(0.04, 0.16)] * 5
    # Mixed tasks: objective tasks tied and agreed, free-text Grok +20 pp. Overall +8 pp.
    mixed = [(0.02, 0.02), (0.18, 0.04), (0.16, 0.04), (0.03, 0.03), (0.02, 0.02)]
    print(f"sims={N_SIM} inner_boot={B} n_per_task={N_PER}")
    for name, spec in [
        ("tie_agree", tie_agree),
        ("tie_noisy", tie_noisy),
        ("grok_plus_12", ahead),
        ("grok_plus_30", large),
        ("hard_plus_4", hard),
        ("qwen_plus_12", qwen),
        ("mixed_plus_8", mixed),
    ]:
        simulate(name, spec)
    print("--- extension n=90, hard zone only ---")
    simulate("hard_plus_4_n90", hard, n_per=90, seed=SEED + 1)
