Measured results for the base model. CORE is the DCLM ensemble (higher is better); GPT-2 1.5B = 0.256525 is the bar. val_bpb is vocab-invariant validation bits-per-byte (lower is better).

**Report-card base column (the $100 / depth-20 reference run):**

| Metric | BASE |
|---|---|
| CORE | 0.2219 |

The depth-20 base model reaches CORE **0.2219** — a strong, talkable base for the budget, though just under the GPT-2 bar at this size (the report-card run is a d20; clearing 0.256525 requires the d24–d26 range).

**The "time-to-GPT-2" speedrun (committed leaderboard, base model only):** the same architecture+optimizer, sized to d24–d26 and tuned over time, does clear GPT-2 on one node in a few hours:

| time (hr) | val_bpb | CORE | change | date | commit |
|---|---|---|---|---|---|
| 168 (2019, 32×TPUv3) | — | 0.2565 | original GPT-2 1.5B checkpoint | 2019 | — |
| 3.04 | 0.74833 | 0.2585 | d24 baseline | Jan 29 2026 | 348fbb3 |
| 2.91 | 0.74504 | 0.2578 | + fp8 training | Feb 2 2026 | a67eba3 |
| 2.76 | 0.74645 | 0.2602 | total batch size → 1M tokens | Feb 5 2026 | 2c062aa |
| 2.02 | 0.71854 | 0.2571 | dataset → NVIDIA ClimbMix | Mar 4 2026 | 324e69c |
| 1.80 | 0.71808 | 0.2690 | autoresearch round 1 | Mar 9 2026 | 6ed7d1d |
| 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 |

Lower-is-better: val_bpb. Higher-is-better: CORE. The 2019 GPT-2 took ~168 GPU-hours at ~$43,000; the current speedrun beats its CORE in ~1.65 hours on an 8×H100 node (~$40 at $24/hr) — a ~600× cost reduction at equal capability.

Provenance: the **CORE 0.2219** base value is from the repo author's published depth-20 report card (the original nanochat "$100 speedrun" announcement; not committed as a file at this repo HEAD). The speedrun table is committed verbatim in the repo `README.md` and `dev/LEADERBOARD.md`. None of these were re-run by us.
