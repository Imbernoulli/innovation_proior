# Forge Furnace Setback Ladder (Format B, isolated)

A forge furnace serves a stream of jobs. Between jobs it sits idle for a
**gap** of unknown length. Right after each job you must commit to ONE of
`K+1` graded **setback levels** (`0` = hold at full operating temperature,
`K` = full shutdown, with intermediate partial-cooling levels between) and
hold it for the whole upcoming gap. Deeper levels use less power while
idle, but when the next job arrives the furnace must be brought back to
operating temperature, and that restart energy is **state-dependent**: it
grows with how deep the level cooled the furnace and how long the gap
lasted, following a **concave, saturating** curve (diminishing extra cost
per extra unit of depth/time). Gap lengths are drawn from a hidden
two-regime process: at every gap the process is in either a "short" or a
"long" regime, and — independently in each regime — the gap length is
drawn from an **Exponential distribution** with that regime's given mean
(`mu_short` or `mu_long`). The regime itself **persists** from gap to gap
(it stays the same with probability `p_stay`, else flips) — recently
realized gaps carry real signal about which regime is currently active.

## How you are invoked
Each of the 10 test cases covers a 40-gap run, graded in **10 causal
chunks of 4 gaps**. Your program is invoked **once per chunk** (a fresh,
isolated process each time — nothing persists between calls) and must
commit ONE setback level for the whole upcoming chunk, given every gap
**already realized** (a pre-run calibration history plus all strictly
earlier chunks of this run) — never the chunk you are about to face.

### Input (stdin JSON, one call per chunk)
```json
{
  "K": 4,
  "tau":        [1.0, 0.7, 0.4, 0.15, 0.0],   // target temp fraction per level
  "hold_power": [..5 floats..],                // idling power per level (decreasing)
  "cool_rate":  [..5 floats..],                // relaxation rate per level (increasing)
  "reheat_coeff": <float>,                     // A in the restart-cost formula
  "reheat_exp":   <float>,                     // exponent p < 1 (concave)
  "mu_short": <float>, "mu_long": <float>,     // the two regimes' mean gap length
  "p_stay":   <float>,                         // P(next gap stays in the same regime)
  "n_chunks": 10, "chunk_size": 4,
  "chunk_index": <int>,                        // which chunk you are deciding, 0-indexed
  "history": [<float>, ...]                    // ALL gaps realized strictly before this chunk
}
```

### Output (stdout JSON)
```json
{"level": <int in [0, K]>}
```
Any other shape/type, a non-finite or out-of-range level fails the WHOLE
test case (score 0 for it).

## Cost model (what the grader replays against the true gaps)
For a chosen level `L` held during a gap of true length `g`:
```
deficit = (1 - tau[L]) * (1 - exp(-cool_rate[L] * g))
cost    = hold_power[L] * g + reheat_coeff * deficit ** reheat_exp
```
Your total for a test case is the sum of this cost over all 40 gaps, using
whatever level you committed for the chunk each gap falls in.

## Objective & scoring
**Minimize** total energy. Per test case the grader computes two
references from the TRUE (hidden) gap sequence: `W` = the worse of the two
constant extremes (holding every gap, or shutting down every gap), and `O`
= the unattainable oracle that picks the pointwise-best level for every
individual true gap. Your score for that case is
`clip((W - your_total) / (W - O), 0, 1)`; the final Ratio is the mean over
the 10 fixed, seeded test cases (some with strong regime persistence and
sharp short/long contrast, some near-i.i.d. controls).

## Why this isn't just two-state ski-rental
Naively you might compute one aggregate average gap length and pick
whichever of hold/shutdown is cheaper for it — classic ski-rental. That
ignores two things the input hands you: (1) the regime **persists**, so
recent gaps predict the current regime far better than a stale global
average, especially right after a switch; (2) because the restart cost is
**concave** in depth/time, the level that minimizes cost *in expectation*
under genuine regime uncertainty is often an **intermediate** rung of the
ladder, not one of the two extremes a binary rent-or-buy argument would
pick.

## Suggested strategies (increasing sophistication)
- **Blind constant** — hold one fixed level for every chunk, no data used.
- **Classic ski-rental** — one global average gap length, binary hold/shutdown.
- **Regime-static ladder** — use `mu_short`/`mu_long` for an expectation-optimal
  fixed level, ignoring `history`'s recency.
- **Bayesian filtering + expected-cost ladder** — forward-filter a belief
  over the current regime from `history` using `p_stay`, then choose the
  level minimizing full expected cost under that belief, each chunk.
