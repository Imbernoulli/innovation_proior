# Ridge-Blind Prospecting: Self-Tuning a Black-Box Landscape Scan

A prospecting drone surveys a bounded box for the richest ore deposit. The
ore-richness field is a fixed, hidden, continuous function of position: some
survey boxes hold one broad, smooth "swell" (unimodal); others hold several
separate deposits of different grade laced with a noisy, high-frequency
mineral-vein texture (multimodal / rugged). You never see the field's
formula — only a small set of readings taken before your run, plus a limited
drilling budget to spend now.

The drone starts parked over a locally promising outcrop. Before your run,
two kinds of pilot readings were already taken and are handed to you:

- **`local_probes`** — a handful of short-hop readings right around the
  start point, at a couple of small, increasing radii along a few
  directions. These sense how fast the field changes over short distances —
  its correlation length / local texture.
- **`scan_probes`** — one coarse reading near *every* named survey anomaly
  across the whole box, not just near the start. These are your only signal
  about deposits far from where you began.

Your job: decide where to sink additional wells (drilling budget `Q`) to
find as high a reading as possible. This is genuinely open-ended — a fixed
recipe that always trusts the local neighbourhood works fine when the field
really is smooth, but on a rugged, multi-deposit field the tallest deposit
can sit far from the start, visible only in the coarse `scan_probes`, so a
purely local search caps out on a shorter, nearer deposit.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from
**stdin**, write ONE JSON object (your answer) to **stdout**. Runs in an
isolated subprocess seeing only the public instance.

### Public instance (stdin)

```json
{
  "name": "survey_304_rugged", "dim": 2,
  "box": [[-6.0, 6.0], [-6.0, 6.0]],
  "budget": 20,
  "start": [1.2, -0.4], "start_value": 6.83,
  "local_probes": [
    {"x": [1.5, -0.4], "value": 6.9, "r": 0.4}, ...
  ],
  "scan_probes": [
    {"x": [1.1, -0.5], "value": 6.95}, ...
  ]
}
```

- `box`: per-dimension `[lo, hi]` bounds, length `dim`.
- `budget`: the maximum number of wells (query points) you may drill.
- `local_probes`: short-hop readings around `start` (a few directions,
  a couple of radii each); `r` is the distance from `start`.
- `scan_probes`: one reading near each survey anomaly, scattered across the
  whole box.

### Answer (stdout)

```json
{ "queries": [[1.4, -0.3], [0.9, 2.1], ...] }
```

- `queries` must be a list of **1 to `budget`** points, each a length-`dim`
  list of finite numbers, each coordinate inside the corresponding `box`
  bound.
- Any violation (wrong length, out of bounds, non-finite, wrong dimension),
  a crash, a timeout, or non-JSON output makes that instance score `0.0`.

## Objective

**Maximize** the best reading found across a fixed, seeded family of 10
instances (mixing smooth single-deposit boxes, several-deposit rugged boxes,
and larger 3-D held-out boxes).

## Scoring (deterministic)

For each instance the evaluator computes, itself, the true hidden field and:

- `f_lo` — the best reading a **blind uniform-random** well sweep (same
  budget) would find — a weak reference,
- `f_hi` — a loose analytic upper bound on the field over the whole box
  (generally unreachable),
- `f_cand` — the best value among `start_value`, every pilot reading, and
  the field evaluated at every valid submitted query,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (f_cand - f_lo) / (f_hi - f_lo), 0, 1 )
```

Matching the blind sweep scores ≈ `0.1`; approaching the loose upper bound
scores near `1.0` (essentially unreachable, so there is real headroom). The
reported **Ratio** is the mean of `r` over all instances; **Vector** lists
the per-instance scores.

## Suggested strategies

1. **Blind sweep** (baseline): ignore the pilot readings, drill the budget
   as uniform-random points.
2. **Local-only refinement**: trust the best pilot reading right around the
   start, drill a fixed-radius cluster of wells around it.
3. **Correlation-aware, elite-seeded search**: use the local probes to
   estimate how rugged the field is near the start; if smooth, refine
   tightly around the single best reading; if rugged, seed several
   restart clusters from the best readings across the whole box (including
   distant `scan_probes`) with a wider step size.
