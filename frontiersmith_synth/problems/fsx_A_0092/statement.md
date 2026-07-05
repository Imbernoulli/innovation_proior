# Rift Valley Geothermal: Brine Reinjection Loops

A geothermal power station taps a rift-valley field. After the turbines, spent
**brine** from many production **wells** must be reinjected back underground through
a set of **injection loops** (pump-and-pipe circuits). Every loop has the same
maximum hydraulic **throughput capacity** `C`. Several wells may share one loop as
long as their combined flow does not exceed `C`.

But brine is hot, and mixing streams of very different temperature in one loop
cracks the pipes through thermal shock. So a loop is physically safe only if the
temperatures of **all** wells sharing it lie within a fixed **band** of width
`band` — that is, `max_temp - min_temp <= band`. Firing up a loop (its pump plus a
full pipe circuit) costs one unit regardless of how full it is.

Your job: write a heuristic that connects **every** well to a loop so that the
whole field is reinjected using **as few active loops as possible**.

This is 1-D bin packing **with a temperature-compatibility constraint**: wells are
items with an integer size (flow) *and* a temperature tag; loops are bins of
capacity `C`; a loop is feasible only if it is not overfilled **and** its
temperature spread stays within `band`. The band is what makes this harder than
plain bin packing — two wells that fit together on flow can be forbidden to share
on temperature, so a size-only packer is easily beaten by one that also clusters
on temperature.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your answer) to **stdout**. It
runs in an isolated subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute an assignment ...
print(json.dumps({"assign": assign}))
```

### Public instance (stdin)

```json
{
  "name": "field101",
  "capacity": 20,             // C, throughput per loop (positive integer)
  "band": 24,                 // max allowed temperature spread within a loop
  "n": 22,                    // N, number of wells
  "flow": [7, 12, 3, 5, ...], // N integer flows, each 1 <= f_i <= C
  "temp": [40, 118, 12, ...]  // N integer temperatures, each t_i >= 0
}
```

### Answer (stdout)

```json
{ "assign": [0, 0, 1, 0, 2, ...] }   // length N; assign[i] = loop index of well i
```

- `assign` must be a list of **exactly N** non-negative integers.
- Loop indices need not be contiguous. A loop is "active" iff at least one well
  joins it, and the score counts the number of **distinct non-empty loops**.
- A layout is **valid** iff, for every loop, (a) the total flow of its wells does
  **not** exceed `C` and (b) the temperature spread (`max - min`) of its wells does
  **not** exceed `band`.

Any invalid output (wrong length, a non-integer or negative index, an overfilled
loop, a temperature-band violation), a crash, a timeout, or non-JSON output makes
that instance score `0.0`.

## Objective

**Minimize** the number of active loops across a fixed, seeded family of 12
instances that vary in well count, loop capacity, flow distribution (uniform,
half-loop "medium", small+large "bimodal", large-flow "heavy") and temperature
layout (uniformly "spread" vs. a few "clustered" hot spots). Several instances are
larger / harder held-out cases.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_lb`   = `ceil(sum(flow) / C)` — the **L1 lower bound**, which *ignores* the
  temperature band and is therefore an unreachable ideal,
- `q_base` = loops used by an internal **constrained next-fit** rule (a weak
  baseline that honours both flow and band),
- `q_cand` = active loops used by **your** layout,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

- Matching next-fit scores ≈ `0.1`; reaching the (generally unreachable) L1 bound
  scores `1.0`; doing worse than next-fit scores below `0.1`.
- Because L1 ignores the temperature band, even excellent temperature-aware
  packers stay strictly below `1.0` on most instances — there is real headroom.

The reported **Ratio** is the mean of `r` over all instances; the **Vector** lists
the per-instance scores.

## Suggested strategies

1. **Constrained next-fit** (baseline): fill the current loop until a well fails on
   flow *or* band, then open a new one — never look back.
2. **First-fit-decreasing** on flow with band checks: seat the largest flows first
   into the first compatible open loop, reusing gaps next-fit wastes.
3. **Temperature-aware best-fit-decreasing**: cluster wells by temperature, then
   best-fit flows within each compatible cluster so band-forbidden pairs never
   block a loop.
4. **Multi-restart + local search**: run best-fit under many seeded orderings and
   relocate wells to disband the smallest loops (deterministic), keeping the
   fewest-loop layout.
