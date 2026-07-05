# Tide Pool Reserve: Intertidal Habitat Delineation

## Background

A marine reserve manager surveys a rocky **intertidal shore** modeled as a
`W x H` integer coordinate grid. On the shore live `N` focal organisms -- a limpet
colony, an anemone cluster, a kelp holdfast, and so on. Organism `i` sits at an
integer survey point `(x_i, y_i)` and needs a protected **tide pool** of target
habitat area `a_i` to sustain its population through the tidal cycle.

Your job: delineate `N` axis-aligned rectangular pool boundaries -- one per
organism -- that fit inside the shore, never overlap, and match each organism's
target area as closely as possible while enclosing its survey point.

This is an *offline, deterministically scored* heuristic-contest instance in the
AtCoder-heuristic-contest "area-target rectangle placement" lineage (AHC001). The
score depends only on your output, never on wall-clock time. As a design ceiling,
a good delineation heuristic uses on the order of `1e7` primitive operations; the
harness enforces this only as a subprocess timeout, never as part of the score.

## Candidate program contract

Your program reads ONE JSON object (the public instance) from **stdin** and writes
ONE JSON object (your answer) to **stdout**. It runs in an isolated subprocess and
sees only the public instance.

### Input (stdin) -- public instance
```json
{
  "name": "shore101",
  "W": 1000,
  "H": 1000,
  "n": 18,
  "x": [ ... ],   "y": [ ... ],   "a": [ ... ]
}
```
- `W`, `H`: shore dimensions.
- `n`: number of organisms (`= len(x) = len(y) = len(a)`).
- `x[i]`, `y[i]`: organism `i`'s survey point, with `0 <= x[i] < W`, `0 <= y[i] < H`
  (all survey points are distinct integer cells).
- `a[i]`: organism `i`'s target habitat area, `a[i] >= 1`. The targets sum to
  roughly 78-90% of the shore area, so the pools genuinely compete for space.

### Output (stdout) -- answer
```json
{ "rects": [ [x1, y1, x2, y2], ... ] }
```
Exactly `n` rectangles (in the same order as the organisms). Each rectangle is
four **integers** with
```
0 <= x1 < x2 <= W        0 <= y1 < y2 <= H
```
Rectangle `i` is the pool boundary for organism `i`.

## Feasibility

A layout is **valid** iff:
1. `rects` is a list of exactly `n` rectangles, each four integers obeying the
   bound/order constraints above; and
2. no two rectangles have overlapping **interiors** (sharing an edge is allowed).

Any structural violation, an overlap, a crash, a timeout, or non-JSON output makes
the **whole instance score 0.0**.

A rectangle that does not enclose its own survey point `(x_i, y_i)` (i.e. not
`x1 <= x_i < x2` and `y1 <= y_i < y2`) is still structurally valid, but that pool
earns **0 quality** -- exactly as in the reference contest scorer.

## Objective and scoring (deterministic)

For each pool `i` with rectangle area `s_i = (x2-x1)*(y2-y1)` and target `a_i`:

- if the rectangle encloses `(x_i, y_i)`:
  `p_i = 1 - min(a_i, s_i) / max(a_i, s_i)`, and `q_i = 1 - p_i^2`  (in `[0, 1]`);
- otherwise `q_i = 0`.

The instance raw quality is the mean `Q = (1/n) * sum_i q_i` (the AHC001 normalized
score). The evaluator anchors this against the manager's trivial all-`1x1` layout
`Q_base`, and reports a normalized per-instance ratio
```
r = clamp( 0.1 + 0.9 * (Q_cand - Q_base) / max(1e-9, 1 - Q_base),  0, 1 ).
```
So the all-`1x1` layout scores about `0.1`, and the (unreachable) perfect layout
would score `1.0`. Because the targets pack the shore to 78-90%, no layout can hit
every target without overlaps, leaving real headroom below `1.0`.

The overall score is the mean of `r` across all instances (a mix of easier and
harder / denser held-out shores).

## Strategy hints

- **Baseline**: one unit cell per organism -- feasible, but nearly zero quality.
- **Greedy growth**: grow each pool one side at a time toward its target, taking
  the largest feasible non-overlapping extension.
- **Construct-then-refine**: choose a good growth order, then locally grow/shrink
  pools -- shrinking an overshooting pool frees space a neighbour can reclaim.
