# Front-Chasing Refinement — Equidistribute a Fixed Cell Budget (Format B, isolated)

You are the mesh-allocation module of an adaptive solver. The true solution field
`F(x)` lives on a 1-D domain of `N` integer grid points `x = 0..N-1`, and it is smooth
almost everywhere except for a handful of sharp fronts or narrow spikes at unknown
locations. Running the fine solver everywhere is too expensive: you get a fixed budget
of exactly `B` reconstruction cells (`B << N`) to cover the whole domain, and you must
decide where to put them. You never see `F` itself — only a coarse, cheap "pilot" pass:
`F` sampled at `N0+1` evenly spaced probe points (`N0 << N`). From that limited proxy
you must **estimate** where the error will be large, **spend your cell budget where
the estimated error is largest** (not spread it evenly across the domain), and
**coarsen (merge) the cells you'd otherwise waste on smooth, over-resolved stretches**
so their share of the budget can be recycled toward the fronts.

## Public instance (stdin JSON)
```json
{
  "N": 360,                       // fine domain: integers 0..N-1
  "N0": 90,                       // number of base probe blocks
  "B": 18,                        // fixed total cell budget (this is what you must hit exactly)
  "BW": 4,                        // probe spacing; probe_x[j] = j * BW
  "probe_x": [0, 4, 8, ..., 360], // N0+1 probe positions
  "probe_f": [f0, f1, ..., fN0]   // F sampled at probe_x ONLY -- your only view of F
}
```

## Answer (stdout JSON)
```json
{"cuts": [c1, c2, ..., c_{B-1}]}
```
Exactly `B-1` real numbers, strictly increasing, each strictly between `0` and `N`.
Together with the implicit boundaries `0` and `N` they define your `B` reconstruction
cells `[0,c1), [c1,c2), ..., [c_{B-1}, N)`. Any wrong count, non-finite value, value
outside `(0, N)`, or non-strictly-increasing list is rejected and scores **0**.

## What the grader computes
For each of your cells, the grader reconstructs it with the field's own **true mean**
over that cell (the best possible constant fit — you are graded purely on *where* you
put the cell boundaries, not on predicting values) and sums the squared deviations:
```
error(cell) = sum_{i in cell} (F(i) - mean_F(cell))^2
total_error = sum over your B cells of error(cell)
```
A cell that straddles a sharp front has huge internal spread (large `error`); a cell
sitting in a flat stretch has almost none. Since your `probe_f` only samples `F` at
`N0+1` points, sharp features narrower than the probe spacing are only *hinted at*
through how much consecutive probe values jump — you must infer, not read off, where
the field is actually misbehaving.

## Objective & scoring
**Minimize** `total_error`. The grader also builds its own reference partition — the
**uniform grid** of `B` equal-width cells — and computes `baseline_error` for it. Your
per-instance score is
```
score_instance = min(1, K * baseline_error / max(total_error, eps))
```
for a fixed positive constant `K` (so reproducing the uniform grid scores a small,
fixed reference value, and cutting `total_error` well below the uniform baseline earns
a much higher score — there is headroom left above even a strong allocator). The final
score is the mean over 10 fixed, seeded instances with varying numbers, widths, and
placements of fronts/spikes against the smooth background — including instances where
one dominant front sits close to a second, smaller feature, so that dumping the whole
budget on the sharpest signal starves the rest of the domain.

## Suggested strategies (increasing sophistication)
- **Uniform grid** — ignore the probes, split into `B` equal-width cells.
- **Top-k raw curvature** — compute a second-difference indicator at each interior
  probe node and put a cut at the `B-1` largest values. Simple, but several adjacent
  nodes around one sharp feature can all score highly, burning most of the budget on a
  single front while a second, real-but-smaller feature never gets a cut.
- **Equidistributing allocation** — treat the probe samples as a proxy field and choose
  cell boundaries so that the *estimated* error is balanced across cells rather than
  chasing raw indicator magnitude, explicitly trading cells from low-estimated-error
  (flat) regions for cells at high-estimated-error (front) regions under the fixed
  total budget.
- **Iterative coarsen/refine (recycle)** — start from any partition, repeatedly split
  the currently worst-estimated cell and pay for it by merging the two adjacent cells
  with the smallest combined estimated error, holding the cell count fixed at `B`
  throughout, until no such swap improves the estimate.
