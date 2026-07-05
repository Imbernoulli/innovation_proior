# Rift Valley Geothermal: Well-Field Siting

## Story

A geothermal operator has surveyed a square tract of the Rift Valley and reduced it
to a discrete heat map: an `N x N` grid where cell `(r, c)` holds an integer
`heat[r][c] >= 0`, the recoverable thermal energy under that cell (arbitrary units).
Heat is concentrated in a few underground plumes, with a diffuse background
everywhere.

You may drill `K` production wells at `K` **distinct** grid cells. A well drilled at
`(r, c)` taps every cell within Chebyshev radius `R` of it -- the `(2R+1) x (2R+1)`
square window centered on the well, clipped to the tract boundary. Because the
reservoir is one shared fluid body, **a cell's heat can be drawn at most once**: no
matter how many wells' windows cover it, it contributes to the field's yield only
once.

Your goal is to choose the `K` well cells to **maximize the field's total yield**:

```
yield = sum of heat[r][c] over every cell covered by at least one well's window.
```

Two wells drilled close together waste capacity by re-tapping the same cells. The
hottest cells all sit on top of the tallest plume, so naively drilling there
double-taps it; spreading wells across separate plumes captures more distinct heat.
This is the coverage-vs-clustering tension you must resolve.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "tract101",
  "n": 14,
  "radius": 2,
  "k": 4,
  "heat": [[ ... N ints ... ], ... N rows ... ]
}
```

- `n` (int): the grid is `n x n`.
- `radius` (int `R`): each well taps the Chebyshev-`R` window around it.
- `k` (int `K`): number of wells to drill.
- `heat` (list of `n` lists of `n` ints `>= 0`): the heat map.

## Output (one JSON object on stdout)

```json
{"wells": [[r0, c0], [r1, c1], ...]}
```

- Exactly `K` entries.
- Each entry is a pair of integers `[r, c]` with `0 <= r < n` and `0 <= c < n`.
- The `K` cells must be **pairwise distinct**.

Any of the following makes the instance score `0.0`: wrong number of wells, a
coordinate out of range, duplicate cells, non-integer coordinates, a crash, a
timeout, or output that is not the JSON object above.

## Objective and scoring (deterministic)

For each instance the evaluator computes:

- `y_base` = union yield of the `K` **hottest** cells (top-`K` by heat, ties broken
  by `(row, col)`). This overlap-blind placement is the weak baseline.
- `y_ub` = total heat of the entire tract (sum over all cells). A loose,
  generally unreachable upper bound.
- `y_cand` = the union yield of your placement.

and normalizes:

```
r = clamp( 0.1 + 0.9 * (y_cand - y_base) / max(1e-9, y_ub - y_base), 0, 1 )
```

Reproducing the top-`K`-hottest baseline scores about `0.1`; doing worse scores
below `0.1`; covering more distinct heat scores higher, capped at `1.0`. Because the
full-tract upper bound is loose, even strong siters stay below `1.0` -- there is
always headroom. Your final score is the mean of `r` over all instances (a mix of
sizes, radii, and plume counts, including harder held-out tracts with more plumes
than wells).

## Notes

- Scoring depends only on your emitted `wells`; it never measures wall-clock time.
  Treat the per-instance limit as an operation budget for search-based methods
  (marginal-gain greedy, local search, annealing, restarts).
- Your program is run in an isolated subprocess and sees only the public instance
  above.
