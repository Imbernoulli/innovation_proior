# Cutting Trees That Have to Grow Back: A Multi-Year Harvest Plan

## Story

You manage a forest stand laid out as an `n x n` grid. Cell `(r, c)` holds an
integer **stage** in `[0, S]`: `0` means bare ground, `1..S-1` are growing
trees, and `S` is a fully mature "old growth" tree. Cutting a tree at stage `k`
banks `k*k` units of value -- a mature (stage-`S`) tree is worth far more than
any younger one.

Every year of a fixed `horizon` of `T` years, in order:

1. **Harvest.** You cut at most `quota` distinct non-empty cells. Each cut
   cell's value is banked and the cell becomes bare (stage `0`).
2. **Growth.** Every remaining tree with stage in `[1, S-1]` grows by exactly
   one stage.
3. **Dispersal.** Every bare cell (stage `0`, whether just cut or already
   bare) looks at the `(2*radius+1) x (2*radius+1)` Chebyshev-`radius` window
   around it, using the board *after* growth. If that window contains at
   least `min_seed` cells at stage `S`, a seedling establishes there (its
   stage becomes `1`). Otherwise it stays bare -- growth alone can never
   create a tree from nothing; only a nearby mature tree can seed one.

The catch: the maturity threshold that lets a tree act as a seed source is
exactly `S` -- the same stage that gives it its highest cutting value.
Cutting the biggest trees maximizes this year's yield, but it is also the
only way a region's seed sources get destroyed. Once every stage-`S` cell
within `radius` of a patch is gone, that patch can never regrow. A plan that
keeps a spatially spread set of mature trees standing -- harvesting *around*
that network instead of *through* it -- can go on cutting near-maximum-value
trees for the whole horizon instead of only the opening years.

You submit **one plan for the entire horizon up front**: everything needed to
simulate the stand forward (the initial grid and every rule and parameter
above) is given to you, and the dynamics are completely deterministic given
your own cuts, so there is no need to observe intermediate years -- you can
simulate the consequences of any plan yourself before submitting it.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "stand05",
  "n": 12,
  "s_max": 6,
  "radius": 2,
  "min_seed": 1,
  "quota": 8,
  "horizon": 30,
  "grid": [[ ... n ints in [0, s_max] ... ], ... n rows ... ]
}
```

- `n`: the grid is `n x n`.
- `s_max` (`S`): the stage cap; a stage-`S` tree is both the most valuable cut
  and the only kind of cell that can seed a bare neighbour.
- `radius`: Chebyshev dispersal radius used in the dispersal step.
- `min_seed`: minimum count of stage-`S` cells in a bare cell's radius window
  (post-growth) needed for that cell to sprout that year.
- `quota`: maximum cells you may cut in any single year.
- `horizon` (`T`): number of years simulated.
- `grid`: the initial stand, `n` rows of `n` ints in `[0, s_max]`.

## Output (one JSON object on stdout)

```json
{"harvests": [[[r0, c0], [r1, c1]], [], [[r2, c2]], ...]}
```

- A list of at most `horizon` year-entries (missing trailing years count as
  "cut nothing that year").
- Year `t`'s entry is a list of at most `quota` `[r, c]` pairs, `0 <= r,c < n`,
  pairwise distinct within that year.
- Every cut cell must actually hold a tree (stage `>= 1`) at the moment year
  `t` is played, given everything you cut in earlier years.

Any of the following makes the instance score `0.0`: wrong types/shape, more
than `horizon` years, more than `quota` cells in a year, an out-of-range or
duplicate cell within a year, a cut claimed on a cell that is not a tree at
that point in the simulation, a crash, a timeout, or output that is not the
JSON object above.

## Objective and scoring (deterministic)

For each instance the evaluator computes:

- `y_triv`: the value of its own weak reference plan -- each year, cut the
  first `quota` non-empty cells found in row-major scan order, ignoring value
  and regrowth entirely.
- `y_ub` = `quota * horizon * s_max * s_max`: a loose, generally unreachable
  bound (it assumes every single cut, every year, is a stage-`S` tree).
- `y_cand`: the value of your plan, replayed by the evaluator against the
  true dynamics.

and normalizes:

```
r = clamp( 0.1 + 0.9 * (y_cand - y_triv) / max(1e-9, y_ub - y_triv), 0, 1 )
```

Matching the weak reference scores about `0.1`; doing worse scores below
`0.1`; a better plan scores higher, capped at `1.0`. Because the upper bound
is loose, even a strong plan stays below `1.0`. Your final score is the mean
of `r` over all instances -- a mix of grid sizes, dispersal radii, and
sparse/clustered starting stands, several engineered so cutting biggest-first
looks great for the first several years and then collapses.

## Notes

- Scoring never measures wall-clock time; treat the horizon as a planning
  problem you can simulate and search over yourself.
- Your program runs in an isolated subprocess and sees only the public
  instance above.
