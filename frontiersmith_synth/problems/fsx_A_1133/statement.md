# Occam's Ornament: Restoring a Damaged Symmetric Tile

A workshop restores square ceramic ornaments. Each ornament is an `N x N` grid of
tile colors (integers `0..K-1`) that was built by picking exactly one hidden
**symmetry group** `G` — some combination of point symmetries (rotations by
90/180/270 degrees, mirror reflections across the horizontal, vertical, or either
diagonal axis) and periodic repetition (a smaller motif tiled across the canvas by
shifting rows/columns by a fixed period, wrapping around the edges) — then coloring
one cell per **orbit** of `G` (every cell reachable from it by a symmetry in `G`)
uniformly. Different hidden groups nest inside each other: the full square symmetry
group has 8 elements, several of its subgroups have 4, several have 2, and the
trivial group has 1 — plus each of these can be combined with a periodic-translation
lattice. Damage (25-45% of cells) then erased some tiles, marked `-1`.

You do not know `G`. You must infer it from the surviving evidence and restore the
grid. This is genuinely ambiguous where evidence has been wiped: whenever an entire
orbit was erased, no observation determines its color — the best you can do is match
the ornament's own remaining color mix, not guess an arbitrary default.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your answer) to **stdout**. It runs
in an isolated subprocess and sees only the public instance — never the hidden group
or the original colors.

```python
import sys, json
inst = json.load(sys.stdin)
# ... infer the symmetry group, restore the grid ...
print(json.dumps({"grid": completed_grid}))
```

### Public instance (stdin)

```json
{ "name": "ornament_1234", "n": 10, "k": 4,
  "grid": [[2, -1, 0, ...], [1, 2, -1, ...], ...] }
```
`n` is the side length, `k` the number of colors, `grid` an `n x n` array where `-1`
marks an erased cell and any other value is `0..k-1`.

### Answer (stdout)

```json
{ "grid": [[2, 3, 0, ...], [1, 2, 3, ...], ...] }
```
An `n x n` array of integers in `0..k-1`. Any cell that was **not** erased must be
echoed back unchanged; changing a known cell, wrong shape, an out-of-range value, a
crash, a timeout, or non-JSON output scores that instance `0.0`.

## Scoring (deterministic, computed by the evaluator itself)

For every orbit of the *true* hidden group:
- If the orbit has at least one surviving cell, its color is **determinable** — every
  cell in it should equal that survivor's true color.
- If the whole orbit was erased, it is **undeterminable** — no single correct color
  exists to check against.

The evaluator computes, per instance:
- `accuracy` — fraction of cells in determinable orbits your answer got exactly right,
- `consistency` — fraction of undeterminable orbits where you at least filled every
  cell in that orbit with the *same* color (internally coherent, even if unprovable),
- `stats` — one minus the total-variation distance between your color mix on
  undeterminable cells and the ornament's true color mix there.

These combine into one quality score (accuracy weighted most, consistency and stats
weighted less but non-trivially), then affine-normalized per instance against an
evaluator-internal no-symmetry baseline (copy known cells, mode-fill the rest) so
that baseline lands near `0.1` and perfect recovery approaches (but a real solver
should not reach) `1.0`. The reported **Ratio** is the mean over 10 fixed instances;
**Vector** lists the per-instance scores.

## Notes

Some instances repeat a small motif via wraparound translation as well as point
symmetry; several use symmetry groups with **no mirror axis at all** (pure rotation,
or pure translation) — checking only reflections will not find them. Damage levels
and patterns vary across the 10 instances, including cases engineered so that entire
orbits are wiped out.
