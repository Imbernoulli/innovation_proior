# Metamaterial Field Layout — Tile a Graded-Index Panel (Format B, isolated)

You are laying out a flat metamaterial panel as a grid of `R x C` unit cells.
A fixed **library of `K` tile types** is available; type `t` is a periodic
unit cell containing an inclusion phase at volume fraction `v_table[t]`
embedded in a background matrix. Its **effective permittivity** is fixed by
the Maxwell-Garnett homogenization mixing rule:

```
eff(t) = em * (2*(1-v_t)*em + (1+2*v_t)*ei) / ((2+v_t)*em + (1-v_t)*ei)
```

where `em`, `ei` are the matrix/inclusion permittivities given in the
instance. `eff_table[t]` (already evaluated for you) gives `eff(t)` for
every library type — but note `eff_table` is **not** evenly spaced: the
mixing rule is nonlinear in `v_t`, so "index `t`" and "achieved property"
are related but not proportional.

Your job: choose a tile type for every one of the `R*C` grid cells so the
achieved property field tracks a given **target field**, while keeping the
panel physically buildable.

## Public instance (stdin JSON)
```json
{
  "R": 6, "C": 9, "K": 7,
  "v_table": [0.05, 0.19, ...],        // K increasing inclusion fractions
  "em": 1.4, "ei": 8.7,                 // matrix / inclusion permittivity
  "eff_table": [1.51, 1.83, ...],       // eff(t) for each library type t
  "target": [[row0...], [row1...], ...],// R x C target property, cell[r][c]
  "interface_weight": 0.5               // lambda, see scoring below
}
```

## Answer (stdout JSON)
```json
{"types": [[t00, t01, ..., t0(C-1)], ..., [t(R-1)0, ..., ]]}
```
An `R x C` grid of integer tile-type indices, each in `[0, K-1]`. Any other
shape, an out-of-range entry, or a non-finite value is rejected (score 0
for that instance); a value within `1e-6` of an integer is accepted as
that integer (floating-point tolerance), anything further from an integer
is rejected.

## Scoring
Two things determine cost, and you minimize their sum:

1. **Field-match error** — for cell `(r,c)` with chosen type `t`, the
   pointwise error is `|eff_table[t] - target[r][c]|`, averaged over all
   `R*C` cells.
2. **Interface penalty** — for every pair of 4-connected neighboring cells
   `(a,b)`, the physical mismatch cost is `(type_a - type_b)^2` (index jump
   between adjacent tiles, squared — a bigger jump in inclusion fraction
   between neighbors means a bigger manufacturability/coupling penalty at
   that seam). Averaged over all adjacent pairs, then scaled by
   `interface_weight`.

```
objective = mean_cells |eff_table[type] - target| + interface_weight * mean_edges (type_a - type_b)^2
```

Lower `objective` is better. Per instance:
`score = min(1, 0.1 * objective_uniform / objective_yours)`, where
`objective_uniform` is the cost of the best single tile type applied to
*every* cell (computed by the grader, not given to you). The final score is
the mean over 10 fixed, seeded instances of varying grid size, contrast
(`em`, `ei`), target-field shape, and interface weight.

There is no closed-form optimum: the field-match term wants each cell's
type chosen independently to match its own target value; the interface
term wants neighboring types to stay close together. These pull in
opposite directions wherever the target field is noisy or has a sharp
local feature, and the tradeoff is not the same in every region of the
grid — a good layout must be solved jointly, not cell by cell.

## Suggested strategies (increasing sophistication)
- **Uniform tiling** — one type everywhere (no field tracking, zero
  interface cost).
- **Nearest-match** — independently pick each cell's closest type to its
  own target value (ignores the interface term entirely).
- **One-shot smoothing** — nearest-match, then a single local-averaging
  pass to soften the worst jumps.
- **Joint relabeling** — treat this as a discrete labeling problem with a
  pairwise smoothness term (an MRF): repeatedly re-optimize each cell's
  type against both its own target *and* its current neighbors' types
  (iterated conditional modes / simulated annealing / graph-cut-style
  moves), trading a little pointwise accuracy for a much cheaper interface
  bill where it matters.
