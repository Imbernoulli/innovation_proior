# The Ornament Ledger — orbit, area, or motif?

A workshop carves small square ornament grids and scores each one with a
**beauty law**. Every ornament is measured into a row of a ledger:

| symbol | meaning |
|--------|---------|
| `n` | grid side length |
| `g` | fold order of the ornament's symmetry (a plain 4-fold rosette has `g=8`; finer kaleidoscopes have larger `g`) |
| `D` | raw symmetry-defect count: how many cells break the grid's own rotation/reflection symmetry |
| `M` | number of symmetry **orbits** at that fold order, `M = (n*n) // g` |
| `A` | area, `A = n*n` |
| `K` | motif count: how many separate ink blobs the grid contains |
| `H` | spacing entropy of the gaps between motifs (already scale-free) |
| `B` | the hidden **beauty score** (the training label) |

The true law has the shape `B = (a term built from D and its correct
normalisation, with one nonlinearity) + (a term built from H) + (a term built
from K and its correct normalisation) + (a constant)`, with unknown weights.
Recover a closed-form expression for `B`.

**The training workshop only ever carved ordinary 4-fold rosettes** (`g = 8`
on every row), so in training `M` is always exactly `A / 8` — the two columns
carry identical information. The grading ledger comes from a *different*
workshop, bigger ornaments at *finer* fold orders (`g` up to 20), where `M`
and `A` pull apart. Fitting the training data is not enough — you must
identify *which* of `M` (per orbit), `A` (per area), or `K` (per motif) the
law actually divides by, since only one still means the same thing once the
fold order changes.

## Input (stdin)
- Line 1: two integers, the row count and a case id.
- Next rows: `n g D M A K H B`, one ornament measurement each (`n,g,D,M,A,K`
  integers, `H,B` floats).

## Output (stdout)
One line: a closed-form Python expression for `B` in the variables `n`, `g`,
`D`, `M`, `A`, `K`, `H`. Allowed: `+ - * / **`, unary `-`, numeric constants,
and the functions `sqrt log exp abs`. Example (illustrative **form only —
NOT the hidden law**): `2.0 * D / K + 0.5 * n`. No other names are accepted.

## Scoring (deterministic, minimization)
Your expression is evaluated on a **held-out ledger**, regenerated inside the
grader, whose grids are bigger and whose fold orders `g` were never seen in
training. Let `p_i` be your prediction and `t_i` the true (noisy) beauty
score at held-out row `i`:

```
metric   = mean_i  min(1, |p_i - t_i| / (|p_i| + |t_i|))     # bounded rel. error
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant predictor mean(train B)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error gives a higher `Ratio` (capped at `1.0`). A constant
predictor scores about `0.1`. `LAMBDA` lightly penalizes an overgrown
expression. A non-finite prediction on any held-out row scores the whole
submission `0`.

## Why the obvious fit is a trap
Regressing `B` linearly on the raw columns `D`, `H`, `K` fits the training
ledger well — but a *linear* combination of raw counts can never reproduce a
*ratio* like "defect per orbit", and since every training row has `g = 8`,
nothing in training distinguishes "divide by the orbit count" from "use a
bigger raw coefficient". On the held-out ledger the grids are larger and the
fold order varies, so `D` and `K` drift to values the training-fitted
coefficients were never calibrated for, and the prediction diverges.

The fix: the law decomposes into three separately normalised pieces. The
defect term wants `D / M` (per orbit — **not** per area, and **not** per
motif `K`, both of which drift for the wrong reason once `g` or the grid
content changes); the motif term wants `K / A` (per area); the entropy term
is already scale-free. A linear fit on these three *engineered* features
recovers weights that survive the regime shift.

## Constraints
- Time limit 5 s, memory 512 MB; row count up to a few hundred.
- Held-out noise leaves irreducible error, so even a correct law does not
  reach `Ratio = 1.0` — there is room above the reference solutions.
