# Mirror Pond Rake Budget

## Problem

A line of monks tends a long ceremonial pond meant to lie perfectly flat, cell
by cell, so it mirrors the sky. After a storm the pond has `N` cells in a row
(`N` a power of two), and cell `i` (0-indexed) has an integer height deviation
`h[i]` from flat (positive = bump, negative = dip, 0 = flat). The pond is
declared *polished* once no cell deviates from flat by more than a threshold
`T`.

The monks have exactly two tools, and every stroke is logged:

- **Point polish `P i`** (cost **1**): one monk shaves cell `i` by exactly one
  unit toward flat — `h[i] -= 1` if `h[i] > 0`, `h[i] += 1` if `h[i] < 0`,
  no effect (but still costs 1) if `h[i] == 0`.
- **Team rake `B a w`** (cost **`w`**, the number of cells touched): a crew
  rakes the aligned block `[a, a+w)`. `w` must be a power of two, `w >= 2`,
  and `a` must be a multiple of `w` with `a + w <= N` (the block must be one
  cell of the canonical size-`w` partition of `[0, N)`). Let `S` be the sum
  of `h[i]` over the block and `avg = floor(S / w)` (floor toward `-inf`,
  standard integer division). The crew subtracts the shared sag `avg` from
  every cell in the block: `h[i] -= avg` for all `i` in `[a, a+w)`.

You submit a **straight-line plan**: a fixed sequence of these ops, executed
once, in the order given. There is no branching and no feedback — you must
decide the whole schedule from the initial disturbance alone.

## Input (stdin)

```
N T MAXOPS
h[0] h[1] ... h[N-1]
```
`N` is a power of two (8..64 across tests). `MAXOPS` bounds how many op
*lines* your plan may contain (not the total cost).

## Output (stdout)

```
M
op_1
op_2
...
op_M
```
`M` is the number of op lines that follow (`0 <= M <= MAXOPS`); each `op_k`
is either `P i` or `B a w` as defined above.

## Feasibility

Execute all `M` ops in order against the given `h`. The plan is feasible iff:
- every op is well-formed and in range (valid index / valid aligned power-of-two
  block), and
- after all ops, `max_i |h[i]| <= T`.

Any violation scores **0**.

## Objective

Minimize the **total cost** = sum over your ops of (1 for each `P`, `w` for
each `B`) needed to reach a feasible state. Op costs are exact integers —
nothing about runtime is ever measured.

## Scoring

The checker executes your plan and, if feasible, compares your total cost
against its own simple reference construction (which polishes every cell
by point-strokes alone). Cheaper valid plans score higher, capped at 1.0;
a plan costing much more than the reference scores near the trivial floor.
Ratio is printed as `Ratio: <float in [0,1]>`.

## Constraints

- `8 <= N <= 64`, `N` a power of two; `1 <= T <= 4`; `|h[i]|` up to a few
  hundred; `MAXOPS = 20000`.
- Time limit: 5s. Memory: 512MB.

## Example (worked score)

`N=8, T=1, MAXOPS=50`, `h = [6, 6, 6, 6, 6, 6, 6, 6]` (the whole strip sagged
evenly by 6).

One rake clears it: `B 0 8` — `S=48`, `avg=6`, every cell becomes `0`.
Total cost = **8**. Residual `0 <= 1`: feasible.

Compare the point-only recipe: each cell needs `6-1=5` strokes, total cost
`40`. The rake plan is 5x cheaper here — but rake width is *not* free:
raking a block whose cells don't share a common sag (e.g. an alternating
`+A, -A, +A, -A, ...` strip) computes `avg ≈ 0` and wastes its whole width
for nothing, so blind raking is not automatically better than point polish.
Whether a rake pays for itself, and at what width, depends on how the
disturbance is actually shaped — measure before you commit the stroke.
