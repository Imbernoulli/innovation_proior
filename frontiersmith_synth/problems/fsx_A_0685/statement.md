# Quarry-Approved Plaza Paving

## Problem

A circular plaza has `n` cell positions arranged around a ring, labelled
`0, 1, ..., n-1` (position `n-1` is adjacent to position `0`). You must pave it
with a *tile pattern*: choose a **tile** `B`, a set of `k` cell offsets, and a
list of `n/k` **translation offsets** `T`. The paving places a copy of `B` at
every offset in `T`: cell `x` is considered covered once for every pair
`(b, t) in B x T` with `(b + t) mod n == x`. A **perfect** paving covers every
one of the `n` cells *exactly once*.

The quarry only approves certain positions for cutting a tile stone: a mask
`a_0, ..., a_{n-1}` (`a_i in {0,1}`) marks which positions may be used **as an
element of B**. Every cell of `B` must be quarry-approved; `T` has no such
restriction (translation offsets are just numbers, not physical stones). Each
quarry-approved position `i` also has a fee `c_i >= 1` (charged once per stone
you actually place in `B`, not per copy).

It is guaranteed `n = k*k` for the given `k` (so `n/k = k` too), and that a
perfect paving is always achievable in principle: every one of the `k` residue
classes `{i : i mod k == j}`, `j = 0..k-1`, contains at least one
quarry-approved cell. However, **no contiguous run of k consecutive positions
(cyclically) is ever fully quarry-approved** — you cannot just carve one
arc-shaped stone and rotate it around.

Read the instance, then print your tile and offsets.

## Input (stdin)

```
n k
a_0 a_1 ... a_{n-1}
c_0 c_1 ... c_{n-1}
```

## Output (stdout)

```
b_1 b_2 ... b_k        (the tile B, k distinct quarry-approved positions)
t_1 t_2 ... t_{n/k}     (the translation offsets T)
```

## Feasibility

`|B| = k`, all distinct, each in `[0, n-1]`, each quarry-approved
(`a_{b_i} = 1`). `|T| = n/k`, each in `[0, n-1]` (repeats allowed, though they
will hurt your score). Any violation, wrong token count, or non-integer/
non-finite token scores `0.0`.

## Objective (minimize)

Let `count[x]` be how many times cell `x` is covered. Define:
- `D = sum_x |count[x] - 1|` — the **defect** (0 = perfect paving).
- `C = sum` of `c_b` over `b` in `B` — the total quarry fee of your tile.
- `W = 1 + sum` of `c_i` over all quarry-approved `i` — a fixed normalizer
  computed from the input.
- `F = 0.35 * (D / n) + C / W`.

You minimize `F`. Because `D/n` and `C/W` are both scaled to roughly `[0,1]`,
reducing the defect by even one cell is worth far more than any cost saving —
get the paving right first, then chase the cheapest stones.

## Scoring

The checker builds its own reference paving (always defect 0) and compares
your `F` against its `F_ref` on a fixed decreasing scale; a smaller `F` gives a
larger score, capped at `1.0`. Score `0.0` on any infeasible output.

## Constraints

`5 <= k <= 37` (`k` prime), `25 <= n <= 1369`, `1 <= c_i <= 97`.

## Example (worked score, illustrative shape only)

Suppose `n=9, k=3`, classes are `{0,3,6}`, `{1,4,7}`, `{2,5,8}`, and (for this
toy example only) every position is approved with fee `1`. Picking `B={0,1,2}`
and `T={0,3,6}` covers every cell exactly once (`D=0`), and among defect-0
tiles the checker further rewards choosing the cheapest approved cell per
class. This illustrates the *shape* of the scoring, not the real instances,
whose approved-cell pattern is far sparser and never includes a full run of
`k` consecutive cells.
