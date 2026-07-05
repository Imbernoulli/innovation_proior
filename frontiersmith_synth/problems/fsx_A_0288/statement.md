# Arena Synergy Tensor: Fewest Trilinear Gadgets

## Problem

The match engine of a global e-sports arena scores every draft with a **3-way synergy
tensor** `T[i][j][k]` — the bonus that **hero `i`**, **item `j`** and **map `k`** produce
together. There are `I` heroes, `J` items and `K` maps.

Per frame the engine must recompute the whole table. It does so with a **bilinear
circuit** built from *trilinear gadgets*. A single gadget is a triple of weight vectors
`(a, b, c)` (over the rationals) contributing

```
gadget(i,j,k) = a[i] * b[j] * c[k]
```

and the engine reconstructs the table as the sum of `R` gadgets:

```
T[i][j][k] = sum_{r=1..R} a_r[i] * b_r[j] * c_r[k]     for all i, j, k.
```

Each gadget costs **one scalar multiplication** on the hot path, so the engine's per-frame
cost is exactly `R`. Your job is a classic **tensor rank / arithmetic-complexity** task:
reproduce the synergy tensor **exactly** using **as few gadgets as possible**.

The shipped tensors are built so that every *map* (frontal) slice is a low-rank matrix,
yet the number of planted gadgets **exceeds every dimension** (the rank is *over-complete*).
There is therefore **no known formula for the optimum** — polynomial diagonalization
methods (which need rank ≤ dimension) do not apply, and multiple strategies of very
different quality exist.

## Input (stdin)

```
I J K
```
followed by `K` frontal slices; slice `k` is `I` lines of `J` integers, where the value on
line `i`, column `j` is `T[i][j][k]`.

## Output (stdout)

```
R
```
then, for each of the `R` gadgets, **three lines**:

- line 1: `a` — `I` rationals (mode-1 weights),
- line 2: `b` — `J` rationals (mode-2 weights),
- line 3: `c` — `K` rationals (mode-3 weights).

Each number may be an integer, a decimal, or an exact fraction `p/q` (e.g. `-3/4`).
`NaN`/`Inf` are rejected.

## Feasibility

The decomposition must reconstruct the tensor **exactly** (exact rational arithmetic):
`sum_r a_r[i] b_r[j] c_r[k] == T[i][j][k]` for every `(i,j,k)`. Any mismatch, malformed
schema, wrong token count, non-finite value, or `R` outside `[1, 3*I*J*K+50]` scores `0`.

## Objective

**Minimize `R`**, the number of gadgets (= scalar multiplications).

## Scoring

Let `F = R` be your gadget count and let `B` be the checker's internal baseline: the number
of non-zero frontal (mode-3) fibers — a trivial always-feasible decomposition. Then

```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```

So matching the trivial baseline scores `0.1`; a `10x`-smaller decomposition caps at `1.0`.
Lower `R` is strictly better. Scoring is exact and deterministic.

## Constraints

- `3 <= I,J,K <= 8`, integer entries.
- `1 <= R <= 3*I*J*K + 50`.
- Exact rational weights only; scoring never depends on time or hardware.

## Example

Suppose `I=J=2, K=1` and the single slice is
```
1 0
0 1
```
The trivial fiber decomposition uses `R=2` gadgets: `(e_0)(e_0)(1)` and `(e_1)(e_1)(1)`,
matching the baseline `B=2` (two non-zero fibers) for `Ratio = 0.1`. This 2x2 identity slice
has matrix rank 2, so no single gadget suffices here — but on the shipped instances the map
slices are planted low-rank, and the best decomposition is genuinely open. *(Illustrative
shape only — not one of the graded instances.)*
