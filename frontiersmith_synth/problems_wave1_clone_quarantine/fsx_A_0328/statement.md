# Greenhouse Coupling Tensor: Fewest Actuator Gadgets

## Problem

An automated greenhouse recomputes, every control tick, a **3-way coupling table**
`T[i][j][k]` — the growth response produced **jointly** by **light-zone `i`**,
**nutrient-band `j`** and **humidity-tier `k`**. There are `I` light-zones, `J`
nutrient-bands and `K` humidity-tiers.

The controller must rebuild the whole table each tick. It does so with a **bilinear
circuit** made of *actuator gadgets*. A single gadget is a triple of weight vectors
`(a, b, c)` (over the rationals) contributing

```
gadget(i,j,k) = a[i] * b[j] * c[k]
```

and the controller reconstructs the table as the sum of `R` gadgets:

```
T[i][j][k] = sum_{r=1..R} a_r[i] * b_r[j] * c_r[k]     for all i, j, k.
```

Each gadget costs **one scalar multiplication** on the hot path, so the per-tick cost is
exactly `R`. Your task is the classic **tensor rank / arithmetic-complexity** problem:
reproduce the coupling tensor **exactly** using **as few gadgets as possible**.

The shipped tensors are built as an **asymmetric low-multilinear-rank (Tucker) tensor**:
every mode-slicing is rank-deficient, but by a *different* amount, so which orientation is
cheapest varies per instance. The planted structure is **over-complete** — the true tensor
rank is genuinely unknown. Polynomial diagonalization (which needs rank ≤ dimension) does
not recover the optimum, and several strategies of very different quality exist.

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

- line 1: `a` — `I` rationals (mode-1 / light-zone weights),
- line 2: `b` — `J` rationals (mode-2 / nutrient-band weights),
- line 3: `c` — `K` rationals (mode-3 / humidity-tier weights).

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

- `1 <= I,J,K <= 64`, integer entries (shipped instances are large: dims 6..11).
- `1 <= R <= 3*I*J*K + 50`.
- Exact rational weights only; scoring never depends on time or hardware.

## Example

Suppose `I=J=2, K=1` and the single slice is
```
1 0
0 1
```
The trivial fiber decomposition uses `R=2` gadgets: `(e_0)(e_0)(1)` and `(e_1)(e_1)(1)`,
matching the baseline `B=2` (two non-zero fibers) for `Ratio = 0.1`. This 2×2 identity
slice has matrix rank 2, so no single gadget suffices here — but on the shipped instances
the coupling slices are planted low-rank across an asymmetric core, so the best
decomposition is genuinely open. *(Illustrative shape only — not one of the graded
instances.)*
