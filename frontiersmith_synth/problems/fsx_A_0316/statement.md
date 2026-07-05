# Deep-Sea Cable Network — Maximally-Decoupled Phase Coupling Matrix

## Problem

A trans-oceanic operator is laying an `N`-node deep-sea cable relay network. Every ordered pair
of relay nodes shares a phase-coupling line whose sign is either constructively aligned (`+1`) or
destructively inverted (`-1`). The full coupling configuration is an `N x N` matrix
`A` with every entry `A[i][j] ∈ {-1, +1}` (the diagonal `A[i][i]` is the node's self-loop, also `±1`).

The network's resilience to correlated failures is governed by the **decoupling margin**, defined as
the absolute value of the determinant of the coupling matrix, `|det(A)|`. A large `|det(A)|` means the
node responses are as linearly independent as physically possible, so no small set of severed cables
can collapse the network onto a degenerate subspace.

Your task: **output an `N x N` `±1` matrix that maximizes `|det(A)|`.**

This is the classical maximal-determinant `±1` problem restricted to **odd, prime `N`** (so no
Hadamard matrix exists and no closed-form optimum is known — the true maximum is open). Multiple
strategies are viable: algebraic quadratic-residue (Legendre) circulants, single-entry sign-flip
hill-climbing (`det` is multilinear in each entry, so a flip rescales it by an exactly computable
factor), simulated annealing, or block constructions.

## Input (stdin)

A single line containing the integer `N` (the number of relay nodes; an odd prime, `7 ≤ N ≤ 41`).

## Output (stdout)

`N` lines, each with `N` space-separated integers, every entry in `{-1, 1}`. Line `i` gives row `i`
of your coupling matrix `A`. Any entry not equal to `-1` or `+1`, a wrong number of rows/columns, or
any non-integer / non-finite token makes the submission **infeasible** (score `0`).

## Feasibility

- Exactly `N` rows, each with exactly `N` tokens (`N*N` integers total).
- Every token is exactly `-1` or `+1`.

## Objective (maximize)

Maximize `F = |det(A)|`, computed **exactly** by the checker via Bareiss fraction-free integer
elimination (no floating point, no tolerance).

## Scoring

The checker builds an internal deterministic baseline matrix `A0` (a lightly-perturbed Legendre
quadratic-residue circulant) and computes `B = |det(A0)|`. Your score is

```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000        # printed on the final line
```

Reproducing the baseline scores `Ratio ≈ 0.1`; reaching `10x` the baseline determinant caps at
`1.0`. The theoretical Hadamard bound `N^(N/2)` is not reachable for these odd orders, so genuine
headroom always remains.

## Constraints

- `N ∈ {7,11,13,17,19,23,29,31,37,41}` across the difficulty ladder.
- Deterministic exact-integer scoring. Time limit 5 s, memory 512 MB.

## Example (worked score)

For `N = 7`, the checker's baseline has `|det(A0)| = 384`. If you emit a matrix with
`|det(A)| = 512`, then `F/B = 1.333`, `sc = 133.3`, and `Ratio = 0.133`. A matrix reaching
`|det| = 3840` (ten times the baseline) would score the capped `Ratio = 1.0`.
