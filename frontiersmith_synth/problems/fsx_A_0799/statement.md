# Drum-Corps Bridge: Band-Gap Stiffness Allocation

## Problem

A footbridge is discretized into **S segments**. Mechanically it is a chain of
`S` springs and `S-1` unit point masses between two fixed anchors: spring `j`
(stiffness `k_j`, a positive integer) connects mass `j-1` to mass `j` (mass `0`
and mass `S` are the fixed anchors). You must choose the integer stiffness
`k_1, ..., k_S`, spending an **exact stiffness budget**: `sum k_j = BUDGET`,
every `k_j >= 1`.

This chain has `S-1` natural vibration modes with **eigenfrequencies**
`omega_1, ..., omega_{S-1}` — the square roots of the eigenvalues of the
tridiagonal stiffness matrix `K` where `K[i][i] = k_i + k_{i+1}` and
`K[i][i+1] = K[i+1][i] = -k_{i+1}` (0-indexed masses `1..S-1`).

A marching drum corps drives the bridge at a **comb of forcing frequencies**
`f_1, ..., f_F`, each with an importance weight `w_f`. Every forcing line
"locks on" to whichever bridge eigenfrequency is nearest it — the closer the
lock, the worse the resonant amplification. Your goal is to choose the
stiffness allocation that **minimizes total resonant amplification**:

```
Objective = sum_f  w_f / ( min_j |f - omega_j| + eps )
```

`eps` (given in the input) is the scoring resolution — the amplification of a
forcing line is largest when some `omega_j` sits almost exactly on top of it.

**Important, and NOT the illustrative point above:** the forcing comb in this
problem is planted **exactly on the eigenfrequencies of the perfectly uniform
allocation** (`k_j = BUDGET/S` for every segment), with extra weight on the
comb lines in the *middle third* of the mode spectrum. Simply moving stiffness
around near-uniformly (or scaling it up/down) still produces a spectrum that
looks like a rescaled/shifted copy of the uniform one and keeps colliding with
most of the comb — you cannot outrun a comb by shifting the spectrum. Breaking
the chain's translational symmetry (e.g. alternating heavy/light segments with
some period) can open a genuine **band gap** — a whole *range* of missing
eigenfrequencies — which is the only way to clear the densely-weighted middle
band at once.

## Input (stdin)

```
S BUDGET
eps
F
f_1 w_1
f_2 w_2
...
f_F w_F
```

`S` segments, integer `BUDGET >= S`, real `eps > 0`, `F` comb lines each with a
real frequency `f_i` and an integer weight `w_i`.

## Output (stdout)

`S` integers `k_1 ... k_S` (whitespace-separated) — your stiffness allocation.

## Feasibility

Rejected (score `0`) unless: exactly `S` finite integer tokens; every
`k_j >= 1`; `sum k_j == BUDGET` exactly.

## Scoring

Let `F_obj` be the objective above for your feasible allocation, and let `B`
be the same objective evaluated at the checker's internal baseline — the
perfectly uniform allocation (`BUDGET` split as evenly as possible). Since
this is a **minimization**:

```
sc    = min(1000, 100 * B / max(1e-9, F_obj))
Ratio = sc / 1000
```

The uniform allocation itself scores `Ratio = 0.1`; an allocation with `10x`
less resonant amplification than the uniform baseline caps at `1.0`.

## Constraints

- `6 <= S <= 20`, `BUDGET = 12*S` exactly, `1 <= F <= S-1`.
- deterministic scoring; symmetric eigendecomposition only (no randomness).
- time limit 5s, memory 512MB.

## Example (worked score, illustrative shape only)

For a tiny 3-segment chain (`S=3`, so 2 masses) with `BUDGET=9`, uniform
allocation `k=(3,3,3)` gives `K = [[6,-3],[-3,6]]`, eigenvalues `{3,9}`, so
`omega = (1.732, 3.0)`. Suppose a single comb line sits at `f=3.0, w=4`, with
`eps=0.1`. The uniform allocation collides exactly: `min|f-omega|=0`, so
`B = 4/0.1 = 40`. An allocation `k=(1,8,0)` is infeasible (`k_3<1`); a feasible
alternative `k=(1,7,1)` gives `K=[[8,-7],[-7,8]]`, eigenvalues `{1,15}`,
`omega≈(1.0, 3.873)`, `min|f-omega|≈0.873`, `F_obj ≈ 4/(0.873+0.1) ≈ 4.111`,
`sc ≈ min(1000, 100*40/4.111) ≈ 972.9`, `Ratio ≈ 0.973`. (This worked example
uses a shape too small to admit real periodic modulation — it only
illustrates the scoring arithmetic, not the intended strategy.)
