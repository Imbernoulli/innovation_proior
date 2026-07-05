# Conflict-Free Recycling Depot Routes

## Problem
A regional recycling operator runs a sorting facility with **n** sequential sorting
stages. Each stage sends a container to exactly one of **3** downstream bins, labelled
`0`, `1`, `2`. A **route** for one container is therefore a vector
`r = (r_1, ..., r_n)` with each `r_i ∈ {0,1,2}` — a point of `F_3^n`.

The dispatcher wants to commission as many distinct routes as possible, but the belt
mechanics forbid certain *triples* of routes. Three **distinct** routes `a, b, c` form a
**conflict triple** when, in **every** stage `i`, the three chosen bins are either all
equal or all different — equivalently

```
a_i + b_i + c_i ≡ 0 (mod 3)   for all i = 1..n.
```

(This is exactly the "no three collinear points in `F_3^n`" / cap-set condition: any
line of `F_3^n` contains exactly 3 points, and a conflict triple is a full line lying
inside the chosen set.)

In addition, a maintenance list marks some routes as **blocked** (belt offline for that
bin pattern); a blocked route may never be commissioned.

You must output a set `S` of routes that is **conflict-free** (contains no conflict
triple) and uses **no blocked route**. Your score grows with `|S|`.

## Input (stdin)
```
n  m
<m lines>   each: r_1 r_2 ... r_n   (a blocked route, r_i ∈ {0,1,2})
```
- `4 ≤ n ≤ 8`, `0 ≤ m`, the `m` blocked routes are distinct.

## Output (stdout)
```
k
<k lines>   each: r_1 r_2 ... r_n   (a commissioned route)
```
- `k` = number of routes you commission, then the `k` routes, one per line.
- Every route must have exactly `n` entries in `{0,1,2}`.

## Feasibility
An output is feasible iff:
1. `0 ≤ k ≤ 3^n` and exactly `k` routes follow, each with `n` entries in `{0,1,2}`;
2. all `k` routes are pairwise distinct;
3. no route is blocked;
4. `S` contains no conflict triple.
Any violation scores `0`.

## Objective
Maximize `|S| = k` (a maximization / `optimize` objective).

## Scoring
Let `F = k` be your feasible set size and let `B` be the size of the internal baseline
construction: the **half sub-cube** `{0} × {0,1}^(n-1)` (stage 1 pinned to bin `0`, the
remaining stages ranging over bins `{0,1}`) with all blocked routes removed. This is
always conflict-free, since three `{0,1}`-vectors sum to `0 mod 3` in a stage only if
they are all equal there, forcing the three routes to coincide. Then

```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```

Reproducing the baseline scores `Ratio ≈ 0.1`; a set 10× larger caps at `1.0`. There is
no known closed-form optimum for large `n`, so the ceiling is genuinely open.

## Constraints
- `4 ≤ n ≤ 8`. Deterministic scoring; exact integer arithmetic only.

## Example
`n = 4`, no blocked routes. The baseline half-cube `{0} × {0,1}^3` has `B = 8` routes
and is conflict-free. Suppose you instead commission a set of `16` conflict-free routes.
Then `F = 16`, `sc = 100*16/8 = 200`, `Ratio = 0.200`. (The numbers above are
illustrative of the scoring arithmetic, not an optimal target.)
