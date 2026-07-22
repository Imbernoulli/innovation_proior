# Snip the Superspreader Links

## Problem
An outbreak spreads over a weighted contact network. Under the standard SIS model the
epidemic **threshold** is governed by the **spectral radius** (largest eigenvalue)
`lambda(A)` of the symmetric weighted adjacency matrix `A`: the smaller `lambda`, the
harder it is for the disease to persist. You are given a budget of `k` contacts you may
sever. Choose **which** contacts to cut so the resulting spectral radius is as small as
possible.

## Input (stdin)
```
n m k
u_1 v_1 w_1
...
u_m v_m w_m
```
`n` nodes (1-indexed), `m` undirected weighted edges, budget `k`. Edge `e` (1-indexed in
listed order) joins `u_e` and `v_e` with integer weight `w_e >= 1`. No self-loops, no
repeated pair. `A[u][v] = A[v][u] = w` for each edge; all other entries `0`.

## Output (stdout)
Whitespace-separated **edge indices** (each in `1..m`) naming the contacts you cut — the
set of edges removed from the graph. They must be **distinct** and number **at most `k`**
(you may cut fewer, never more). Output nothing to cut nothing.

## Feasibility
Any token that is not an integer, an index outside `1..m`, a repeated index, or more than
`k` indices total makes the submission infeasible (score `0`).

## Objective
Let `lambda_0` be the spectral radius of the full graph and `lambda_S` the spectral radius
after removing your chosen edge set `S`. Removing edges can only lower the spectral radius,
so your achieved **eigen-drop** is `D(S) = lambda_0 - lambda_S >= 0`. **Maximize `D(S)`**
(equivalently, minimize `lambda_S`).

## Scoring
The checker computes `lambda` by deterministic power iteration (fixed all-ones start, fixed
iteration count). It builds a baseline drop `D_base` by cutting the first `k` listed edges,
then reports
```
Ratio = min(1.0, 0.1 * D(S) / D_base).
```
Cutting the first `k` edges scores `~0.1`; an eigen-drop `10x` the baseline caps the score.
The mechanism that matters: the first-order marginal drop from cutting edge `(i,j)` of
weight `w` is proportional to `w * x_i * x_j`, where `x` is the leading eigenvector
(eigenvector centrality) of the **current** matrix — not to `w` or to endpoint degree. The
heaviest edges need not sit on the spectral core, and each cut reshapes `x`.

## Constraints
`6 <= n <= 60`, `k <= 6`, `1 <= w <= 70`, time limit 5 s, memory 512 MB.

## Example
For `n=4`, edges `(1,2,10),(3,4,10),(2,3,3)` and `k=1`: cutting edge 3 (weight 3) splits the
graph and lowers `lambda` far more than cutting either weight-10 edge, whose endpoints have
smaller eigenvector-centrality product. Weight alone would mislead you.
