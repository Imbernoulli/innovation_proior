# Radiator Network: Multi-Scale Cooling-Curve Match

## Problem

A radiator network is a fixed graph of `n` nodes (radiator segments) and `m`
pipes (edges). Pipe `e` has an integer conductance cap `cap_e`; you choose an
integer conductance (weight) `w_e` with `1 <= w_e <= cap_e` for every pipe.

The network's heat-loss behaviour is summarized by the **heat-kernel trace**
of its weighted Laplacian `L_w` (the usual graph Laplacian built from your
chosen edge weights: diagonal = weighted degree, off-diagonal `-w_e` for each
edge):

```
Tr(exp(-t * L_w)) = sum_i exp(-t * lambda_i(L_w))
```

where `lambda_i(L_w)` are the eigenvalues of `L_w`. This quantity is exactly
`n` at `t = 0` and decays toward the number of connected components as
`t -> infinity`. You are given target trace values at a grid of times `t_1 <
t_2 < ... < t_T` spanning several decades (from `t_1` far below 1 to `t_T`
far above 1), and must choose integer edge weights whose trace curve matches
the targets as closely as possible, simultaneously at every one of these
times.

**Mechanism, stated honestly:** the trace near `t -> 0` is governed by the
low-order moments of `L_w` (total edge weight and the degree sequence);
the trace near `t -> infinity` is governed by the spectral gap (the second-
smallest eigenvalue of `L_w`, i.e. how weakly the two sides of the sparsest
bottleneck in the network are joined). The SAME weight vector controls both
regimes at once, through different combinations of the same numbers -- the
input does not tell you which pipes matter for which regime, or by how
much; that structure has to be worked out from the given topology and caps.

## Input (stdin)

```
n m T
u_1 v_1 cap_1
...
u_m v_m cap_m
t_1 t_2 ... t_T
g_1 g_2 ... g_T
```
`0 <= u_i, v_i < n` are 0-indexed node ids, `cap_i >= 1`. The graph
(topology only, i.e. every edge present regardless of weight) is connected.
`t_1 < ... < t_T` are positive reals. `g_1, ..., g_T` are the positive
target trace values.

## Output (stdout)

`m` integers `w_1 ... w_m` (whitespace/newline separated, any layout), the
chosen conductance for pipe `i` in the SAME order the pipes were listed on
input.

## Feasibility

Output must contain exactly `m` tokens, each parsing as an integer, with
`1 <= w_i <= cap_i`. Any violation (wrong count, non-integer token,
out-of-range value, non-finite value) scores `0`.

## Scoring

Let `F = max_j |ln(Tr(exp(-t_j * L_w))) - ln(g_j)|` be your worst-case
log-error across the `T` grid points (smaller is better). The checker also
computes `B`, the same error `F` obtained by the trivial construction
`w_e = floor(cap_e / 2)` for every pipe. Your score is

```
ratio = min(1.0, B / F)
```
(a feasible submission that matches the trivial construction's error scores
about `0.1`; a submission with much smaller worst-case log-error scores
higher, capped at `1.0`). The grid targets carry a small amount of
irreducible mismatch, so no integer weight vector reaches error `0`.

## Constraints

`8 <= n <= 30`, `12 <= m <= 55`, `T = 9`, `3 <= cap_i <= 10`. Time limit 5s,
memory 512MB.

## Example (worked score, illustrative shape only)

Suppose `n=4`, a 4-cycle with caps `[6,6,6,6]`, `T=2` with `t = (0.1, 5.0)`
and targets `g = (3.7, 1.05)`. The trivial construction sets all four
weights to `3`, giving some baseline error `B`. A submission of
`w = (5,5,1,1)` (heavier on two opposite pipes, near-minimal on the other
two) redistributes the same 12 units of total conductance to reshape the
spectral gap without changing the total weight much -- if this lowers the
worst-case log-error below `B`, it scores above `0.1`. (This tiny 4-cycle
example is only for illustrating the scoring mechanics; the real test cases
have larger, clustered topologies.)
