# Pocket Atlas: Spectrum-Preserving Sparsifier Duel

## Problem
A cartography service maintains a weighted graph **G** on `n` nodes and `m` edges: the "atlas",
where edge weight is a connectivity strength. Every query the service ever actually answers is
one of a **published, frozen family of K test vectors** `x^(1), ..., x^(K) in R^n` (given to you
in the input) — each query's answer depends only on the quadratic form

```
Q_G(x) = x^T L_G x = sum over edges (u,v,w) of  w * (x_u - x_v)^2
```

where `L_G` is the (weighted, combinatorial) Laplacian of `G`. You must ship a **pocket atlas**
`H`: a subset of at most `s` edges of `G` (no new edges — only edges that exist in `G`), where
each kept edge's weight may be **turned down but never up** (`0 <= w'_e <= w_e`, dimming or
dropping only). `s` is far below what you would need to faithfully preserve `Q` for *every*
direction in `R^n` — but you only ever have to answer the `K` published queries.

## Input (stdin)
```
n m s K
u_1 v_1 w_1
...
u_m v_m w_m
x^(1)_1 ... x^(1)_n
...
x^(K)_1 ... x^(K)_n
```
- `1 <= u_i, v_i <= n`, `u_i != v_i`, `w_i > 0` (up to 3 decimals). `G` is simple and connected.
- Each `x^(k)` is a length-`n` real vector (up to 3 decimals). Guaranteed `x^(k)^T L_G x^(k) > 0`
  for every `k`.

## Output (stdout)
```
k
p_1 q_1 r_1
...
p_k q_k r_k
```
`k` (`0 <= k <= s`) is how many edges you keep; each `(p_i, q_i)` must be one of the input's
`m` edges (either endpoint order), each `r_i` its new weight.

## Feasibility
Output is feasible iff: `k <= s`; every `(p_i, q_i)` (as an unordered pair) is a distinct edge
of `G`, listed at most once; every `r_i` is finite, `0 <= r_i <= w_{(p_i,q_i)} + 1e-6` (the
edge's original weight — no boosting). Any violation scores `0`.

## Objective
For your chosen `H`, let `Q_H(x^(k))` be the same quadratic form evaluated with `H`'s edges and
weights (missing edges contribute nothing). Minimize the **worst-case relative error** over the
published family:

```
F(H) = max over k=1..K of | Q_H(x^(k)) / Q_G(x^(k)) - 1 |
```

## Scoring
The checker also builds an internal baseline `B`: the `s` heaviest original edges of `G`, kept
at their original weight (a construction that never looks at the `x^(k)` at all), and computes
its own `F_base` the same way; `B = F_base`. Your score is

```
Ratio = min(1.0, 0.1 * B / F(H))
```

Matching the baseline's error scores about `0.1`; every 10x reduction in worst-case relative
error relative to `B` adds toward the cap of `1.0`.

## Constraints
- `n <= 150`, `m <= 170`, `K = 4`, `s = 19` (fixed).
- Time limit 5 s, memory 512 MB.

## Example
`n=4`, one edge `(1,2,w=2)`, `s=1`, `K=1`, `x^(1) = [1,0,1,0]` (illustrative FORM only, not the
real test data): `Q_G(x^(1)) = 2*(1-0)^2 = 2`. Output `1\n1 2 2.0` reproduces `H=G` exactly on
this vector: `Q_H = 2`, `F = 0`. If instead you dropped the edge (`k=0`), `Q_H = 0`, `F = 1.0`
(worst possible on this single-vector toy case).
