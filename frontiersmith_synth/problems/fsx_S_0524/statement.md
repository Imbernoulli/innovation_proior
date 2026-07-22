# Loom Draft Symmetry Forge

## Problem
You design a handloom **weaving draft**. The loom has `S` shafts and `T` treadles.
A draft is three parts:

- **threading** — `N` integers, `threading[j] in [1,S]`: the shaft that warp column `j` is on;
- **tie-up** — a `T x S` bit matrix `tieup[t][s] in {0,1}`;
- **treadling** — `N` integers, `treadling[i] in [1,T]`: the treadle used on weft row `i`.

These generate the woven **drawdown**, an `N x N` binary image

```
P[i][j] = tieup[ treadling[i] ][ threading[j] ]      (1-indexed lookups)
```

`P[i][j] = 1` means the warp shows at that cell, `0` means the weft shows. Because the image is
this product, it has at most `S` distinct column patterns and `T` distinct row patterns — you never
paint pixels directly; every change to one `threading`/`treadling` entry rewrites a whole
column/row class of `P` at once, and a tie-up bit flips a whole class-block.

Design a draft that is both **symmetric** and **motif-rich**, without weak long floats.

## Input (stdin)
```
N S T L k
ch cv cr
lnum lden
```
`N` (odd) is the drawdown size, `S`,`T` the shaft/treadle budgets, `L` the maximum float run,
`k` the motif window size. `ch cv cr` are non-negative symmetry weights. `lam = lnum/lden`.

## Output (stdout)
Exactly `2*N + T*S` whitespace-separated integers, in order:
`threading` (`N` values in `[1,S]`), then `treadling` (`N` values in `[1,T]`), then the tie-up
as `T` rows of `S` bits (row `t` = `tieup[t][0..S-1]`).

## Feasibility (all required, else score 0)
- ranges: `threading in [1,S]`, `treadling in [1,T]`, tie-up bits in `{0,1}`, exact token count;
- **float cap**: no *float* — a maximal run of equal values along any row or any column of `P` —
  may exceed length `L` (this holds for both warp floats and weft floats).

## Objective (maximize)
Let, over all `N*N` cells,
- `a_h` = fraction with `P[i][j] = P[i][N-1-j]` (left–right mirror),
- `a_v` = fraction with `P[i][j] = P[N-1-i][j]` (top–bottom mirror),
- `a_r` = fraction with `P[i][j] = P[N-1-i][N-1-j]` (180° rotation),
- `sym = (ch*a_h + cv*a_v + cr*a_r) / (ch+cv+cr)`.

Let `D` = number of **distinct** `k x k` windows appearing anywhere in `P`, and
`div = D / min( (N-k+1)^2 , S^k * T^k )` (the denominator is the true window budget: only
`S^k * T^k` windows can ever occur). Then

```
F = sym * ( lam + (1 - lam) * div ).
```

Raising `sym` and raising `div` both help, but they pull in different directions in the
factored search space, and the small alphabet caps how many distinct motifs exist — so the best
attainable `F` is genuinely open.

## Scoring
The checker builds an internal plain-weave baseline `B` (a tabby draft) and reports
`Ratio = min(1000, 100*F/B) / 1000`. A plain weave scores about `0.1`; higher is better.

## Constraints
`31 <= N <= 199` (odd), `2 <= S,T <= 8`, `1 <= k <= 8`, `3 <= L <= 8`. Time limit 5 s,
memory 512 MB.

## Example (illustrative, not an optimal draft)
`N=5, S=2, T=2, L=3, k=2, ch=cv=cr=1, lam=1/10`. Output `threading = 1 2 1 2 1`,
`treadling = 1 2 1 2 1`, tie-up `[[0,1],[1,0]]`. Then `P[i][j] = (i+j) mod 2`, a checkerboard:
every float has length 1 (feasible), `sym = 1` (odd `N` makes all three agreements exact), but
only two distinct `2x2` windows exist, so `div` is tiny and `F ≈ lam`. A richer, still-symmetric
draft scores much higher.
