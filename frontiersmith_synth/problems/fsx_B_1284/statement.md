# ESG Screen: Excluding the Worst Without Wrecking the Tracking

## Problem
A benchmark index holds `N` names with weights `w_i >= 0` summing to `1`. Each name belongs
to exactly one of `S` sectors and has a continuous "size" loading. An ESG-screened fund must
**exclude** every name whose ESG score falls below a threshold `T` (its weight must be exactly
`0`) but still wants its portfolio to track the *original, unscreened* benchmark as closely as
possible, in the sense of **factor exposure**, not raw weight. Each surviving name also has a
**substitution capacity** `cap_i`: the most extra weight its available float can absorb (a name
cannot simply be given unlimited weight to plug a gap).

Choose portfolio weights `x_i` for the surviving names to **minimize the tracking error**
against the benchmark.

## Input (stdin)
```
N S
T
F[0][0] ... F[0][K-1]
...
F[K-1][0] ... F[K-1][K-1]
sector_1 size_1 esg_1 w_1 cap_1 d_1
...
sector_N size_N esg_N w_N cap_N d_N
```
`K = S + 1` (the `S` sector loadings plus one continuous "size" loading). `F` is the
`K x K` factor covariance matrix (symmetric, positive semi-definite). For name `i`:
`sector_i` in `{0,...,S-1}`, `size_i` its size loading, `esg_i` its ESG score, `w_i` its
benchmark weight, `cap_i >= w_i` its substitution capacity, `d_i > 0` its idiosyncratic
(name-specific) variance.

## Output (stdout)
Exactly `N` numbers `x_1 ... x_N` (whitespace/newline separated) -- the portfolio weight of
each name, in input order.

## Feasibility
- Every `x_i` is finite and `x_i >= 0` (tol `1e-6`).
- If `esg_i < T`: `x_i` must equal `0` (within tol) -- the name is excluded.
- If `esg_i >= T`: `x_i <= cap_i` (within tol) -- substitution cannot exceed capacity.
- `sum_i x_i = 1` (within tol `1e-4`).
Any violation scores `Ratio: 0.0`. (The instance always admits a feasible solution: total
capacity among eligible names comfortably exceeds `1`.)

## Objective (minimize)
Let `L_i` be name `i`'s `K`-dim loading vector: `1` in coordinate `sector_i`, `0` in the other
sector coordinates, and `size_i` in the last coordinate. Let `v_i = x_i - w_i`. The portfolio's
factor-exposure gap against the benchmark is `e_k = sum_i L_i[k] * v_i`. The tracking error is
```
TE = sqrt( e^T F e  +  sum_i d_i * v_i^2 )
```
The first term charges drift in aggregate factor exposure (sector and size tilts); the second
charges leftover name-specific drift. Smaller `TE = F_obj` is better.

## Scoring
The checker builds an internal baseline `B`: the tracking error of the **equal-weight**
portfolio over the eligible names (`x_i = 1/M` for each of the `M` eligible names, `0`
elsewhere). With your tracking error `F_obj`,
```
sc    = min(1000, 100 * B / max(1e-9, F_obj))
Ratio = sc / 1000
```
so matching the equal-weight baseline scores about `0.1`, and a ten-times-smaller tracking
error caps the score at `1.0`.

## Constraints
`16 <= N <= 68`, `S = 4` sectors fixed, `K = 5`. All computation is `O(N*K + K^2)`, well within
the time limit.

## Example (illustrative only, tiny and NOT the scored form)
`N=2, S=1` (so `K=2`), `w = (0.6, 0.4)`, both eligible, `cap = (1,1)`, sectors `(0,0)`,
sizes `(1,-1)`, `esg` both above `T`, `d=(0,0)`, `F = I`. Outputting `x=(0.6,0.4)` reproduces
the benchmark exactly: `v=(0,0)`, so `e=(0,0)` and `TE = 0`. Outputting `x=(1,0)` gives
`v=(0.4,-0.4)`, sector-gap `e_0 = 0.4-0.4=0`, size-gap `e_1 = 0.4*1 + (-0.4)*(-1) = 0.8`, so
`TE = 0.8`. Matching the benchmark exactly is always best when nothing is excluded; the real
instances force at least one exclusion, so `TE=0` is not achievable there.
