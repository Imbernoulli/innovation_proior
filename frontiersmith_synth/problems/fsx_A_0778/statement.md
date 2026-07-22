# Museum Match Curve: Inverse Design Against Crowd-Out

## Problem
A museum runs a donation drive with `N` donors. Donor `i` has a **warm-glow strength**
`a_i > 0` and a **wealth cap** `w_i > 0` (they can give at most `w_i`). If donor `i` gives
own amount `g in [0, w_i]` and the museum's matching schedule adds `M(g)` on top of it,
the donor's utility is
```
U_i(g) = a_i * ln(1 + g + M(g)) - g
```
Each donor independently picks the `g` that **maximizes their own utility** (their "best
response"); you never choose on their behalf. `M` is a **piecewise-linear matching
schedule** you design: `M(0) = 0`, and it is defined by `K` segments via `K-1` increasing
breakpoints `0 < t_1 < t_2 < ... < t_{K-1}` and `K` per-segment rates `m_1, ..., m_K`
(`m_k` applies to the `k`-th segment `[t_{k-1}, t_k]`, with `t_0 = 0` and `t_K = +inf`):
for `t` in segment `k`, `M(t) = M(t_{k-1}) + m_k * (t - t_{k-1})`. A **flat 1:1 match**
is the special case `K=1, m_1=1`; a schedule that pays nothing below some threshold and
a positive rate above it is the special case `K=2, m_1=0`.

Because the marginal match rate can jump at a breakpoint, a donor's utility need not be
concave across the whole schedule -- their true optimum may lie at any segment's interior
stationary point OR at a breakpoint OR at their cap `w_i`; you must account for all of
these when reasoning about how donors will respond.

The museum has a match **budget** `B`: the total `M(g_i*)` paid across all donors (at
their best-response gifts `g_i*`) must not exceed `B`. Subject to that, you want to
maximize **total funds raised** `F = sum_i (g_i* + M(g_i*))`.

## Input (stdin)
```
N K_MAX R_MAX B
a_1 w_1
a_2 w_2
...
a_N w_N
```
`N` donors, `K_MAX` = max segments your schedule may use, `R_MAX` = max allowed rate on
any segment, `B` = match budget (all floats; `B > 0`).

## Output (stdout)
Whitespace-separated tokens (any line layout): first `K` (`1 <= K <= K_MAX`), then the
`K-1` breakpoints `t_1 ... t_{K-1}` in strictly increasing order (all `> 0`), then the
`K` rates `m_1 ... m_K` (each in `[0, R_MAX]`).

## Feasibility
- `1 <= K <= K_MAX`; exactly `K-1` breakpoints, strictly increasing, all `> 0`.
- Every rate finite and in `[0, R_MAX]`.
- All numbers finite (no `nan`/`inf`).
- The checker computes every donor's exact best response `g_i*` under your schedule
  (checking every segment's endpoints and interior stationary point) and the resulting
  total match payout. That payout must be `<= B` (tolerance `1e-6 * max(1,B)`).

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum_i (g_i* + M(g_i*))`, the total funds (own gifts plus match) the museum ends up
with once every donor has best-responded to your schedule.

## Scoring
Let `g0_i = clip(a_i - 1, 0, w_i)` be donor `i`'s optimum with **no** matching program at
all, and let `BASE = 0.4 * sum_i g0_i` -- a deliberately modest reference (well below
what even a naive matching program raises). With your feasible `F`,
```
sc    = min(1000, 100 * F / max(1e-9, BASE))
Ratio = sc / 1000
```

## Constraints
`3 <= N <= 320`, `2 <= K_MAX <= 4`, `R_MAX = 4.0`. Runs in well under the time limit.

## Example
Two donors, `K_MAX=2`, `R_MAX=4`, `B=6`: donor 1 `(a=20, w=60)`, donor 2 `(a=8, w=30)`.
With no match (`K=1, m_1=0`): donor 1's optimum solves `a/(1+g)=1 -> g=19`, donor 2's
gives `g=7`; `F = 26`, payout `0`. Now try `K=2` with `t_1=15, m_1=0, m_2=1.0` (shield the
first 15 dollars, match 1:1 above it): donor 1's segment-2 stationary point is
`g = a - (1 - m_2*t_1)/(1+m_2) = 20 - (1-15)/2 = 27`, paying `1.0*(27-15)=12 > B`, so this
particular schedule is infeasible here (illustrating why the rate must be backed out
against the *whole* population, not guessed). A feasible schedule might instead use a
smaller `m_2`; the checker performs this exact best-response computation for whatever
schedule you submit.
