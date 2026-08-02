# State-Dependent Glidepath: De-Risking on Funded Ratio and Horizon Together

## Problem

A pension fund has `T` years left until its liabilities come due. It holds
assets `A` (initially `A0`) against a liability `L` (initially `L0`;
`funded ratio = A/L`). Each year `t = 1..T` it must choose one number: the
fraction `w(t) in [0,1]` of that year's investable pool to put in a risky
asset (the rest goes to a safe asset). The industry-standard practice is a
**linear age-based glidepath**: reduce `w` in a straight line from a young,
risky level down to an old, safe level, based on age alone. This problem
asks you to do better by making the allocation a function of **both** the
current funded ratio and the years remaining, not of age alone.

You are given `M` deterministic **scenario blocks** -- each one a full
`T`-year path of returns and rate shocks. You must submit **one** policy
grid that will be replayed, unmodified, against every block. The grid is
indexed by year `t` and by which of 5 **funded-ratio buckets** the fund is
in at the start of that year (boundaries given in the input, e.g.
`[0.80, 1.00, 1.20, 1.50]` splits funded ratio into 5 ranges). Two
different blocks that reach the same (year, bucket) state are forced to
reuse the same weight -- you cannot hand-tailor a block-by-block answer.

**Per-year mechanics** (year `t`, `remaining = T - t + 1`):
- Bucket `b` = which range `FR_prev = A/L` (before this year) falls in.
- **Contribution flexibility**: a contribution `C = c_base * flex[b]` is
  added before investing (`flex` given in the input; it is *larger* when
  underfunded -- catch-up contributions -- and *smaller* when overfunded).
- **Liability duration match**: `L *= clip(1 + g - remaining*dr, 0.5, 1.5)`,
  where `g` is the year's baseline liability growth and `dr` is that
  year's interest-rate shock. The **remaining years act as the liability's
  duration**: the same rate shock moves `L` far more when many years remain
  than when few do -- this is why sequence risk is sharpest right before
  the horizon, not gradually across the whole horizon.
- **Sequence-of-returns risk**: `A = (A + C) * (1 + w*r_risky + (1-w)*r_safe)`,
  with `w = w(t,b)` your submitted weight for that state. Because this
  compounds sequentially, the *order* of good and bad years matters, not
  just their average.

## Input (stdin)
```
T M
A0 L0 c_base
b1 b2 b3 b4              (4 boundaries -> 5 buckets: <b1, [b1,b2), [b2,b3), [b3,b4), >=b4)
flex0 flex1 flex2 flex3 flex4
M blocks, each T lines:  r_risky r_safe dr g
```

## Output (stdout)
`T` lines, each with 5 floats `w(t,0) w(t,1) w(t,2) w(t,3) w(t,4)` for
`t = 1..T` in order (t=1 first). Every value must lie in `[0,1]`.

## Feasibility
Reject (score 0) on: wrong token count (must be exactly `5*T` numbers),
a non-numeric token, or any value that is non-finite or outside `[0,1]`.

## Scoring
Replay every scenario block under your grid. For a block, let
`fr = A_T / L_T` at the horizon. Its score is `min(1.5, fr)` if `fr >= 1`,
else `fr**2` (a shortfall is punished super-linearly -- getting close
still counts for something, but missing badly counts for much less). Let
`F` be the mean of this over all `M` blocks. Let `B` be the same mean
computed under the fully-immunized baseline grid (`w = 0` in every state
-- take zero risk, ever). `Ratio = min(1.0, 0.1 * F / B)`.

## Constraints
`6 <= T <= 30`, `3 <= M <= 7`, all rates given to 6 decimals. Time limit 5s.

## Example (illustrative form only -- not a real test case)
Toy 2-year, single-block, single-bucket case: `A0=100, L0=100`, year 1
`(r_risky=0.10, r_safe=0.02, dr=0, g=0.05, C=0)`, year 2 the same. Choosing
`w=1` both years: `A1 = 110`, `L1 = 105`; `A2 = 121`, `L2 = 110.25`; final
`fr = 1.097`, score `1.097`. This only shows the update arithmetic --
which bucket you land in, and hence which weight even applies, depends on
the funded-ratio path, which this toy example does not exercise.
