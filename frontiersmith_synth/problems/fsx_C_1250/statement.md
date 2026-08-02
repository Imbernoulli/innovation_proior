# Multiplying Almost Right: Compensated Truncation for MAC Chains

## Problem
A chip has `K` multiply-accumulate (MAC) units, grouped into `G` accumulation chains
(e.g. dot-product lanes). Chain `g` sums the products of its member MAC units:
`sum_{i in g} a_i * b_i`, evaluated on `S` given sample vectors.

A full-precision multiplier is expensive in area. An **approximate multiplier**
saves area by clearing the low `t` bits of each operand before multiplying — a
"truncation depth" `t` (0 = full precision). Two truncation *modes* are available
per multiplier:

- **mode 0 (floor)**: `trunc0(x,t) = (x >> t) << t` — always rounds a value DOWN,
  so its error is one-sided. Summed across many multipliers in one chain, this bias
  does not cancel — it drifts, growing roughly with chain length.
- **mode 1 (compensated)**: `trunc1(x,t)` rounds `x` to the *nearest* multiple of
  `2^t` (adds `2^(t-1)` before clearing, for `t>=1`). Its error can be positive or
  negative, so across a long chain the errors tend to cancel rather than drift —
  but enabling mode 1 on a multiplier costs `comp_extra` extra area (given in the
  input) on top of its truncation-depth area.

The chip's area table `area[0..TMAX]` (given in the input, strictly decreasing) is
the area cost of a multiplier at each truncation depth under mode 0; using mode 1
adds `comp_extra` to whichever depth's area you pick. Every multiplier's `(t_i,c_i)`
may be chosen independently.

## Input (stdin)
```
K G S TMAX comp_extra
area[0] area[1] ... area[TMAX]
```
then `G` lines, one per chain (chain 0 first, etc.):
```
L_g B_g
```
(`L_g` = number of MAC units in the chain, `B_g` = its absolute error budget), then
`K` lines total, grouped contiguously by chain (the first `L_0` lines are chain 0's
units, the next `L_1` are chain 1's, ...). Each line lists that unit's `S` sample
operand pairs:
```
a_1 b_1 a_2 b_2 ... a_S b_S
```
All operands are non-negative integers.

## Output (stdout)
Exactly `K` lines, in the same position order as the input: `t_i c_i`, with
`0 <= t_i <= TMAX` and `c_i` in `{0,1}`.

## Feasibility
For chain `g` and sample `s`, let `exact` be the true sum of products and `approx`
be the sum using each member's `(t_i,c_i)` (mode 0 or 1 truncation applied to both
operands before multiplying). The output is feasible iff, for **every** chain and
**every** sample, `|approx - exact| <= B_g`. Any parse error, wrong token count,
out-of-range/non-finite token, or budget violation scores 0.

## Objective
Minimize total area: `sum_i area[t_i] + (comp_extra if c_i == 1 else 0)`.

## Scoring
Let `B = K * area[0]` (the checker's own baseline: every multiplier at full
precision, `t=0,c=0` — always feasible since it has zero error). With your total
area `F`:
```
Ratio = min(1, 0.1 * B / F)
```
Full precision everywhere scores `0.1`. Halving total area doubles the ratio;
reaching a tenth of `B` caps at `1.0`.

## Constraints
- `1 <= K <= 100000`, `1 <= G <= K`, `1 <= S <= 1000`, `0 <= TMAX <= 30`.
- `area[0] > area[1] > ... > area[TMAX] > 0`; `comp_extra >= 0`.
- Deterministic exact-integer scoring; no timing.

## Example (illustrative only — smaller numbers than any real test)
`TMAX=2`, `area=[10,7,5]`, `comp_extra=3`, one chain of `2` units, budget `B_0=4`.
If mode-0 truncation at `t=2` on both units gives chain error `6 > 4` (infeasible)
but `t=1` gives error `3 <= 4`, submitting `t_i=1,c_i=0` for both units costs
`F=2*7=14` against baseline `B=2*10=20`, scoring `min(1,0.1*20/14)=0.143`.
