# Mountain Rescue Relay Density — Minimum Self-Overlap

## Problem

A long alpine ridge trail is divided into `n` consecutive segments, numbered `0 .. n-1`.
Rescue command must decide how many relay teams `f_i >= 0` to station on each segment `i`
(an integer relay strength, capped at `U` teams per segment).

During a full sweep, every pair of relay stations exchanges signals. The instantaneous
**relay congestion at lag `k`** is the number of ordered station-pairs whose segment indices
sum to `k`, weighted by their strengths:

```
g_k = sum over all i, j with i + j = k of  f_i * f_j            (0 <= k <= 2n-2)
```

`g_k` is exactly the self-convolution (autocorrelation) of the relay-density vector `f`.
The **peak congestion** is `P(f) = max_k g_k`. A plan that piles teams onto a few segments
creates a huge congestion spike; a plan that merely spreads teams uniformly still spikes in
the middle. The rescue planner wants the flattest possible autocorrelation **relative to the
total deployed force**.

Formally, define the scale-invariant congestion constant

```
              2 * n * P(f)
    c(f)  =  ---------------
               ( sum_i f_i )^2
```

Deploy relays so that `c(f)` is **as small as possible**. This is a discrete instance of the
first autocorrelation inequality: `c(f)` cannot be pushed below a universal constant (known to
lie strictly above ~1.5), so no plan reaches 0 — the frontier is genuinely open and admits many
distinct near-optimal shapes.

## Input (stdin)

One line:

```
n U
```

- `n` — number of trail segments.
- `U` — maximum relay strength allowed on a single segment.

## Output (stdout)

Exactly `n` non-negative integers `f_0 f_1 ... f_{n-1}` (whitespace-separated, any layout),
with `0 <= f_i <= U`. At least one `f_i` must be positive.

## Feasibility

An output is rejected (score 0) unless it parses as exactly `n` integers, every value is in
`[0, U]`, and the total `sum_i f_i` is strictly positive. Non-integer, non-finite, missing,
or extra tokens all fail.

## Objective

Minimize `c(f) = 2 * n * P(f) / (sum_i f_i)^2`.

## Scoring

Let `F = c(f)` be your objective and let `B` be the constant achieved by the reference
"concentrate teams on the lower half of the trail" plan (a block of `ceil(n/2)` unit stations).
Because this is a minimization, the score is

```
Ratio = min(1.0, 0.1 * B / F)
```

Reproducing the reference block scores `0.1`; spreading uniformly scores about `0.2`; a
carefully shaped, flat-autocorrelation deployment scores higher. The theoretical wall near
`c ~ 1.5` keeps the score well below `1.0`.

## Constraints

- `24 <= n <= 128`
- `U = 1000000`

## Example

For `n = 4, U = 1000000`, the uniform plan `f = [1, 1, 1, 1]` has autocorrelation
`g = [1, 2, 3, 2, 1]`, so `P = 3`, `sum = 4`, and `c = 2*4*3 / 16 = 1.5`. The reference
half-block `f = [1, 1, 0, 0]` has `g = [1, 2, 1, 0, 0]`, `P = 2`, `sum = 2`, and
`c = 2*4*2 / 4 = 4`, so `B = 4`. The uniform plan would score
`0.1 * 4 / 1.5 = 0.2667`. (This tiny `n` is illustrative only; test cases use `n >= 24`.)
