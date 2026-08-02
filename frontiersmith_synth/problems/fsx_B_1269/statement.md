# Treaty-Shopping Router: Routes, Not Rates

## Problem

A multinational wants to move a principal amount `V0` from an operating
company in jurisdiction `0` up to an ultimate parent in jurisdiction `n-1`,
through a network of bilateral tax treaties. Every treaty link `u -> v` has
a withholding rate `rate(u,v)` (in basis points, taken out of whatever
arrives at `u`) plus a required **holding period** `hold(u,v)` (periods the
funds must sit at `u` before this hop) and an **instrument type**
(`0` or `1`) that the transfer is characterized as (debt-like vs.
equity-like).

You choose a **route**: a sequence of treaty links from `0` to `n-1`. The
route's net-of-tax value is `V0 * product(1 - rate(u,v)/10000)` over its
hops — cheaper hops compound into more money delivered. But a route is only
usable if it clears two compliance rules that apply to the **route as a
whole**, not hop-by-hop:

**Timing window.** Let `T = sum(hold(u,v))` over the route's hops. The
route is compliant only if `T_min <= T <= T_max` (both given in the input):
too fast looks like an artificial round-trip, too slow misses the filing
window and triggers a cross-border timing mismatch.

**Substance / anti-conduit test.** Every intermediate jurisdiction `x` on
the route has an economic-substance score `substance(x)`. Let `benefit_bp =
sum(max(0, baseline_rate_bp - rate(u,v)))` over the route's hops (how much
treaty benefit, in basis points, the whole route claims versus the
non-treaty statutory rate `baseline_rate_bp`). This sets a required
aggregate substance `required = ceil(gamma * benefit_bp / 10000)` (`gamma`
given in the input). For each intermediate `x` reached via hop-in
instrument type `t_in` and departing via hop-out instrument type `t_out`,
its *effective* substance is `substance(x)` if `t_in == t_out`, else
`substance(x) // 2` (a characterization mismatch across `x` weakens its
credited substance). The route is compliant only if the sum of effective
substance over all its intermediates is `>= required`.

A route that fails either rule is worth 0, however cheap its hops look
individually — routing decisions and compliance cannot be decided edge by
edge.

## Input (stdin)
```
n m
V0 baseline_rate_bp gamma T_min T_max
substance_0 substance_1 ... substance_{n-1}
m lines: u v rate_bp hold instrument_type
b
b0 b1 ... b_{b-1}
```
Jurisdiction `0` is the source, `n-1` is the target. The `m` edge lines
each describe one directed treaty link. The final two lines give a fixed
reference route (`b` node ids) that is always structurally valid and
compliant — provided purely as a convenience baseline, not as a hint about
the best route.

## Output (stdout)
```
k
v0 v1 ... v_{k-1}
```
`k` followed by the `k` jurisdiction ids of your chosen route: `v0 = 0`,
`v_{k-1} = n-1`, and every consecutive pair `(v_i, v_{i+1})` must be one of
the given treaty links. No repeated jurisdictions, no extra tokens.

## Feasibility
Reject (score 0) on: malformed/non-integer output, wrong endpoints, a
non-existent hop, a repeated jurisdiction, timing outside `[T_min,T_max]`,
or aggregate effective substance below `required`.

## Scoring
Let `F` be your compliant route's net-of-tax value (0 if infeasible). Let
`B` be the net-of-tax value of the reference route from the input.
`Ratio = min(1.0, 0.1 * F / B)`.

## Constraints
`4 <= n <= 22`, hops per route `<= 5`, `rate_bp` in `[50,2400]`,
`hold` in `[1,3]`. Time limit 5s.

## Example (illustrative form only, not a real test case)
Two hops, `0 -> 1 -> 2`, rates `500` and `1500` bp, `V0 = 1000`:
net value `= 1000 * 0.95 * 0.85 = 807.5`. This only shows the arithmetic
of compounding hop rates; whether `0 -> 1 -> 2` is even *compliant* depends
on `1`'s substance, the instrument types of both hops, and the timing
window — none of which this toy example fixes.
