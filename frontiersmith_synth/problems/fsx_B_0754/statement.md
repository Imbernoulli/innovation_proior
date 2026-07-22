# Fair-Share Capacity Market

## Problem

You manage a data network with `E` numbered links `0..E-1` and `F` flows, each with a
**fixed route**: a nonempty set of link indices the flow must cross. Link `e` has existing
capacity `cap_e > 0`. You have a budget of `Budget` currency units and may purchase EXTRA
capacity `x_e >= 0` on each link, at cost `cost_e` per unit, subject to
`sum_e cost_e * x_e <= Budget`.

After your purchase, the network automatically settles into the **proportional-fair rate
equilibrium**: the unique rate vector `r_f >= 0` (one rate per flow) that is feasible — for
every link `e`, the sum of `r_f` over all flows crossing `e` does not exceed `cap_e + x_e` —
and that maximizes `sum_f log(r_f)`, the standard network-utility-maximization objective.
You do not output rates; the checker computes this equilibrium for you from your capacity
purchase.

## Output (stdout)

`E` nonnegative reals `x_0 x_1 ... x_{E-1}` (extra capacity bought per link), separated by
whitespace/newlines. Any violation — wrong count, a negative or non-finite value, or
`sum cost_e*x_e > Budget` (tolerance `1e-6`) — scores 0.

## Scoring

Let `V0` be the total log-utility using only the *existing* capacities (no purchase). Let
`F_you = V(cap + x) - V0` be your utility GAIN. Let `F_ref` be the utility gain of the
checker's own reference purchase, which spends your `Budget` in dollars **proportional to
each link's existing capacity** ("top up the biggest trunk first" — a plausible-sounding but
naive rule). Your score is `min(1, F_you / (10 * F_ref))`: matching the reference exactly
scores `0.1`; a purchase worth 10x the reference's utility gain saturates at `1.0`.

## Why this is nonlinear (read this before optimizing)

Adding capacity to link `e` only helps if `e` is actually a *binding* constraint for some flow
after the whole network re-equilibrates. If a link is crossed by many flows, its true marginal
value is the **sum of what each of those flows would gain at the margin** — a quantity you can
only know by resolving the *entire* network's equilibrium, not by inspecting any single flow or
link in isolation. A link touched by many flows is not automatically the best buy: if those
flows are already capped somewhere else on their route, pouring money into this link wastes the
budget entirely, no matter how many flows "use" it. Conversely a lightly-shared link that is one
flow's *only* route can be worth far more per dollar than the busiest link in the network.

## Input (stdin)

```
E F Budget
cap_0 cost_0
...
cap_{E-1} cost_{E-1}
L_0 id_1 id_2 ... id_{L_0}
...
L_{F-1} id_1 id_2 ... id_{L_{F-1}}
```
`L_f` is flow `f`'s route length, followed by `L_f` distinct link ids (`0..E-1`) it crosses.

## Constraints

`2 <= E <= 40`, `1 <= F <= 25`, `1 <= L_f <= E`, `cap_e, cost_e` in `[0.1, 300]`, `Budget` in
`[1, 100]`. Time limit 5s, memory 512MB.

## Example (illustrative shape only — not a graded test)

`E=2, F=2`: flow 0 = `[0]`, flow 1 = `[0, 1]`. `cap = [1, 1]`, `cost = [1, 1]`, `Budget = 2`.
With no purchase, both flows split link 0 evenly: `r = (0.5, 0.5)`. Buying `x = [2, 0]` raises
link 0's capacity to 3; the new equilibrium gives roughly `r = (1.5, 1.5)` (link 1 stays slack,
never binding), a large log-utility gain. In this toy example the two links happen to have
tied capacity, so the checker's "proportional to capacity" reference buys the same thing — on
the graded instances the reference and the well-reasoned purchase usually diverge sharply,
because some links only *look* important (many flows, big existing capacity) while carrying
none of the network's real scarcity.
