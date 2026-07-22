# One Price List, Five Tribes

## Problem
A shop sells `M` products. Product `j` has production cost `cost_j` and a maximum
allowed markup `markup_j`: you must post an integer price `p_j` with
`cost_j <= p_j <= cost_j + markup_j`. You post **one** price vector; it is shown to
everyone.

Customers come from `K = 5` cohorts ("tribes"). Each cohort `k` contains one or more
customer **types**; type `t` has a population count `n_t`, an outside-option utility
`o_t` (the value of walking away and buying nothing), and a value `v_{t,j}` for owning
each product `j`.

**Deterministic discrete choice.** Given your prices, a customer of type `t` computes,
for every product `j`, the surplus `v_{t,j} - p_j`, and looks at `U_t = max_j (v_{t,j} - p_j)`
(ties broken by the smallest index `j`). If `U_t > o_t` the customer buys that
utility-maximizing product; otherwise they buy nothing. This choice rule is fixed and
known to you -- it is not something you get to influence, only exploit.

## Input (stdin)
```
M K
cost_1 markup_1
...
cost_M markup_M
T_1
n_1 o_1 v_{1,1} v_{1,2} ... v_{1,M}
...
T_2
... (cohort 2's types)
...                        (K = 5 cohorts total, in order)
```
Each of the `K` cohorts first states its type count `T_k`, then lists `T_k` lines of
`n o v_1 ... v_M`. All values are non-negative integers.

## Output (stdout)
Exactly `M` integers `p_1 p_2 ... p_M` (whitespace-separated, any layout) -- one price
per product.

## Feasibility
- Exactly `M` integer tokens, each finite.
- `cost_j <= p_j <= cost_j + markup_j` for every `j`.
Any violation scores `Ratio: 0.0`.

## Objective
For a cohort `k`, its **average profit per customer** is
`(sum over types t in k of n_t * profit_t) / (sum over types t in k of n_t)`,
where `profit_t = p_{j*} - cost_{j*}` if type `t` buys product `j*`, else `0`
(a customer who defects to the outside option contributes zero profit -- pricing a
cohort out of the menu entirely is the worst thing you can do to it). Your objective
`F` is the **minimum** of the five cohorts' average profits: the menu is only as good
as the tribe it treats worst. Maximizing the *sum* of profit across cohorts is a
different (and, as shown below, often a bad) proxy for maximizing this minimum.

## Scoring
The checker also builds its own modest, uniform-markup price vector as an internal
baseline `B` (feasible, using the same discrete-choice rule) and reports
`Ratio = min(1000, 100 * F / B) / 1000`. Matching the baseline scores ~0.10; clearing
it by a wide margin pushes the ratio toward 1.0 (capped).

## Constraints
`6 <= M <= 16`, `K = 5`, `1 <= T_k <= 2`, all costs/markups/values/populations fit in
32-bit integers, time limit 5s, memory 512MB.

## Example (worked, illustrative shape only)
`M=1`, one product, `cost_1=10, markup_1=10`. One cohort has a single type
`n=1, o=0, v_1=15`. Checker baseline prices at `10+max(1,10//5)=12`; that type buys at
utility `15-12=3>0`, profit `2`, so `B=2`. If you price at `p_1=14`
(utility `1>0`, still buys), your profit is `4`, giving `F=4` and
`Ratio=min(1000,100*4/2)/1000=1.0` -- but if you price at `p_1=16` the type's utility
`15-16=-1<=0`, they walk away, `F=0`, `Ratio=0.0`. The real instances have 5 cohorts
and up to 16 products, so the same "don't price a tribe out" logic must hold
simultaneously for all five at once.
