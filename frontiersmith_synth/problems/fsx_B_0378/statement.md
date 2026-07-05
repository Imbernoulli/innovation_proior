# Asteroid-Belt Buffer Allocation across a Refining Tree

A mining flotilla runs a rooted tree of refining stations. Every station buffers a shared reagent
to meet an uncertain per-shift demand. Reagent is scarce: the total buffer you may pre-position is
capped by a **budget**. When a station runs short, its **parent** station can route leftover reagent
DOWN to cover the shortfall (single-level risk pooling) at a per-unit transfer cost. Unmet demand is
expensive; leftover reagent that is never used still costs holding. Decide how much buffer to place
at each station to minimize expected total cost, subject to a service-level floor.

You write a **standalone program**: read ONE JSON instance from stdin, write ONE JSON answer to
stdout. Your program is run in isolation once per instance and only ever sees the public inputs
below. The realized demand scenarios used for scoring are held out.

## Public instance (stdin JSON)
```
{
  "n":              int,              # number of stations, root is index 0
  "parent":         [int]*n,          # parent[i] = parent station index; parent[0] = -1 (root)
  "h":              [float]*n,        # holding cost per unit of leftover buffer at station i
  "p":              [float]*n,        # shortage penalty per unit of unmet demand at station i
  "t":              [float]*n,        # transfer cost per unit routed from i down to its children
  "mean":           [float]*n,        # announced mean demand at station i
  "std":            [float]*n,        # announced demand std-dev at station i
  "budget":         float,            # cap on the sum of buffer levels: sum(stock) <= budget
  "service_target": float,            # required overall fill rate (fraction of demand met)
  "n_scenarios":    int               # number of held-out demand scenarios used for scoring
}
```
Demand at station `i` in each held-out scenario is an independent draw of `max(0, Normal(mean[i],
std[i]))`. You are given `mean`/`std` but NOT the realized draws.

## Answer (stdout JSON)
```
{"stock": [float]*n}      # non-negative buffer level placed at each station
```

## Feasibility (violation => score 0 on that instance)
- Every `stock[i]` finite and `>= 0`.
- `sum(stock) <= budget`.
- Overall fill rate over the held-out scenarios `>= service_target`.

## Cost model (per scenario, summed and averaged over scenarios)
For each scenario with realized demand `D`:
1. Each station's local surplus `s_i = max(stock_i - D_i, 0)` and deficit `def_i = max(D_i - stock_i, 0)`.
2. Each parent station routes `transfer_i = min(s_i, sum of children deficits)` down, split across its
   deficit children in proportion to their deficits. Covered demand is removed from the child's deficit.
3. Cost `= sum_i [ h_i * (s_i - transfer_i)  +  t_i * transfer_i  +  p_i * residual_deficit_i ]`.

The objective is the mean cost over all held-out scenarios (lower is better).

## Scoring
For each instance, `ratio = min(1, 0.1 * baseline / your_cost)`, where `baseline` is the cost of the
trivial equal-split allocation (`stock_i = budget / n`). The equal-split allocation scores ~0.1; a
strong allocation scores higher. The final score is the mean ratio over all instances, which include
harder held-out instances (costlier shortages, tighter budgets, higher volatility). Scoring is fully
deterministic. There is no closed-form optimum: the budget couples all stations and the tree pooling
rewards concentrating buffer at cheap-holding parents, so multiple strategies are viable.
