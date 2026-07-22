# Trunk Pricing

## Problem

A logistics operator runs **K independent commodities**. Commodity *k* must ship
exactly `d_k` units from a source to a sink across its own small routing network:
a directed acyclic graph on `n_k` nodes (source = node 0, sink = node `n_k-1`) with
`m_k` capacitated, costed edges. Routing one unit of flow across an edge costs
`cost` and uses up to `cap` units of that edge's capacity.

Some edges are **trunk edges** (`shared = 1`): they draw on a single **global
backbone** whose total usage, summed across *every* commodity, is capped at `C`
units. Routing one unit of flow on a trunk edge with `weight = w` consumes `w`
units of the backbone. Non-trunk edges (`shared = 0`, `weight = 0`) are
commodity-local and never touch the backbone.

Drop the backbone constraint and every commodity's min-cost-flow problem is
completely independent of the others — the backbone budget is the *only* thing
coupling the K commodities together.

## Input
```
K
for each commodity k = 1..K, in this exact order:
  n m s t d
  m lines: u v cap cost shared weight
C
```
`s`, `t` are always the local source/sink indices `0` and `n-1`. Every edge has
`u < v` (the graph is acyclic). `weight = 0` whenever `shared = 0`; when
`shared = 1`, `weight` is a positive integer.

## Output
Print, in the SAME order as the input (commodity by commodity, then edge by edge
within a commodity), one non-negative integer per edge: the flow you route on it.
Total integers printed = the sum of all `m_k`. Whitespace/newlines between numbers
are both fine.

## Feasibility
1. Every printed flow is an integer with `0 <= flow_e <= cap_e`.
2. Flow conservation holds at every intermediate node of every commodity; the
   source of commodity k has net outflow exactly `d_k`; the sink has net inflow
   exactly `d_k`.
3. **Global coupling constraint**: summed over ALL commodities and ALL trunk
   edges, `sum(flow_e * weight_e) <= C`.

Any violation (bad token, wrong count, non-finite value, capacity breach, broken
conservation, or a busted backbone budget) scores `Ratio: 0.0`.

## Objective
Minimize total cost: `sum over every edge of flow_e * cost_e`.

## Scoring
The checker also builds an internal baseline **B**: the cheapest way to satisfy
every commodity's demand using *only* its non-trunk edges — i.e. never touching
the backbone at all. This is always achievable (every commodity has a trunk-free
route with enough capacity for its full demand) and therefore always feasible.
Your score is
```
score = min(1, B / (10 * F))
```
where `F` is your total cost. Never touching the backbone scores ≈0.1; using it
well pushes the score well above that; the score saturates at 1.0 only if your
cost undercuts `B` by more than 10x (it won't, on the harder cases).

## Example (worked, illustrative shape only)

One commodity, `d = 5`. Trunk edge: cap 3, cost 2, `shared=1, weight=1`. Bypass
edge: cap 10, cost 9, `shared=0`. Backbone budget `C = 3` (exactly enough for
this commodity alone). Routing 3 units on the trunk and 2 on the bypass costs
`3*2 + 2*9 = 24`. The trunk-free baseline routes all 5 units on the bypass:
`5*9 = 45`. Score = `min(1, 45/(10*24)) = 0.1875` — using the trunk as much as
the backbone allows beats avoiding it.

## Constraints
`1 <= K <= 9`, `n_k = 3`, `3 <= m_k <= 4`, `4 <= d_k <= 12`. Capacities and
demands are positive integers `<= 15`. Costs are non-negative integers `<= 70`
(a non-trunk completion edge may legitimately cost 0). `weight` is 0 for every
`shared = 0` edge and a positive integer `<= 6` for a trunk edge. `C >= 1`.
Time limit: 5s. Memory: 512MB.
