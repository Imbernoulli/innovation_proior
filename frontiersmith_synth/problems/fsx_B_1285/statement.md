# Discovery or Settle: A Staged Litigation Policy

## Problem
A plaintiff's case will go through `T` rounds of discovery before it must resolve.
Model the case as a **complete binary tree of depth `T`**, heap-indexed: node `0` is
"before any discovery"; a node `i` at depth `d < T` has two children `2i+1` ("this
round's signal came back unfavorable") and `2i+2` ("...favorable"), reached after one
more round of discovery. Each round's signal is an independent fair coin flip — this
is the **information-arrival schedule**. Nodes at depth `T` are terminal: no discovery
remains, and the case must resolve there.

Every node `i` (at any depth, including terminal ones) carries a **settlement value**
`S[i]`: the net amount the plaintiff would recover by settling there, reflecting the
opponent's reservation price given everything revealed so far (the
**opponent-reservation-model**). Every terminal node also carries a **trial payoff**
`L[j]`, the net recovery if the case is actually tried once all discovery is in.
Reaching depth `d` means `d` rounds have been paid for: round `k` (`1<=k<=T`) costs
`cost[k]`, and these accrue along every path that reaches it (**cost-accrual**) —
these round costs need not be equal, and are *not* necessarily increasing.

You submit a **policy**: a decision for every node, `S` (settle right there) or `C`
(continue). At a non-terminal node, `C` means: pay the next round's cost, and the case
proceeds to *both* children with probability 1/2 each (you don't choose which). At a
terminal node, `C` means going to trial. The realized value of a node is its
settlement/trial figure minus the total cost accrued to reach it; for a `C` at a
non-terminal node, the node's value is the average of its two children's realized
values (the exact expectation over the fixed, fair information process — no
simulation is needed to compute it).

Settling on round 0 avoids all cost but locks in today's price, forgoing whatever
discovery would have revealed. Committing to trial everywhere pays every round's cost
on every branch, even branches that turn out weak. A policy that revisits the decision
at each node can do better than either fixed commitment by riding favorable branches
deeper and cutting losses on unfavorable ones.

## Input (stdin)
```
T
cost[1] cost[2] ... cost[T]
S[0] S[1] ... S[M-1]              (M = 2^(T+1) - 1, heap order, all depths)
L[0] L[1] ... L[2^T - 1]          (terminal-node trial payoffs, leaf order)
```
All values are integers (settlement/trial figures may be negative, representing a net
loss). `1 <= T <= 5`.

## Output (stdout)
```
M
d[0] d[1] ... d[M-1]
```
`M` must equal `2^(T+1)-1`; each `d[i]` is exactly the character `S` or `C`, the
decision at heap-indexed node `i`.

## Feasibility
- The first token must parse as the exact integer `M = 2^(T+1)-1`.
- Exactly `M` decision tokens must follow, each exactly `S` or `C`.
Any violation scores `Ratio: 0.0`.

## Objective
Define `value(i)` recursively over the submitted policy: if `d[i]=S`,
`value(i) = S[i] - accrued_cost(i)`; if `d[i]=C` and `i` is terminal,
`value(i) = L[leaf(i)] - accrued_cost(i)`; if `d[i]=C` and `i` is non-terminal,
`value(i) = (value(2i+1) + value(2i+2)) / 2`. `accrued_cost(i)` is the sum of
`cost[1..d]` for a node at depth `d`. Maximize `F = value(0)`.

## Scoring
`B` = the checker's own trivial construction: settle at node `0` immediately, so
`B = S[0]` (always positive by construction — this is always a feasible policy).
```
sc = min(1000.0, 100.0 * max(0,F) / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the immediate-settlement baseline scores `Ratio = 0.1`.

## Constraints
- `1 <= T <= 5`, so `M <= 63` and up to `32` terminal nodes.
- All input integers fit in `[-2000000, 2000000]`.
- Time limit 5s, memory 256m.

## Example
`T=1`, `cost=[5]`, `S=[20, 20, 20]`, `L=[20, 60]`. Settling at node 0 gives `F=20`
(`Ratio=0.1`). Continuing (`d=[C,?,?]`, the `?`s at the leaves being `C` too, i.e.
going to trial on both branches) gives `F = (20 + 60)/2 - 5 = 35`, i.e.
`sc=100*35/20=175`, `Ratio=0.175` — here going to trial always happens to already be
optimal (this is a warm-up case; larger cases in the test set are not this simple).
