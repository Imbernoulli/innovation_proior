# Kinetics From the Endpoint: A Perturbation Budget

## Problem

A source `S` feeds flux into `N` intermediate nodes of a reaction pathway.
Node `i` carries a **known** flux weight `w_i` (a positive integer). Node `i`
then splits its own flux among a fixed subset of final products (endpoints)
with a **hidden** branching-ratio vector: a probability distribution over
exactly the products it feeds. You are told, for every node, *which*
products it feeds (the pathway topology) and its flux weight `w_i` -- but
never the branching ratios themselves.

You are also given the **baseline product distribution**: for every product,
the flux-weighted sum of every node's contribution to it, as actually
observed. This is real data, but it is a lossy statistic. Whenever two or
more nodes feed the *exact same* set of products, baseline data can report
only their combined contribution -- infinitely many different individual
branching ratios for those nodes reproduce the identical baseline. Call such
a group of nodes (sharing an identical product set) a **cluster**; a cluster
of size 1 is already uniquely determined by baseline alone.

You may request up to `Q` **perturbation queries**. A query names one node
`i`: it isolates node `i` (routes all flux through it alone) and reveals its
branching-ratio vector exactly. You must decide, within the budget, *which*
nodes to query -- your output is simply the set of node ids you choose.

## Input (stdin)

Line 1: `testId N L Q W`.
Line 2: `N` integers `w_1 .. w_N`.
Next `N` lines: `deg e_1 .. e_deg` -- the sorted 1-indexed product ids node
`i` (`i = 1..N`, in order) feeds. Two nodes with the identical sorted product
list belong to the same cluster.
Last line: `L` floats -- the observed baseline product distribution
(context; the flux weights and topology above already determine everything
you need to decide where to spend the budget).

## Output (stdout)

Line 1: `M`, the number of queries you spend (`0 <= M <= Q`).
Line 2: `M` distinct integers in `[1,N]` -- the queried node ids.

## Feasibility (any violation scores `Ratio: 0.0`)

`M` must be an integer with `0 <= M <= Q`; all `M` ids must be distinct
integers in `[1,N]`; output must be well-formed (no extra/missing tokens,
no non-finite tokens).

## Scoring

A cluster of size `g` is **separated** once you have queried `>= g-1` of its
nodes (the last member is then pinned down exactly, by subtracting the known
members' contribution from the cluster's baseline total) -- every node in a
separated cluster is scored as exactly identified (accuracy `1.0`). A
cluster that is **not** fully separated (you queried `0..g-2` of its nodes)
is scored as if you had queried none of it: every member's estimate is the
flux-weighted average branching ratio of the whole cluster -- partial
queries into an unfinished cluster earn nothing by themselves. A node's
identification accuracy is `1 - 0.5*L1(estimate, truth)` (both are
probability vectors over the cluster's product set). The objective `F` is
the flux-weighted average of all `N` nodes' accuracy. The checker's internal
baseline `B` is `F` with an **empty** query set (spend nothing). Final score
`Ratio = min(1000, 100*F/B) / 1000`.

## Constraints

`4 <= N <= 45`, `3 <= L <= 45`, `1 <= Q <= 17`, time limit 5s.

## Example (worked, illustrative shape only -- not the hidden instance)

Suppose `N=3`, two nodes (1,2) share products `{1,2}` (a size-2 cluster,
cost 1 to separate) and node 3 alone feeds product `{3}` (already unique).
Spending your one query on node 3 wastes it (already known); spending it on
node 1 or node 2 separates the cluster, pinning down both nodes exactly and
raising `F` above the "query nothing" baseline `B`. A budget spent on the
*higher-flux* node in isolation, ignoring whether its cluster can actually
be finished, is not guaranteed to beat spending it on a smaller, fully
separable cluster instead -- that allocation choice is the entire problem.
