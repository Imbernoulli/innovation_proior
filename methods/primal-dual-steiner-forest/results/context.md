# Context: minimum-cost connection of given terminal pairs

## Research question

A telephone company has a list of client requests: each request names a pair of cities that must
be linked. There is an underlying graph `G = (V, E)` of possible fiber links, each link `e` carrying
a nonnegative installation cost `c_e`, and a link, once installed, may carry many clients at once.
The company must choose a set of links `F ⊆ E` so that for every requested pair `(s_j, t_j)` there
is a path from `s_j` to `t_j` using only chosen links, and the total cost `∑_{e∈F} c_e` is as small
as possible.

The subtlety that separates this from the classical Steiner tree problem is that **the requested
pairs need not all be tied together**. Different clients want different, possibly disjoint, pairs
connected; the cheapest legal answer can be a *disconnected* forest of several trees. Forcing one
connected network over all the terminals (a Steiner tree) can be arbitrarily more expensive than
the true optimum. The problem is NP-hard — it contains the Steiner tree problem, one of Karp's
original NP-complete problems — so an exact polynomial algorithm is not expected. The goal is a
polynomial-time algorithm whose output is provably within a small constant factor of optimal, and
the target constant is 2.

## Background

**Connectivity as a covering condition.** Whether `s_j` and `t_j` end up connected in the chosen
subgraph is governed entirely by cuts. For a vertex set `S`, write `δ(S)` for the edges with
exactly one endpoint in `S`. Call `S` *separating* for pair `j` if it contains exactly one of
`s_j, t_j`. By max-flow/min-cut, `s_j` and `t_j` are connected in `(V, F)` if and only if `F`
contains at least one edge of `δ(S)` for *every* separating `S`. So feasibility is exactly: for
every set `S` that separates at least one requested pair, `|F ∩ δ(S)| ≥ 1`. This is a covering
(or "hitting set") condition with one requirement per separating set — exponentially many of
them.

**Linear-programming relaxation and weak duality.** Assigning a fractional `x_e ≥ 0` to each edge
and minimizing `∑ c_e x_e` subject to `∑_{e∈δ(S)} x_e ≥ 1` for all separating `S` gives a lower
bound on the integer optimum. Its dual assigns a variable `y_S ≥ 0` to each separating set,
maximizing `∑_S y_S` subject to `∑_{S: e∈δ(S)} y_S ≤ c_e` for every edge `e`. Any feasible dual
`y` is, by weak duality, a lower bound on the cost of any feasible network. This is the lever: if
an algorithm produces a network of cost at most `α` times the value of a dual it also produces,
that network is within `α` of optimal — without ever solving the LP. Geometrically the dual
variables can be drawn as concentric rings ("moats") of width `y_S` around vertex sets; the
constraint says the moats piled onto any single edge cannot total more than its cost.

**Self-refining structure of the cut lower bound.** In the unit-cost case a feasible network has
size at least half the maximum number of cuts you can pack so that each separates some pair and no
edge lies in more than two cuts (each cut must be hit, and each network edge is counted in at most
two cuts). For integer costs, subdividing an edge of cost `c_e` into a path of `c_e` unit edges
carries the same statement to weighted instances; rational costs follow by scaling and real costs
by the usual limiting argument. This is the same cost-weighted lower-bound intuition that the
moat-packing dual makes explicit.

**Why the obvious primal-dual choice is not enough (the diagnostic).** A primal-dual covering
algorithm keeps a feasible dual and a partial primal, and at each step raises the dual of a single
*violated* set (one not yet hit) until some edge constraint becomes tight, then buys that edge. For
covering with sets of size at most `f`, this yields an `f`-approximation, because each tight edge's
cost is shared among at most `f` of the raised duals. For connection problems `f` is the size of a
cut, which can be as large as the number of pairs `k`. The failure is concrete: take one common
source `s = s_1 = … = s_k` and distinct sinks `t_1, …, t_k`. If the algorithm raises only the dual
of the singleton `{s}`, the cheapest legal completion buys all `k` star edges, every one of which
crosses `δ({s})`, so a single moat is charged `k` times — a `k`-approximation. Yet if one looks at
the same `k` star edges against *all* the natural violated sets `{s}, {t_1}, …, {t_k}`, each
`δ({t_j})` is crossed once and `δ({s})` `k` times: `k+1` sets, total crossings `2k`, an average of
`2k/(k+1) < 2`. The crossings are cheap *on average over the violated sets* even though they are
expensive against the one set the single-violation rule happened to pick.

## Baselines

- **Steiner tree heuristics (Takahashi–Matsuyama 1980; Kou–Markowsky–Berman 1981).** Build a tree
  over a single terminal set, e.g. via shortest-path metric closure and a spanning tree, giving
  factor `2(1 − 1/|T|)` for terminal set `T`. Core gap: they assume *all* terminals belong to one
  component; applied to a forest instance by connecting every terminal, they can be arbitrarily worse
  than the true (possibly disconnected) optimum.

- **Solve-the-LP-then-take-tight-edges (Hochbaum 1982).** Compute an optimal dual `y*`, then take
  every edge whose dual constraint is tight. Gives `f`-approximation. Two gaps: it must actually
  solve an LP with exponentially many constraints, and `f` (max cut size) is as bad as `k` for
  connection problems.

- **Single-violation primal-dual (Bar-Yehuda–Even 1981).** Grow one violated set's dual at a time;
  no need to solve the LP, only to keep raising a feasible dual. Gives `f`-approximation by the same
  charging. Gap: for Steiner forest the single-violation rule gives only `k`, as the star example
  shows.

- **Goemans–Bertsimas survivable-network tree heuristic (1993).** Handles requirements of the
  special form `r_ij = min(r_i, r_j)` by decomposing into a sequence of ordinary Steiner tree
  problems solved by a known heuristic, factor `2·min(log R, p)`. Gap: restricted to that special
  requirement form; it cannot handle arbitrary `0/1` pairs, where each subproblem is no longer an
  ordinary Steiner tree.

- **Reverse-delete cleanup as an analysis tool (introduced for primal-dual covering, then refined
  to delete in reverse order of addition).** After a feasible edge set is built, remove redundant
  edges. The refinement — examining edges for deletion in the *reverse* of the order they were
  added — turns the remaining edges, relative to any intermediate state, into a *minimal*
  augmentation, which is what makes a degree-counting bound possible. Without it, edges that become
  unnecessary after later mergers can survive, and an inactive component can remain as a leaf in the
  contracted counting graph.

## Evaluation settings

Instances are undirected graphs with nonnegative edge costs and a list of terminal pairs. The
natural special cases used to sanity-check a pair-connection algorithm are a shortest `s`–`t` path
(one pair), a minimum spanning tree (all vertices pairwise required), and a Steiner tree (one
terminal set whose vertices must all be connected). The yardstick for an approximation algorithm is
its solution cost against the optimum (computable by brute force on tiny graphs) and the worst-case
ratio over families of instances; the running time is measured in `n = |V|` and `m = |E|`.

## Code framework

The primitives that already exist: a graph with edge costs, a union-find structure to track which
vertices have been merged into a common component, and a feasibility test (are all requested pairs
connected by the chosen edges). The open slot is the rule for which edge to buy next and how to
certify the result is near-optimal.

```python
def connect_pairs(n, edges, pairs):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    # TODO: bookkeeping the selection rule will need
    #       (e.g. per-vertex labels and a component status test)

    chosen = []
    while True:
        # TODO: identify components whose boundary still matters
        # TODO: stopping condition (all pairs connected)
        # TODO: pick the next edge to buy from remaining edge slack
        # TODO: update the bookkeeping for the relevant components
        # TODO: merge the two components the bought edge joins
        break

    # TODO: cleanup pass that removes edges not needed for feasibility
    return chosen

def all_connected(n, forest, pairs):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (u, v, _c) in forest:
        parent[find(u)] = find(v)
    return all(find(s) == find(t) for s, t in pairs)
```
