# Context: fast approximate search over a combinatorial decision graph

## Research question

A great many problems reduce to the same shape: a solution is built by a
sequence of decisions, each decision branches into several alternatives, and we
want the sequence whose final cumulative score is best. Decode a sentence from
acoustic frames (each frame chooses among phonetic transitions); schedule jobs
on a machine (each step chooses which job to run next); pack a knapsack (each
item is taken or skipped). The decision graph is layered: layer `t` holds the
partial solutions that have made `t` decisions, and a node at layer `t` fans out
into a handful of children at layer `t+1`.

The trouble is breadth. If every node has branching factor `b`, layer `t` holds
up to `b^t` nodes; for a depth-`D` problem an exact frontier is `b^D` — millions
of phonetic paths for a single utterance, `2^n` subsets for an `n`-item
knapsack. Holding or even enumerating that frontier is hopeless. The question is
how to search such a graph **fast enough to be practical and deterministic in
runtime**, while keeping the final answer close to the best one — accepting that
we may not prove optimality, as long as the score is high and the time is
bounded. Concretely, a usable search must run in time and memory that grow
**linearly in the depth**, not exponentially, and must not depend on backing up
out of dead ends.

## Background

The decision graph is searched by maintaining a *frontier* (the set of partial
solutions currently live) and repeatedly expanding frontier nodes into their
children. Several classical disciplines for choosing what to expand were on the
table, each with a known failure mode at this scale.

**Cost of completeness.** The exact search disciplines below all share one
property: they are *complete* or *optimal*, and they pay for that guarantee with
a frontier (and therefore a memory and time bill) that is exponential in the
worst case. At the scale of speech decoding or large combinatorial instances
that bill is simply unpayable. This is the diagnostic fact that motivates giving
up the guarantee on purpose.

**Backtracking is expensive in a large graph.** A search that commits to a path,
hits a dead end, and unwinds to try another wastes the work along the abandoned
path; in a deep graph with a large effective branching factor the unwinding
dominates, so backtracking is the dominant cost for any committed-path method
at this scale.

**Dynamic programming collapses a graph exactly, when the graph is small
enough.** When the partial-solution state is summarized by a small label (e.g. a
network state at a frame), distinct paths reaching the same label can be merged
and only the best one kept. Sweeping all labels forward, frame by frame, finds
the globally optimal path with no backtracking, in time proportional to
(#labels × depth). This is the Viterbi/forward sweep over a transition network.
It is exact and backtrack-free — but its cost is the *full* label set updated at
every step, which is exactly the breadth we cannot always afford when the
network is large or the per-label update is heavy.

**A heuristic evaluation can be cheap or expensive, and the expensive one is the
accurate one.** Ranking a partial solution by how promising it is requires an
evaluation function. A cheap local score (the immediate cost of the last
decision) is fast but myopic; an accurate global score (an estimate of the total
cost of the best completion) is what actually orders nodes well, but it is
costly to compute. Both scoring functions are available off the shelf.

These pieces — a layered decision graph, a frontier, an evaluation function, a
desire to avoid backtracking, and an exact full-breadth sweep that is correct
but too slow — are the raw materials. The pain is uniform: full breadth is
exact but exponential (or at best full-label) in cost; a single committed path
is cheap but fragile and needs backtracking.

## Baselines

**Breadth-first search.** Expand the frontier in FIFO order, finishing all of
layer `t` before any of layer `t+1`. It is complete — if a solution exists it is
found, and the shallowest one first. But the frontier *is* the entire layer, so
memory and time grow as `b^t`. It exploits no pruning against a best-so-far
because complete solutions sit at the deepest layer, reached last.

**Best-first search / A\*.** Order the frontier by an evaluation `f(n) = g(n) +
h(n)`, where `g(n)` is the cost accumulated to reach `n` and `h(n)` estimates
the remaining cost to a goal. When `h` is *admissible* — a lower bound on the
true remaining cost, `h(n) ≤ C*(n)` — A\* is complete and returns an optimal
solution; with a *consistent* `h` (`h(n) ≤ c(n,n') + h(n')`) it never needs to
reopen a node. The gap it leaves: A\* still keeps every generated-but-unexpanded
node on the frontier, and for hard problems that frontier grows exponentially.
Optimality is bought with unbounded memory.

**Branch and bound.** Keep an incumbent (the best complete solution found so
far). For each subproblem compute a bound on the best score reachable inside it;
if that bound cannot beat the incumbent, prune the whole subtree. It is sound —
it never prunes the optimum — and it can cut enormous swaths of the graph when
the bounds are tight. But the guarantee is still "optimal", so the worst case is
still exponential; a breadth-first exploration order in particular reaches
complete solutions only at the bottom, so it has no good incumbent to prune
against early and its memory stays high.

**A single greedy path (hill-climbing).** Keep exactly one partial solution; at
each layer extend it by its single best-looking child. Memory is `O(1)` per
layer and time is linear in depth — exactly the budget we want — but it is
brittle: one locally attractive but globally poor decision dooms the whole run,
and there is no second candidate to fall back on. It throws away too much
breadth.

The shape of the gap is now sharp. Full-breadth disciplines (BFS, A\*,
branch-and-bound, full-label DP) are exact but pay exponential or full-label
breadth; the single greedy path pays linear cost but keeps so little breadth
that one bad turn ruins it. The available disciplines cluster at these two
extremes.

## Evaluation settings

The natural yardsticks are problems where a forward, layered search is the
idiomatic solver and where an exact reference is computable on small instances
so the quality gap can be read off directly.

- **Connected speech recognition over a transition network.** Decode an
  utterance (a sequence of ~10 ms acoustic frames) into the best-scoring path
  through a finite-state syntactic/phonetic network. The score is a product (or
  sum of log) of per-frame state-match probabilities. Metrics: sentence and word
  accuracy; runtime measured as a multiple of real time (e.g. "N× real time").
  The exact reference is the full-breadth forward sweep over the whole network.
- **Single-machine / job-shop scheduling.** Build a schedule one job-placement
  at a time; objective such as total weighted tardiness. The yardstick is
  solution cost versus search effort, and against an exact solver on small
  instances.
- **0/1 knapsack.** Choose a subset of `n` weighted, valued items under a
  capacity. Objective: total value. For small `n` the exact optimum is available
  by enumerating all `2^n` subsets, giving an exact gap.

The protocol throughout: fix a time/effort budget, run the search, and report
the achieved objective and runtime against the exact reference where one exists.

## Code framework

The primitives already exist: a problem exposes an initial state, a way to
enumerate a state's children (the decisions available at the next layer), and an
evaluation function that scores a state. What is missing is the search
discipline itself — the rule for which states to explore. That is the empty
slot below.

```python
from dataclasses import dataclass

@dataclass
class State:
    idx: int        # how many decisions made (which layer)
    value: float    # objective accumulated so far
    # ... problem-specific fields (remaining capacity, partial schedule, ...)

def children(state, problem):
    """Enumerate the states reachable by one more decision. Known: this is the
    problem's transition relation."""
    ...

def evaluate(state, problem):
    """Score a (partial) state -- higher is more promising. A cheap local score
    and an expensive accurate score are both available."""
    ...

def exact_full_breadth(problem):
    """Reference: sweep the entire layer forward with no pruning. Correct but
    its cost is the full breadth at every layer."""
    layer = [problem.start]
    for _ in range(problem.depth):
        layer = [c for s in layer for c in children(s, problem)]
        # no pruning -- this is the breadth we cannot afford at scale
    ...

def search(problem):
    # TODO: the search discipline to design.
    pass
```
