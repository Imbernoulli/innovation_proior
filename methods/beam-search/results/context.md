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

The breadth grows quickly. If every node has branching factor `b`, layer `t`
holds up to `b^t` nodes; for a depth-`D` problem an exact frontier is `b^D` —
millions of phonetic paths for a single utterance, `2^n` subsets for an `n`-item
knapsack. The question is how to search such a graph fast enough to be practical,
with runtime and memory that grow linearly in the depth, returning a final
answer whose score is high and whose runtime is bounded — accepting that we may
not prove optimality.

## Background

The decision graph is searched by maintaining a *frontier* (the set of partial
solutions currently live) and repeatedly expanding frontier nodes into their
children. Several classical disciplines for choosing what to expand were on the
table.

**Exact and optimal disciplines.** The exact search disciplines below are
*complete* or *optimal*: they are guaranteed to find a solution, or the best one,
and they maintain a frontier whose size is exponential in the worst case. They
are the reference for correctness.

**Backtracking.** A search can commit to a path, hit a dead end, and unwind to
try another, retrying the abandoned branches.

**Dynamic programming.** When the partial-solution state is summarized by a small
label (e.g. a network state at a frame), distinct paths reaching the same label
can be merged and only the best one kept. Sweeping all labels forward, frame by
frame, finds the globally optimal path with no backtracking, in time proportional
to (#labels × depth). This is the Viterbi/forward sweep over a transition
network: exact and backtrack-free, with cost equal to the full label set updated
at every step.

**Heuristic evaluation, cheap and accurate.** Ranking a partial solution by how
promising it is requires an evaluation function. A cheap local score (the
immediate cost of the last decision) is fast; an accurate global score (an
estimate of the total cost of the best completion) orders nodes well but is
costly to compute. Both scoring functions are available off the shelf.

These pieces — a layered decision graph, a frontier, an evaluation function,
backtracking, and an exact full-breadth sweep — are the raw materials.

## Baselines

**Breadth-first search.** Expand the frontier in FIFO order, finishing all of
layer `t` before any of layer `t+1`. It is complete — if a solution exists it is
found, and the shallowest one first. The frontier is the entire layer, so memory
and time grow as `b^t`.

**Best-first search / A\*.** Order the frontier by an evaluation `f(n) = g(n) +
h(n)`, where `g(n)` is the cost accumulated to reach `n` and `h(n)` estimates
the remaining cost to a goal. When `h` is *admissible* — a lower bound on the
true remaining cost, `h(n) ≤ C*(n)` — A\* is complete and returns an optimal
solution; with a *consistent* `h` (`h(n) ≤ c(n,n') + h(n')`) it never needs to
reopen a node. A\* keeps every generated-but-unexpanded node on the frontier.

**Branch and bound.** Keep an incumbent (the best complete solution found so
far). For each subproblem compute a bound on the best score reachable inside it;
if that bound cannot beat the incumbent, prune the whole subtree. It is sound —
it never prunes the optimum — and it can cut large swaths of the graph when the
bounds are tight.

**A single greedy path (hill-climbing).** Keep exactly one partial solution; at
each layer extend it by its single best-looking child. Memory is `O(1)` per
layer and time is linear in depth.

These disciplines span exact full-breadth sweeps (BFS, A\*, branch-and-bound,
full-label DP) on one side and a single greedy path on the other.

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
    """Reference: sweep the entire layer forward with no pruning. Correct; its
    cost is the full breadth at every layer."""
    layer = [problem.start]
    for _ in range(problem.depth):
        layer = [c for s in layer for c in children(s, problem)]
        # no pruning
    ...

def search(problem):
    # TODO: the search discipline to design.
    pass
```
