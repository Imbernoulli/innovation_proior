# The Shifting Bottleneck Procedure

## Problem

Job shop makespan minimization. `n` jobs, `m` machines; each job is a fixed route of operations,
operation `i` runs on a prescribed machine for an uninterruptible duration `d_i`; a machine processes
one operation at a time. Choose, for every machine, the order of its operations to minimize the
makespan (completion time of the last job).

On the **disjunctive graph** `G = (N, A, E)` — operations as nodes, conjunctive arcs `A` for job
precedence (arc `i -> j` of weight `d_i`), disjunctive cliques `E_k` for the operations sharing
machine `k` — scheduling means orienting every clique acyclically. Once oriented into a DAG `D_S`, the
**makespan equals the longest path from source to sink**. Orienting all cliques jointly to minimize
the longest path is NP-hard.

## Key idea

Decompose by sequencing machines **one at a time**, always next sequencing the machine that is the
worst **bottleneck**, and re-optimizing the already-sequenced machines after each insertion.

For a partially built schedule with `M_0` the set of already-sequenced machines, an unsequenced
machine `k` defines a **single-machine subproblem**: keep the job arcs `A` and the fixed selections
`S_p` (`p ∈ M_0`), delete the other unsequenced cliques, and read each operation's **head** and
**tail** from longest paths in the current graph:

    r_i = L(0, i)            (head: earliest start)
    q_i = L(i, n) - d_i      (tail: downstream work after i)

This is `1|r_i, q_i|C_max`, equivalent to minimizing maximum lateness `1|r_i|L_max` (with due date
`f_i = L(0,n) - q_i`). Its optimal value `v(k, M_0) = L_max` is exactly how much sequencing `k`
optimally lengthens the longest path, so it measures `k`'s "bottleneck quality" as a *degree*, not a
yes/no criticality flag.

## Algorithm

```
M_0 = ∅
while M_0 ≠ M:
    for each unsequenced machine k:
        compute heads/tails from longest paths (job arcs + S_p, p∈M_0; ignore other unsequenced cliques)
        v(k, M_0) = optimal L_max of k's single-machine 1|r|L_max subproblem
    bottleneck m = argmax_k v(k, M_0)
    fix m's optimal selection into the graph;  M_0 ← M_0 ∪ {m}
    reoptimize: for ~3 cycles, pull each already-sequenced machine out, recompute its heads/tails,
                re-solve it, drop it back (worst-first); plus a non-critical-machine perturbation
final local reoptimization: keep cycling until a full sweep yields no improvement
```

Notes:
- The single-machine subproblem is solved **exactly** (Carlier 1982 branch-and-bound: Schrage
  dispatch for an upper bound, critical-block lower bound `h(J) = min_J r + Σ_J d + min_J q`, branch on
  the critical job before/after the block; trees rarely exceed `2n` nodes). Exactness matters because
  the machine *ranking* depends on `v(k, M_0)`. A clean modern equivalent is a CP-SAT model.
- **Re-optimization** is essential: inserting a machine changes every other machine's heads and tails,
  so earlier greedy choices become improvable.
- Longest paths drive the inner loop; an acyclic complete order is the transitive closure of its
  Hamiltonian path, so only the consecutive arcs matter, giving an **O(n)** longest-path labeling over
  per-job and per-machine adjacency lists.
- Rare cycle from a fixed selection: re-solve that subproblem with the offending precedence enforced.
- `max v(k, ∅)` (first bottleneck value) is also a lower bound on the optimal makespan.

## Code

```python
import numpy as np
from collections import namedtuple
from ortools.sat.python import cp_model

Task = namedtuple("Task", "p head tail")


class DisjunctiveGraph:
    """Conjunctive (job) arcs fixed in column 0; disjunctive (machine) arcs in
    column 1, added/removed as machines are sequenced. Longest paths -> heads,
    tails, makespan, critical path (O(n) over job/machine adjacency lists)."""

    def __init__(self, costs, machines, num_j, num_m):
        self.num_j, self.num_m = num_j, num_m
        self.num_nodes = num_j * num_m
        self.costs = costs.reshape(-1).astype(np.float32)
        self.machines = machines.reshape(-1).astype(np.int32)
        self.successors = -np.ones((self.num_nodes, 2), dtype=np.int64)
        self.predecessors = -np.ones((self.num_nodes, 2), dtype=np.int64)
        for job in range(num_j):
            for idx in range(num_m - 1):
                n = job * num_m + idx
                self.successors[n, 0] = n + 1
                self.predecessors[n + 1, 0] = n

    def add_arc(self, s, e):
        self.successors[s, 1] = e
        self.predecessors[e, 1] = s

    def remove_arc(self, s, e):
        self.successors[s, 1] = -1
        self.predecessors[e, 1] = -1

    def makespan(self, reverse=False):
        """Longest-path labeling. Forward -> earliest starts (heads); reverse ->
        longest path from each node (tails). Returns (makespan, critical_path,
        distances); makespan is None if the orientation contains a cycle."""
        nbr = self.predecessors if reverse else self.successors
        rev = self.successors if reverse else self.predecessors
        dist = np.zeros(self.num_nodes, dtype=np.float64)
        indeg = np.array([(rev[n] >= 0).sum() for n in range(self.num_nodes)])
        pred = -np.ones(self.num_nodes, dtype=np.int64)
        stack = [n for n in range(self.num_nodes) if indeg[n] == 0]
        for n in stack:
            dist[n] = self.costs[n]
        seen = 0
        while stack:
            n = stack.pop(); seen += 1
            for nb in nbr[n]:
                if nb < 0:
                    continue
                if dist[n] + self.costs[nb] > dist[nb]:
                    dist[nb] = dist[n] + self.costs[nb]
                    pred[nb] = n
                indeg[nb] -= 1
                if indeg[nb] == 0:
                    stack.append(nb)
        if seen < self.num_nodes:
            return None, {}, dist                # cycle
        sink = int(dist.argmax()); ms = float(dist[sink])
        cp, n = [], sink
        while n >= 0:
            cp.append(n); n = pred[n]
        return ms, list(reversed(cp)), dist


def solve_lmax(tasks, precs=None, horizon=20000):
    """Exact single-machine 1|r,q|C_max == 1|r|L_max. tasks[op]=Task(p,head,tail)."""
    mdl = cp_model.CpModel()
    iv = {}
    for op, t in tasks.items():
        s = mdl.NewIntVar(t.head, horizon, f"s_{op}")
        e = mdl.NewIntVar(t.head + t.p, horizon, f"e_{op}")
        iv[op] = (s, e, t.tail, mdl.NewIntervalVar(s, t.p, e, f"i_{op}"))
    mdl.AddNoOverlap([x[3] for x in iv.values()])
    if precs:
        for a, b in precs:
            mdl.Add(iv[b][0] < iv[a][0])
    lmax = mdl.NewIntVar(-horizon, horizon, "Lmax")
    mdl.AddMaxEquality(lmax, [e - tail for (_, e, tail, _) in iv.values()])
    mdl.Minimize(lmax)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.Solve(mdl)
    order = [op for _, op in sorted((solver.Value(iv[op][0]), op) for op in iv)]
    return solver.ObjectiveValue(), np.array(order)


def heads_and_tails(g, ops, costs):
    ms, _, long_to = g.makespan()
    _, _, long_from = g.makespan(reverse=True)
    return {op: Task(p=int(costs[op]),
                     head=int(long_to[op] - costs[op]),
                     tail=int(ms - long_from[op] + costs[op])) for op in ops}


def _offending_pair(cycle, order):
    for s in order:
        if s in cycle:
            n = cycle[s]
            while n not in order:
                n = cycle[n]
            while cycle.get(n) in order:
                n = cycle[n]
            if s != n:
                return (s, n)
    raise RuntimeError("no offending precedence found")


def insert_machine(g, tasks):
    """Solve one machine; if its order creates a cycle, enforce the offending
    precedence and re-solve (rare)."""
    ms, precs = None, []
    while ms is None:
        lmax, order = solve_lmax(tasks, precs)
        for s, e in zip(order[:-1], order[1:]):
            g.add_arc(s, e)
        ms, cycle, _ = g.makespan()
        if ms is None:
            for s, e in zip(order[:-1], order[1:]):
                g.remove_arc(s, e)
            precs.append(_offending_pair(cycle, order))
    return order, lmax, ms


def reoptimize(g, sol, scheduled, machine_ops, costs, max_reopt=3):
    """Re-solve each already-sequenced machine for a few cycles (worst-first)."""
    for _ in range(max_reopt):
        scores = []
        for m in scheduled:
            for s, e in zip(sol[m][:-1], sol[m][1:]):
                g.remove_arc(s, e)
            tasks = heads_and_tails(g, machine_ops[m], costs)
            order, lmax, _ = insert_machine(g, tasks)
            sol[m] = order
            scores.append((lmax, m))
        scheduled[:] = [m for _, m in sorted(scores, reverse=True)]


def last_reoptimize(g, sol, scheduled, machine_ops, costs, max_iters=200):
    """Cycle until a full sweep yields no makespan improvement; keep the best."""
    best_ms = g.makespan()[0]
    for _ in range(max_iters):
        improved = False
        for m in list(scheduled):
            for s, e in zip(sol[m][:-1], sol[m][1:]):
                g.remove_arc(s, e)
            tasks = heads_and_tails(g, machine_ops[m], costs)
            order, _, ms = insert_machine(g, tasks)
            sol[m] = order
            if ms < best_ms:
                best_ms = ms; improved = True
        if not improved:
            break
    return sol, best_ms


def shifting_bottleneck(instance, max_reopt=3):
    """instance: {'j','m','costs'(j,m),'machines'(j,m)}. Returns (sol, makespan)."""
    num_j, num_m = instance["j"], instance["m"]
    machines = instance["machines"].reshape(-1)
    machine_ops = [np.argwhere(machines == m).reshape(-1) for m in range(num_m)]
    g = DisjunctiveGraph(instance["costs"], instance["machines"], num_j, num_m)
    costs = g.costs.astype(int)
    sol = -np.ones((num_m, num_j), dtype=np.int32)
    to_schedule, scheduled = set(range(num_m)), []

    for it in range(num_m):
        max_lmax, bm, bperm = -np.inf, None, None
        for k in to_schedule:                          # rank every unsequenced machine
            tasks = heads_and_tails(g, machine_ops[k], costs)
            order, lmax, _ = insert_machine(g, tasks)  # optimal single-machine L_max
            if lmax > max_lmax:                        # bottleneck = largest L_max
                max_lmax, bm, bperm = lmax, k, order
            for s, e in zip(order[:-1], order[1:]):    # undo trial insertion
                g.remove_arc(s, e)
        sol[bm] = bperm                                # commit the bottleneck
        for s, e in zip(bperm[:-1], bperm[1:]):
            g.add_arc(s, e)
        if 0 < it < num_m - 1:                         # re-optimize previously sequenced machines
            reoptimize(g, sol, scheduled, machine_ops, costs, max_reopt)
        to_schedule.remove(bm); scheduled.append(bm)

    return last_reoptimize(g, sol, scheduled, machine_ops, costs)
```

Evaluating the makespan of a fully oriented schedule with a CP/MIP model (interval per operation,
`end == start + duration`, job precedence `end <= next.start`, no-overlap per machine, minimize the
maximum end) is the standard exact check used to score any schedule.
