We are given a job shop: $n$ jobs, $m$ machines, each job a fixed route of operations where operation $i$ runs on a prescribed machine for an uninterruptible duration $d_i$, and a machine handles one operation at a time. We must choose, for every machine, the order in which it processes its operations so as to minimize the makespan — the time the last job finishes. Writing $t_i$ for the start time of operation $i$, the job-precedence constraints $t_j - t_i \ge d_i$ are rigid because the route is given; all the difficulty lives in the machine-capacity constraints, which for every pair $(i,j)$ sharing a machine demand $t_j - t_i \ge d_i$ *or* $t_i - t_j \ge d_j$. Those disjunctions — for each pair, one goes first, but which is our choice — are the entire combinatorial problem. The cleanest way to hold this is the disjunctive graph $G=(N,A,E)$: operations as nodes (plus a dummy source $0$ and sink $n$), fixed conjunctive arcs $A$ (a directed $i \to j$ of weight $d_i$ for each within-job successor), and one disjunctive clique $E_k$ per machine over the operations that share it. To schedule a machine is to *orient* its clique into a linear order, a selection $S_k$ that is valid exactly when acyclic; a complete acyclic selection $S = \bigcup_k S_k$ turns the graph into an ordinary DAG $D_S=(N, A\cup S)$. The load-bearing fact is that once everything is oriented, the earliest each operation can start is the longest path reaching it from the source, so the makespan is the longest path from $0$ to $n$ in $D_S$. Computing a longest path in a DAG is linear; all the hardness is in choosing the orientation, and orienting all the cliques jointly to minimize the longest path is NP-hard — a ten-by-ten instance stood unsolved for twenty years. So we will not find the optimum at realistic sizes. The standard fast answer is greedy priority dispatching — SPT, MWKR, FCFS — which makes sequencing decisions on every machine at once, interleaved in time, each myopic and permanent, with no sense of which machine actually drives the makespan; no single rule wins across instances, and randomizing for the best of many runs buys only modest, unreliable gains. We want to spend extra compute precisely where it pays.

I propose the Shifting Bottleneck Procedure. The guiding question is: if orienting all the cliques at once is hopeless, what is the smallest piece we *can* orient optimally? One machine. Freeze the rest of the world and consider a single machine $k$ in isolation. Its operations are not free-floating — each sits inside a job with upstream work that must precede it and downstream work that must follow. If we summarize "everything before operation $i$" as a release time (head) $r_i$ and "everything after $i$" as a tail $q_i$, then machine $k$ alone becomes a single-machine problem with heads and tails, $1|r_i,q_i|C_{\max}$: each operation cannot start before $r_i$, runs for $d_i$, and carries $q_i$ of downstream work after it. The heads and tails come straight from the current graph. Let $L(a,b)$ be the longest path length from $a$ to $b$. With some machines already sequenced (call that set $M_0$, their cliques fixed),
$$r_i = L(0,i), \qquad q_i = L(i,n) - d_i,$$
and the current makespan is $H = L(0,n)$. One forward longest-path pass from the source gives every head, one backward pass from the sink gives every tail — the entire rest of the shop condensed into two numbers per operation. The subtlety is what to do with the machines not yet sequenced: their cliques have no orientation, and respecting unoriented disjunctions would either be meaningless or drag us back to the full NP-hard problem, so for machine $k$'s subproblem we simply *delete* every still-unoriented clique except $k$'s own, keeping only $A$ and the fixed $S_p$ for $p \in M_0$. The resulting heads and tails are therefore optimistic — they ignore contention on the unsequenced machines — which makes the subproblem a relaxation, a fact worth remembering.

The decisive question is which machine to sequence next, and here is where we beat greedy. The classical handle is criticality — a machine is critical if one of its arcs lies on a longest path, and any improving schedule must reverse an arc on every critical path — but that is a yes/no flag; it does not say which critical machine is *more* of a bottleneck. We want a degree, not a flag, and the single-machine subproblem supplies it. Solving machine $k$ in isolation gives the smallest path length we can force through it, $B(k,M_0) = \max_i (C_i + q_i)$. Setting a due date $f_i = H - q_i$ turns this into maximum lateness, because
$$C_i - f_i = C_i + q_i - H,$$
so $1|r_i,q_i|C_{\max}$ is equivalent to $1|r_i|L_{\max}$, and we report $\ell(k,M_0) = B(k,M_0) - H$, the optimal $L_{\max}$. Once machine $k$ is inserted, paths not using it stay $\le H$ and paths through it are $\le B(k,M_0)$, so the new partial makespan is $\max(H, B(k,M_0))$ and the actual increase is $\max(0, \ell(k,M_0))$. At a fixed iteration $H$ is common to every candidate, so ranking by $\ell$ is ranking by $B$, and the bottleneck is
$$\text{bottleneck} = \arg\max_{k \in M\setminus M_0} \ell(k,M_0).$$
A machine with large optimal lateness is one whose *best possible* order still presses hardest against the current bound — so we sequence it first, while the graph is still loose and we have the most freedom to accommodate it, and let the easier machines fit around it. This is grounded, not arbitrary: with $M_0$ empty, every full schedule must respect each one-machine relaxation, so $\max_k B(k,\emptyset)$ (equivalently $H_0 + \max_k \ell(k,\emptyset)$ for initial job-precedence longest path $H_0$) is a genuine lower bound on the optimal makespan.

Because the ranking depends on getting $\ell(k,M_0)$ right, the single-machine subproblem must be solved *exactly*. It is $1|r_i|L_{\max}$, NP-hard in the strong sense but only at sizes far beyond one machine in a ten-job shop, and Carlier-style branch and bound dispatches hundreds of jobs in seconds: a Schrage rule (at the current clock, among released jobs dispatch the largest tail, ties by largest duration) gives an upper bound and a critical path; from the critical block $J$ after the critical job one reads the lower bound
$$h(J) = \min_{i\in J} r_i + \sum_{i\in J} d_i + \min_{i\in J} q_i,$$
treating $J$ as one all-or-nothing block; the critical job branches before-or-after $J$, and trees rarely exceed $2n$ nodes. A clean modern equivalent is a small CP-SAT model storing the due date $f_i = H - q_i$ directly, with one interval per operation released no earlier than its head, a no-overlap, and the objective $\min \max_i (\text{end}_i - f_i)$, which is exactly $L_{\max}$.

The crucial second half of the method is re-optimization. The naive loop — pick the bottleneck, fix it, move on — is only a slightly smarter one-pass greedy, and it leaves gains on the table for the same reason dispatching does: the *moment* a new machine $m$ is added to the graph, every other operation's heads and tails change, because the graph got longer and more constrained. The order chosen for a machine sequenced three steps ago was optimal against a graph that no longer exists. So after fixing each new bottleneck we go back and re-solve the already-sequenced machines against the now-current labels: pull a machine's arcs out, recompute everyone's heads and tails *without* it so it sees the up-to-date rest of the shop, re-solve its single-machine problem, and drop the new selection back in. Since each re-optimization changes the graph again, we sweep more than once — about three cycles for intermediate partial schedules, re-optimizing worst-first by decreasing subproblem value and re-sorting after each sweep. Strictly, only critical machines (those contributing an arc to a current critical path) can shorten the makespan; but a sly refinement escapes shallow local optima by temporarily removing about $\sqrt{|M_0|}$ of the *non*-critical machines and reinserting them one at a time under the updated labels, exploring arrangements a critical-only pass would never reach. Only at the very end, with all machines sequenced, do we cycle until a full sweep yields no improvement — that is where grinding to a local optimum pays. One last honesty: because we deleted unsequenced cliques and ignored precedence that earlier selections induced among $k$'s own jobs, the optimal single-machine order can occasionally create a cycle; the deterministic repair finds the offending precedence pair and re-solves that one subproblem with that pair enforced as an extra constraint. Finally, the inner longest-path computation is the hot spot, run for every candidate on every iteration. A sequenced machine's fixed order is a complete acyclic digraph — the transitive closure of its unique Hamiltonian path — so all but the $n-1$ consecutive arcs are redundant for longest paths; keeping only those, every node has at most a job-predecessor and a machine-predecessor, the labeling is $O(n)$ over per-job and per-machine adjacency lists, and the whole procedure stays efficient. The bottleneck keeps shifting from machine to machine as the graph fills in, which is the whole character of the thing.

```python
import numpy as np
from collections import namedtuple
from ortools.sat.python import cp_model

Task = namedtuple("Task", "p head due")


class DisjunctiveGraph:
    """Conjunctive (job) arcs fixed in column 0; disjunctive (machine) arcs in
    column 1, added/removed as machines are sequenced. Longest paths -> release
    labels, due-date labels, makespan, critical path (O(n) over adjacency lists)."""

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

    def _cycle_map(self):
        seen, active, parent = set(), set(), {}

        def dfs(n):
            seen.add(n); active.add(n)
            for nb in self.successors[n]:
                if nb < 0:
                    continue
                if nb not in seen:
                    parent[nb] = n
                    cycle = dfs(nb)
                    if cycle:
                        return cycle
                elif nb in active:
                    cycle, cur = {n: nb}, n
                    while cur != nb:
                        prev = parent[cur]
                        cycle[prev] = cur
                        cur = prev
                    return cycle
            active.remove(n)
            return None

        for n in range(self.num_nodes):
            if n not in seen:
                cycle = dfs(n)
                if cycle:
                    return cycle
        return {}

    def makespan(self, reverse=False):
        """Longest-path labeling. Forward labels include each operation duration;
        reverse labels give paths from an operation to the end. Returns
        (makespan, critical_path, distances); makespan is None on a cycle."""
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
            return None, self._cycle_map(), dist
        sink = int(dist.argmax()); ms = float(dist[sink])
        cp, n = [], sink
        while n >= 0:
            cp.append(n); n = pred[n]
        return ms, list(reversed(cp)), dist


def solve_lmax(tasks, precs=None, horizon=20000):
    """Exact single-machine 1|r|L_max. tasks[op] = Task(p, head, due)."""
    mdl = cp_model.CpModel()
    iv = {}
    for op, t in tasks.items():
        s = mdl.NewIntVar(t.head, horizon, f"s_{op}")
        e = mdl.NewIntVar(t.head + t.p, horizon, f"e_{op}")
        iv[op] = (s, e, t.due, mdl.NewIntervalVar(s, t.p, e, f"i_{op}"))
    mdl.AddNoOverlap([x[3] for x in iv.values()])
    if precs:
        for a, b in precs:
            mdl.Add(iv[b][0] < iv[a][0])
    lmax = mdl.NewIntVar(-horizon, horizon, "Lmax")
    mdl.AddMaxEquality(lmax, [e - due for (_, e, due, _) in iv.values()])
    mdl.Minimize(lmax)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.Solve(mdl)
    order = [op for _, op in sorted((solver.Value(iv[op][0]), op) for op in iv)]
    return solver.ObjectiveValue(), np.array(order)


def heads_and_due_dates(g, ops, costs):
    ms, _, long_to = g.makespan()
    _, _, long_from = g.makespan(reverse=True)
    return {op: Task(p=int(costs[op]),
                     head=int(long_to[op] - costs[op]),
                     due=int(ms - long_from[op] + costs[op])) for op in ops}


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
    precedence and re-solve."""
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
            tasks = heads_and_due_dates(g, machine_ops[m], costs)
            order, lmax, _ = insert_machine(g, tasks)
            sol[m] = order
            scores.append((lmax, m))
        scheduled[:] = [m for _, m in sorted(scores, reverse=True)]
    noncritical_reoptimize(g, sol, scheduled, machine_ops, costs)


def noncritical_reoptimize(g, sol, scheduled, machine_ops, costs):
    """Temporarily remove about sqrt(|M0|) non-critical machines and reinsert them."""
    ms, critical_path, _ = g.makespan()
    if ms is None:
        return
    critical_machines = {int(g.machines[op]) for op in critical_path}
    candidates = [m for m in reversed(scheduled) if m not in critical_machines]
    num_remove = min(int(np.sqrt(len(scheduled))), len(candidates))
    removed = candidates[:num_remove]
    for m in removed:
        for s, e in zip(sol[m][:-1], sol[m][1:]):
            g.remove_arc(s, e)
    for m in reversed(removed):
        tasks = heads_and_due_dates(g, machine_ops[m], costs)
        order, _, _ = insert_machine(g, tasks)
        sol[m] = order


def last_reoptimize(g, sol, scheduled, machine_ops, costs, max_iters=200):
    """Cycle until a full sweep yields no makespan improvement; keep the best."""
    best_ms = g.makespan()[0]
    best_sol = sol.copy()
    for _ in range(max_iters):
        improved = False
        for m in list(scheduled):
            for s, e in zip(sol[m][:-1], sol[m][1:]):
                g.remove_arc(s, e)
            tasks = heads_and_due_dates(g, machine_ops[m], costs)
            order, _, ms = insert_machine(g, tasks)
            sol[m] = order
            if ms < best_ms:
                best_ms = ms; best_sol = sol.copy(); improved = True
        if not improved:
            break
    return best_sol, best_ms


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
            tasks = heads_and_due_dates(g, machine_ops[k], costs)
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
