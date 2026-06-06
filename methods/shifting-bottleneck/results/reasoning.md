OK, let me think this through from scratch. I've got a job shop: `n` jobs, `m` machines, every job
is a fixed chain of operations — operation `i` runs on a prescribed machine for a fixed duration
`d_i`, no interruptions, and a machine does one operation at a time. I want to set the order in which
each machine processes its operations so the whole thing finishes as early as possible — minimize the
makespan. And I know going in that this is brutal: people can solve traveling-salesman problems with
hundreds of cities to optimality, but a ten-by-ten job shop has stood unsolved for twenty years. So
I'm not going to find the optimum at realistic sizes. I want a schedule that's reliably *good*, fast.

Let me get the structure straight before I do anything clever. Write the constraints with `t_i` the
start time of operation `i`:

    t_j - t_i >= d_i  for every job-precedence pair (i,j)        — this is forced
    t_i >= 0
    for every pair (i,j) on the same machine: t_j - t_i >= d_i  OR  t_i - t_j >= d_j

The job-precedence constraints are rigid — the route is given. The pain is entirely in that last
line: for every two operations sharing a machine, one has to go before the other, and *which* is my
decision. Those "or" constraints, the disjunctions, are the whole combinatorial problem.

The cleanest way I know to hold this is the disjunctive graph. Nodes are operations (plus a dummy
source `0` and sink `n`). Draw a directed *conjunctive* arc `i -> j`, weight `d_i`, whenever `j`
follows `i` in a job — those are fixed. For every pair on the same machine, draw an undirected
*disjunctive* edge. The disjunctive edges fall into cliques, one clique `E_k` per machine, because on
machine `k` every pair of its operations conflicts. To *schedule a machine* is to *orient* its
clique: pick a direction for each edge so the operations get a linear order. Call an orientation of
one machine's clique a selection `S_k`; it's valid (acyclic) exactly when it's a genuine sequence. A
complete selection `S = ∪_k S_k` orients every machine.

Once I've oriented everything, the graph `D_S = (N, A ∪ S)` is an ordinary DAG, and the earliest
operation `i` can start is just the longest path from the source to `i` — because every incoming arc is
a "you must wait this long" constraint, and the binding one is the longest. So the makespan is the
longest path from `0` to `n` in `D_S`. Scheduling is: orient all the machine cliques, acyclically, to
minimize the longest path. Computing a longest path in a DAG is trivial, linear. All the hardness is in
the orientation.

So the honest statement of my problem is: choose an acyclic complete selection `S` minimizing the
length of the longest path in `D_S`. And that's NP-hard — orienting all the cliques jointly is the
hard combinatorial object, no way around it at the full scale.

Fine. Greedy dispatching is the standard fast answer — SPT, MWKR, FCFS and friends. At each tick,
look at the operations that are eligible right now, pick one by some priority, commit it, never look
back; the eligibility set is rigged so you get an active schedule. It's fast and usually not
terrible. But stare at what it's doing: it's making sequencing decisions on *every machine at once*,
interleaved in time, each one myopic and permanent, with no sense of which machine is actually the
one driving the makespan. No single rule wins across instances; you randomize and take the best of
many runs for unreliable, modest gains. I want to do better, and I want to spend the extra compute
where it pays.

Let me ask: if orienting *all* the cliques at once is hopeless, what's the smallest piece I *can*
orient optimally? One machine. If I freeze the rest of the world and just decide the order on a
single machine `k`, what does that subproblem look like? Machine `k` has its set of operations, each
with a duration. But these operations aren't free-floating — each one sits inside a job, with stuff
that has to happen *before* it (earlier operations of the same job, on other machines) and stuff that
has to happen *after* it. If I could summarize "everything before operation `i`" as a release time
and "everything after `i`" as a tail, then machine `k` in isolation is just: schedule these
operations on one machine, each can't start before its release `r_i`, each runs `d_i`, and after it
finishes there's still `q_i` of downstream work before that job is done. That's a single-machine
problem with heads and tails — `1|r_i,q_i|C_max`.

Where do the heads and tails come from? From the graph itself, *given the orientations I've already
fixed*. Suppose some machines are already sequenced — call that set `M_0` — and I've oriented their
cliques. Then in the partially-oriented DAG, the longest path from the source `0` to operation `i` is
exactly the earliest `i` can possibly start: that's its head. And the longest path from `i` onward to
the sink, minus `i`'s own duration, is exactly how much must still happen after `i` — its tail. Let me
write `L(a,b)` for the longest path length from `a` to `b` in the current graph. Then

    r_i = L(0, i)              — head: earliest start, all upstream work
    q_i = L(i, n) - d_i        — tail: downstream work after i finishes

and the current makespan is `L(0,n)`. Two longest-path passes — one forward from the source giving
all the heads, one backward from the sink giving all the tails — hand me a head and a tail for every
operation. So the single-machine subproblem for machine `k` is completely determined by the current
state of the graph. Nice: the heads and tails are *exactly* how the rest of the shop bears on machine
`k`, condensed into two numbers per operation.

Wait — there's a subtlety about the machines that *aren't* sequenced yet, the ones in `M\M_0` other
than `k`. Their cliques aren't oriented. So when I compute longest paths, what do I do with their
disjunctive edges? I can't use them as precedence — they have no direction yet; using an unoriented
disjunction as a constraint would be meaningless, and if I tried to respect all of them I'd be back to
the full NP-hard problem. So I just *delete* them. For the purpose of machine `k`'s subproblem, I keep
only the conjunctive job arcs `A` and the already-fixed selections `S_p` for `p ∈ M_0`, and I drop
every still-unoriented clique except `k`'s own. That means the heads and tails are *optimistic* — they
ignore the contention on machines not yet sequenced, so they're a lower estimate of the real pressure.
That's a relaxation, and I'll want to remember that it's optimistic.

Now — which machine should I sequence first? This is the real question, and it's where I can be smart
instead of greedy. The classical handle is "criticality": a machine is critical for the current
selection if one of its arcs lies on a longest path. That's a clean, theoretically grounded notion —
Balas's result is that any schedule better than the current one must reverse an arc on every critical
path, so improvement *has* to touch a critical machine. But criticality is a yes/no label. It splits
the machines into critical and non-critical and stops there; it doesn't tell me which critical
machine is *more* of a bottleneck. I need a degree, not a flag.

What I'd really love to rank by is each machine's *marginal effect on the makespan* — how much does
committing this machine's order push the finish time up? But I can't cheaply compute that. So I need a
computable proxy. And I already built the right object: the single-machine subproblem. When I solve
machine `k` in isolation with its heads and tails, the optimal value tells me the smallest path length
I can force through that machine. On the subproblem, "finish time of job through `k`" is
`start + d + q`, head accounted for in `start >= r`. Minimizing the maximum of
`start_i + d_i + q_i` gives a tail-augmented one-machine makespan; call it `B(k, M_0)`. If the current
graph bound is `H = L(0,n)` and I set a due date `f_i = H - q_i`, then

    start_i + d_i - f_i = start_i + d_i + q_i - H

is the lateness of operation `i`. So minimizing the tail-augmented makespan `B(k, M_0)` is the same as
minimizing the maximum lateness `ell(k, M_0) = B(k, M_0) - H`. Once I insert the machine, paths that do
not use it still have length at most `H`, and paths through it have length at most `B(k, M_0)`, so the
new partial makespan is `max(H, B(k, M_0))`; the actual increase is `max(0, ell(k, M_0))`. At a fixed
iteration `H` is the same for every candidate machine, so ranking by `B` and ranking by `ell` are the
same. A machine with a large optimal lateness is one whose best possible order still presses hardest
against the current bound:

    ell(k, M_0) = optimal L_max of machine k's subproblem
    bottleneck m = argmax over unsequenced k of ell(k, M_0)

Now "which machine is the bottleneck" is no longer a yes/no property; it is a number I can compute and
maximize. And it's intuitive in scheduling terms: deal with the hardest, most-constraining machine
*first*, while the graph is still loose and I have the most freedom to accommodate it — then let the
easier machines fit around it. Priority to the bottleneck.

Let me also sanity-check that this is a sensible *first* yardstick. With `M_0` empty, `B(k, ∅)` is the
best tail-augmented makespan of machine `k` against just the job-precedence arcs. Every full schedule
has to respect every one of those one-machine relaxations, so `max_k B(k, ∅)` is a lower bound on the
whole problem's makespan. Equivalently, if the initial job-precedence longest path is `H_0`, then
`H_0 + max_k ell(k, ∅)` is that same lower bound. Good — the scoring criterion isn't arbitrary; it's
grounded in a relaxation.

Now I need to actually solve the single-machine subproblem to optimality, because the *ranking* of
machines depends on getting `ell(k, M_0)` right. The subproblem is `1|r_i,q_i|C_max`, equivalently
`1|r_i|L_max`. That's NP-hard in the strong sense — but only at sizes far beyond what one machine in a
ten-job shop produces, and there's good branch-and-bound for it that chews through hundreds or a
thousand jobs in seconds. Let me reconstruct the B&B so I know it's exact and fast.

Start with a heuristic schedule to get an upper bound and, crucially, a critical structure to branch
on. The Schrage-style rule: set the clock `t` to the earliest release; among jobs already released by
`t`, dispatch the one with the *largest tail* `q` (break ties by largest duration) — because a long
tail means downstream work I'd better not delay; schedule it at `t`; advance `t` to the max of "that
job's completion" and "the next release." Repeat. This gives a feasible schedule and its critical path.
Now read off the critical sequence `j(1), ..., j(p)` (if several critical paths, take the one with the
most arcs). Find the largest index `k` along it with `q_{j(k)} < q_{j(p)}` — that's the "critical job"
`j(k)` — and let `J = {j(k+1), ..., j(p)}` be the block after it. This block gives a lower bound: any
schedule has to wait for the earliest release in `J`, then run *all* of `J`, then suffer the smallest
tail in `J`, so

    h(J) = min_{i∈J} r_i + Σ_{i∈J} d_i + min_{i∈J} q_i

is a valid lower bound on the tail-augmented makespan of the one-machine problem; subtract `H` if I am
working in the `L_max` scale. Think of `J` as one fat all-or-nothing block: earliest you can start it,
total work, minimum unavoidable downstream tail. And the branch: in any optimal schedule the critical
job `j(k)` goes either entirely *before* every job of `J` or entirely *after* — those are the two
children. To force "before," give `j(k)` a tail large enough to push it ahead of `J`; to force "after,"
give it a head large enough to put it behind. Prune a node when `h(J)` is at least the incumbent on the
tail-augmented makespan scale. These trees stay tiny — rarely more than `2n` nodes — so the subproblem
really is cheap to solve exactly. (If I want only a lower bound at a node, there's an even cheaper one:
allow preemption, solve `1|r,pmtn|L_max` by preemptive earliest-due-date — preemption only helps, so its
`L_max` under-bounds the non-preemptive optimum.)

So the skeleton is taking shape: pick the bottleneck by max `ell(k, M_0)`, solve it to optimality, fix
its selection into the graph, repeat on the remaining machines, with heads and tails recomputed each
time from the growing graph. Let me write that loop and then poke at it for problems.

    M_0 = ∅
    while M_0 ≠ M:
        for each k in M\M_0:
            compute heads r_i, tails q_i from longest paths in current graph (deleting unsequenced cliques except k)
            ell(k, M_0) = optimal L_max of single-machine subproblem on k
        m = argmax_k ell(k, M_0)
        fix S_m (its optimal order) into the graph; M_0 ← M_0 ∪ {m}

There's a wall here, though, and I should hit it head-on. I sequence machines one at a time, and once
I fix a machine's order it's done. But the *moment* I add a new machine `m` to the graph, every other
operation's heads and tails change — the graph got longer and more constrained, so the relaxation I
used to sequence machine `m`'s predecessors was optimistic in a way that's now outdated. The order I
chose for some machine sequenced three steps ago was optimal *against a graph that no longer exists*.
It might be improvable now. If I never revisit, I'm just a slightly smarter one-pass greedy, and I'll
leave gains on the table exactly where dispatching does.

So after I fix each new bottleneck, I should go back and re-solve the machines I already sequenced,
against the now-current heads and tails. Re-optimizing machine `k` means: pull `S_k`'s arcs out of the
graph, recompute everybody's heads and tails *without* `k` (so `k` sees the up-to-date rest-of-shop),
re-solve `k`'s single-machine problem, and drop the new selection back in. Do this for the machines
already in `M_0`, one at a time. Because each re-optimization changes the graph again, I sweep more
than once — cycle through `M_0` a few times. For intermediate partial schedules I can cap this at about
three cycles; the order I re-optimize in can follow decreasing subproblem value so I touch the worst
machines first, and I re-sort after each full sweep. Only at the very end, when all machines are
sequenced, do I keep cycling until there's truly no improvement on a full pass — that's where it's
worth grinding to a local optimum.

Which machines are even worth re-optimizing? A machine whose selection contributes no arc to any
critical path isn't constraining the makespan right now — re-ordering it can't shorten the longest
path. So strictly I only *need* to re-optimize the critical machines. But there's a slyer refinement.
Take the machines that are currently *non-critical*, temporarily yank a few of them out of the graph,
recompute, and reinsert them one at a time. Pulling a non-critical machine out and slotting it back
under the updated heads/tails can free up a better arrangement that a pure critical-only pass would
never have explored — it perturbs the schedule out of a shallow local optimum. Remove about
`√|M_0|` of the non-critical machines, then reinsert them successively, so the local search has a way
to escape a schedule that is only stable under critical-machine moves.

One more wrinkle to be honest about. When I solve machine `k`'s subproblem, I deleted the unsequenced
cliques and ignored precedence relations that earlier selections induced among `k`'s own jobs. Once in
a while the optimal single-machine order, dropped into the graph, can create a *cycle* — an impossible
circular wait. I still need a deterministic repair. When a cycle shows up during a longest-path
computation, I find the offending precedence pair that caused it and re-solve that one subproblem
subject to that precedence as an extra constraint, which breaks the cycle without changing the rest of
the machinery.

Let me also make sure the inner loop is fast, because the longest-path computation runs for every
candidate machine on every iteration and again on every re-optimization — it's the hot spot. The graph
`D_T` I run longest paths on is dense: each sequenced machine contributes a *complete* subgraph on its
operations (every pair is comparable once the order is fixed — that's the transitive closure of the
order). So naively `α = O(n^2)` arcs and an `O(α)` labeling. But — an acyclic complete digraph *is* the
transitive closure of its unique Hamiltonian path. All the extra arcs are redundant for longest paths;
only the `n-1` consecutive arcs of the order actually matter. If I keep only those, every node has at
most one or two relevant predecessors (its job-predecessor and its machine-predecessor), and the
labeling becomes `O(n)`. In practice I don't even build that reduced graph explicitly: I keep a "job
list" (the operations in route order, per job) and a "machine list" (the operations in sequence order,
per sequenced machine). Every operation lives on exactly one job list and at most one machine list, and
its relevant neighbors are just its successors/predecessors on those two lists. Longest paths over that
structure are linear, and since longest-path is the most time-consuming part of the whole procedure,
that's what keeps it efficient.

The procedure has its shape now: makespan is a longest path once the machine cliques are oriented;
orienting them all at once is NP-hard, so I orient one machine at a time; one machine in isolation,
with the rest of the shop folded into heads and tails via longest paths, is a single-machine
max-lateness problem I can solve exactly; that subproblem's optimal `ell(k, M_0)` measures how hard
the machine presses against the current graph bound, so I sequence the worst — the bottleneck — first;
and because fixing a machine changes everyone else's heads and tails, I re-optimize the
already-sequenced machines in cycles after each insertion, and grind to a local optimum at the end. The
bottleneck keeps "shifting" from machine to machine as the graph fills in, which is the whole character
of the thing.

Now let me land it on real code. The graph carries conjunctive arcs (job routes) as a fixed first set
of successors/predecessors and disjunctive arcs (machine orders) as a second set I add and remove. A
forward longest-path pass gives release dates, a backward pass gives downstream tails, and a
topological longest path gives the makespan and the critical path.

```python
import numpy as np
from collections import namedtuple

Task = namedtuple("Task", "p head due")      # duration, release (head), due date f

class DisjunctiveGraph:
    """Conjunctive (job) arcs are fixed; disjunctive (machine) arcs are added /
    removed as machines get sequenced. Longest paths give releases, due dates, makespan."""
    def __init__(self, costs, machines, num_j, num_m):
        self.num_j, self.num_m = num_j, num_m
        self.num_nodes = num_j * num_m
        self.costs = costs.reshape(-1).astype(np.float32)      # d_i
        self.machines = machines.reshape(-1).astype(np.int32)  # machine of op i
        # column 0 = conjunctive (job) neighbor, column 1 = disjunctive (machine) neighbor
        self.successors   = -np.ones((self.num_nodes, 2), dtype=np.int64)
        self.predecessors = -np.ones((self.num_nodes, 2), dtype=np.int64)
        for job in range(num_j):                               # wire the job routes (fixed)
            for idx in range(num_m - 1):
                n = job * num_m + idx
                self.successors[n, 0]       = n + 1
                self.predecessors[n + 1, 0] = n

    def add_arc(self, s, e):      # orient one machine edge s -> e
        self.successors[s, 1] = e; self.predecessors[e, 1] = s
    def remove_arc(self, s, e):
        self.successors[s, 1] = -1; self.predecessors[e, 1] = -1

    def makespan(self, reverse=False):
        # forward labels include each op's duration; reverse labels give paths from op to the end
        # returns (makespan, critical_path, distances) — None makespan signals a cycle
        ...   # O(n) labeling over the job-list / machine-list structure
```

The single-machine subproblem `1|r_i,q_i|C_max == 1|r_i|L_max`. I can drop in Carlier's branch and
bound exactly as derived above, or — equivalently and more cleanly to code — a tiny
constraint-programming model. The model stores the due date `f_i = H - q_i`, not the delivery tail
itself: an interval per operation released no earlier than its head, no two overlapping, and minimize
the maximum of `end - due`, which is exactly `L_max`.

```python
from ortools.sat.python import cp_model

def solve_lmax(tasks, precs=None, horizon=20000):
    """Exact 1|r_i|L_max on one machine. tasks[op] = Task(p, head, due)."""
    mdl = cp_model.CpModel()
    iv = {}
    for op, t in tasks.items():
        s = mdl.NewIntVar(t.head, horizon, f"s_{op}")          # start >= head r_i
        e = mdl.NewIntVar(t.head + t.p, horizon, f"e_{op}")
        iv[op] = (s, e, t.due, mdl.NewIntervalVar(s, t.p, e, f"i_{op}"))
    mdl.AddNoOverlap([x[3] for x in iv.values()])              # one op at a time
    if precs:                                                  # cycle-avoidance constraints
        for a, b in precs:
            mdl.Add(iv[b][0] < iv[a][0])
    lmax = mdl.NewIntVar(-horizon, horizon, "Lmax")
    mdl.AddMaxEquality(lmax, [e - due for (_, e, due, _) in iv.values()])
    mdl.Minimize(lmax)                                         # minimize max lateness
    solver = cp_model.CpSolver(); solver.Solve(mdl)
    order = [op for _, op in sorted((solver.Value(iv[op][0]), op) for op in iv)]
    return solver.ObjectiveValue(), np.array(order)
```

Release and due-date labels for a machine's operations are read straight off the two longest-path
passes. The head is the earliest start (forward longest path to the op, minus its own duration). The
backward label gives the duration of the longest path from the op to the end, including the op; so the
code's due date is `H - q_i = H - (long_from_i - d_i)`:

```python
def heads_and_due_dates(g, ops, costs):
    ms, _, long_to   = g.makespan()              # forward longest paths
    _,  _, long_from = g.makespan(reverse=True)  # backward longest paths
    return {op: Task(p=int(costs[op]),
                     head=int(long_to[op]   - costs[op]),
                     due=int(ms - long_from[op] + costs[op])) for op in ops}
```

Solving one machine into the graph, re-solving if the new order created a cycle (add the offending
precedence and try again):

```python
def insert_machine(g, tasks):
    ms, precs = None, []
    while ms is None:
        lmax, order = solve_lmax(tasks, precs)
        for s, e in zip(order[:-1], order[1:]):
            g.add_arc(s, e)
        ms, cycle, _ = g.makespan()              # None -> cycle, find culprit precedence
        if ms is None:
            for s, e in zip(order[:-1], order[1:]):
                g.remove_arc(s, e)
            precs.append(_offending_pair(cycle, order))
    return order, lmax, ms
```

And the driver — the shifting bottleneck loop itself: each iteration scores every unsequenced machine
by its optimal `L_max` relative to the same current bound, fixes the worst (the bottleneck), then
re-optimizes the already-sequenced machines in cycles; a final local-reoptimization grinds to a local
optimum.

```python
def shifting_bottleneck(instance, max_reopt=3):
    num_j, num_m = instance["j"], instance["m"]
    machines = instance["machines"].reshape(-1)
    machine_ops = [np.argwhere(machines == m).reshape(-1) for m in range(num_m)]
    g = DisjunctiveGraph(instance["costs"], instance["machines"], num_j, num_m)
    costs = g.costs.astype(int)
    sol = -np.ones((num_m, num_j), dtype=np.int32)
    to_schedule, scheduled = set(range(num_m)), []

    for it in range(num_m):
        max_lmax, bm, bperm = -np.inf, None, None
        for k in to_schedule:                                  # score every unsequenced machine
            tasks = heads_and_due_dates(g, machine_ops[k], costs)
            order, lmax, _ = insert_machine(g, tasks)          # optimal single-machine L_max
            if lmax > max_lmax:                                # the bottleneck = largest L_max
                max_lmax, bm, bperm = lmax, k, order
            for s, e in zip(order[:-1], order[1:]):            # undo the trial insertion
                g.remove_arc(s, e)
        sol[bm] = bperm                                        # commit the bottleneck's order
        for s, e in zip(bperm[:-1], bperm[1:]):
            g.add_arc(s, e)
        if 0 < it < num_m - 1:                                 # re-optimize previously sequenced
            reoptimize(g, sol, scheduled, machine_ops, costs, max_reopt)
        to_schedule.remove(bm); scheduled.append(bm)
    return last_reoptimize(g, sol, scheduled, machine_ops, costs)

def reoptimize(g, sol, scheduled, machine_ops, costs, max_reopt):
    for _ in range(max_reopt):                                 # ~3 cycles, as the procedure prescribes
        scores = []
        for m in scheduled:
            for s, e in zip(sol[m][:-1], sol[m][1:]):          # pull machine m out
                g.remove_arc(s, e)
            tasks = heads_and_due_dates(g, machine_ops[m], costs)  # fresh labels without m
            order, lmax, _ = insert_machine(g, tasks)          # re-solve and drop back in
            sol[m] = order; scores.append((lmax, m))
        scheduled[:] = [m for _, m in sorted(scores, reverse=True)]   # worst-first next sweep
```

Makespan is a longest path in the oriented disjunctive graph; the orientation is NP-hard jointly, so I
take it one machine at a time. Each machine, with the rest of the shop compressed into heads and tails
by two longest-path passes, becomes an exact single-machine `1|r|L_max` problem; its optimal lateness is
a computable measure of how hard that machine presses against the current graph bound, so I sequence
the largest — the bottleneck — first and fix it into the graph. Fixing a machine changes every other
machine's heads and tails, so I re-optimize the already-sequenced machines in a few cycles after each
insertion, then grind to a local optimum at the end — and the bottleneck shifts from machine to machine
as the graph fills in.
