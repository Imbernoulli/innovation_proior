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

I need a representation that holds both the rigid arcs and the free ones in one object. The
disjunctive graph does this. Nodes are operations (plus a dummy source `0` and sink `n`). Draw a
directed *conjunctive* arc `i -> j`, weight `d_i`, whenever `j` follows `i` in a job — those are
fixed. For every pair on the same machine, draw an undirected *disjunctive* edge. The disjunctive
edges fall into cliques, one clique `E_k` per machine, because on machine `k` every pair of its
operations conflicts. To *schedule a machine* is to *orient* its clique: pick a direction for each
edge so the operations get a linear order. Call an orientation of one machine's clique a selection
`S_k`; it's valid (acyclic) exactly when it's a genuine sequence. A complete selection `S = ∪_k S_k`
orients every machine.

Once I've oriented everything, the graph `D_S = (N, A ∪ S)` is an ordinary DAG, and the earliest
operation `i` can start is just the longest path from the source to `i` — because every incoming arc is
a "you must wait this long" constraint, and the binding one is the longest. So the makespan is the
longest path from `0` to `n` in `D_S`. Scheduling is: orient all the machine cliques, acyclically, to
minimize the longest path. Computing a longest path in a DAG is trivial, linear. All the hardness is in
the orientation.

So the honest statement of my problem is: choose an acyclic complete selection `S` minimizing the
length of the longest path in `D_S`. And that's NP-hard — orienting all the cliques jointly is the
hard combinatorial object, no way around it at the full scale.

I want a small instance in hand to check my reasoning against as I go, something I can solve by brute
force later to see whether the ideas actually do what I'll claim. Take 3 jobs on 3 machines, routes
and durations:

    Job 0:  M0(3) -> M1(2) -> M2(2)
    Job 1:  M0(2) -> M2(1) -> M1(4)
    Job 2:  M1(4) -> M0(3) -> M2(3)

Nine operations. I'll number them `op = 3*job + position`, so op0..op2 are job 0's, etc. With only
the job arcs in the graph and no machine oriented yet, the longest path from source to sink is just
the longest single job route: job 2 runs `4+3+3 = 10`. So the bare job-precedence makespan is
`H_0 = 10`. That's the loosest the graph ever is.

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
state of the graph. The heads and tails are, I think, *exactly* how the rest of the shop bears on
machine `k`, condensed into two numbers per operation — but let me actually compute them on the small
instance with no machine yet oriented, so I'm not just asserting it. Running the two longest-path
passes over the six job arcs:

    op0 M0 d3: r=0  q=4        op3 M0 d2: r=0  q=5        op6 M1 d4: r=0  q=6
    op1 M1 d2: r=3  q=2        op4 M2 d1: r=2  q=4        op7 M0 d3: r=4  q=3
    op2 M2 d2: r=5  q=0        op5 M1 d4: r=3  q=0        op8 M2 d3: r=7  q=0

I can spot-check one by hand: op7 (job 2's second operation) has `r=4` — it can't start until M1's
op6 finishes at time 4 — and `q=3`, the duration of op8 still downstream. Both match the route. So
the labels are right, and each operation now carries a release date and a leftover tail without my
having said anything about machine sequencing yet.

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
committing this machine's order push the finish time up? But I can't cheaply compute that exactly. So
I need a computable proxy. And I already built the right object: the single-machine subproblem. When I
solve machine `k` in isolation with its heads and tails, the optimal value tells me the smallest path
length I can force through that machine. On the subproblem, "finish time of job through `k`" is
`start + d + q`, head accounted for in `start >= r`. Minimizing the maximum of
`start_i + d_i + q_i` gives a tail-augmented one-machine makespan; call it `B(k, M_0)`. If the current
graph bound is `H = L(0,n)` and I set a due date `f_i = H - q_i`, then

    start_i + d_i - f_i = start_i + d_i + q_i - H

should be the lateness of operation `i`, which would make minimizing the tail-augmented makespan
`B(k, M_0)` the same as minimizing the maximum lateness `ell(k, M_0) = B(k, M_0) - H`. Before I lean
on that equivalence let me check it concretely. Take machine 0 in the small instance, ops {0,3,7},
with the heads and tails I just computed and `H = H_0 = 10`. Try the order op0, op3, op7: op0 runs
[0,3], op3 runs [3,5], op7 runs [5,8] — wait, op7 has `r=4`, which is already satisfied, but I should
respect it, and `max(5,4)=5`, so op7 runs [5,8], completion 8. Tail-augmented completions:

    op0: C=3, q=4 -> C+q = 7
    op3: C=5, q=5 -> C+q = 10
    op7: C=8, q=3 -> C+q = 11

so `B = 11` for this order. Now the lateness form, `f_i = H - q_i = 10 - q_i`:

    op0: f=6,  C-f = 3-6  = -3,   and C+q-H = 7-10  = -3   ✓
    op3: f=5,  C-f = 5-5  =  0,   and C+q-H = 10-10 =  0   ✓
    op7: f=7,  C-f = 8-7  =  1,   and C+q-H = 11-10 =  1   ✓

The two expressions agree op-by-op, and `L_max = max(-3,0,1) = 1 = B - H = 11 - 10`. Good — the
identity is exact, so I can solve the cleaner `1|r|L_max` and read `ell = L_max` off it. And is this
order optimal for machine 0? It's the only one that keeps op0 (the long-tail M0 op of job 0) early;
checking the other five permutations by the same arithmetic, none beats `B = 11`, so `B(0,∅) = 11`,
`ell(0,∅) = 1`.

Now, what happens to the makespan when I actually drop this order into the graph? My instinct is that
paths not using machine 0 stay at length `≤ H = 10`, paths through it reach length `≤ B = 11`, so the
new bound is `max(H, B) = 11`. Rather than trust that, I orient op0->op3->op7 in the graph and recompute
the longest path: it comes out to `11`. So inserting the bottleneck raised the makespan from 10 to 11,
exactly `max(0, ell) = 1` of increase, as predicted. The proxy is measuring a real thing.

A machine with a large optimal lateness is one whose best possible order still presses hardest against
the current bound:

    ell(k, M_0) = optimal L_max of machine k's subproblem
    bottleneck m = argmax over unsequenced k of ell(k, M_0)

So "which machine is the bottleneck" is no longer a yes/no property; it is a number I can compute and
maximize. Let me see what it picks on the example. Solving all three machines' subproblems against the
job-arc-only labels:

    machine 0  {0,3,7}:  B = 11,  ell = 1   (order 0,3,7)
    machine 1  {1,5,6}:  B = 10,  ell = 0   (order 6,1,5)
    machine 2  {2,4,8}:  B = 10,  ell = 0   (order 4,2,8)

Machine 0 is the unique bottleneck, `ell = 1`; machines 1 and 2 can be sequenced without pushing past
the current `H = 10` at all. That matches the scheduling intuition: deal with the hardest,
most-constraining machine *first*, while the graph is still loose and I have the most freedom to
accommodate it — then let the easier machines fit around it. Priority to the bottleneck.

There's a sanity check on the *scoring* hiding in those numbers too. With `M_0` empty, `B(k, ∅)` is
the best tail-augmented makespan of machine `k` against just the job-precedence arcs, and every full
schedule has to respect each of those one-machine relaxations — so `max_k B(k, ∅)` ought to be a lower
bound on the whole problem's optimal makespan. Here that is `max(11,10,10) = 11`. I can test whether
that's really a valid bound by brute-forcing this tiny instance: enumerate all `6·6·6 = 216` machine
orderings, discard the ones that create a cycle, take the longest path of each, and keep the minimum.
That comes out to an optimal makespan of `11`. So the bound `max_k B(k,∅) = 11` is not only valid but
tight on this instance — and equivalently `H_0 + max_k ell(k,∅) = 10 + 1 = 11`. The scoring criterion
isn't arbitrary; it's a genuine relaxation bound, and on this instance the bottleneck-first first move
already lands on the optimum.

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
order). So naively `α = O(n^2)` arcs and an `O(α)` labeling. But an acyclic complete digraph is the
transitive closure of its unique Hamiltonian path, and I suspect the extra arcs are redundant for
longest paths — only the `n-1` consecutive arcs of the order should matter. Let me check that rather
than assume it. Three operations in fixed order a,b,c with durations 3,2,5. Full closure has arcs
a->b, a->c, b->c; the Hamiltonian path keeps only a->b, b->c. Completion labels (longest path from
source, plus own duration):

    full closure:  a=3, b=5, c=10
    Hamiltonian:   a=3, b=5, c=10

Identical — the direct a->c arc never wins, because the path a->b->c is at least as long (it includes
b's duration on top). That's general: in a complete order any "skip" arc i->j is dominated by the
chain through the intervening operations, so dropping it changes no longest path. If I keep only the
consecutive arcs, every node has at most one or two relevant predecessors (its job-predecessor and its
machine-predecessor), and the labeling becomes `O(n)`. In practice I don't even build that reduced graph explicitly: I keep a "job
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

The exact Carlier `1|r_j|L_max` bottleneck ranking and re-optimization loop is the part I'd most
easily get wrong under time pressure; if I weren't confident I could implement it correctly in the
budget, I'd fall back to an active-schedule MWKR/SPT greedy dispatcher that I've already traced as
correct and ship that -- a plain correct submission beats an ambitious broken one.

Now let me land it on real code, as a single self-contained C++17 program that reads a job-shop
instance from stdin (the standard OR-Library format — `n m`, then one route of `machine duration`
pairs per job) and prints the makespan. The graph carries conjunctive arcs (job routes) as a fixed
first column of successors/predecessors and disjunctive arcs (machine orders) as a second column I add
and remove. A forward longest-path pass gives release dates, a backward pass gives downstream tails,
and a topological longest path gives the makespan and the critical path — all `O(n)` over the two
adjacency columns, with `long long` for the times.

```cpp
int n_jobs, n_mach, n_ops;
vector<ll> dur;                       // dur[op]
vector<int> mach;                     // machine of op
vector<array<int,2>> succ, pred;      // col 0 = job neighbor (fixed), col 1 = machine neighbor
inline void add_arc(int s,int e){ succ[s][1]=e; pred[e][1]=s; }      // orient one machine edge
inline void remove_arc(int s,int e){ succ[s][1]=-1; pred[e][1]=-1; }
// makespan(): forward labels include each op's duration; longest_path(false,...) gives paths from op
// to the end; both run in O(n) over the two adjacency columns, returning -1 / false on a cycle.
```

The single-machine subproblem `1|r_i,q_i|C_max == 1|r_i|L_max`. I drop in Carlier's branch and bound
exactly as derived above: Schrage gives the upper bound and the critical block, `h(J)` the lower
bound, and the critical job branches before/after the block by inflating its head or tail. Tasks carry
duration, head `r`, and tail `q`, and the subproblem returns the optimal `L_max = C_max_with_tails - H`
and the order; extra precedence pairs (for cycle repair) are enforced by inflating heads/tails.

```cpp
struct CTask { ll p, r, q; int id; };           // duration, head, tail
// schrage(T, seq): dispatch released jobs largest-tail-first -> upper bound + dispatch order.
// carlier_rec(T): from the Schrage critical block J, lower bound h(J)=min r + sum p + min q; prune
//   when h(J) >= incumbent; else branch the critical job AFTER J (bump its head) or BEFORE J (bump
//   its tail). Trees rarely exceed 2n nodes. carlier_best / carlier_seq hold the optimum.
struct LmaxResult { ll ell; vector<int> order; };
LmaxResult solve_lmax(const vector<int>& ops, const vector<ll>& head, const vector<ll>& tail,
                      ll H, const vector<pair<int,int>>& extraPrec);   // returns ell = L_max
```

Release and tail labels for a machine's operations are read straight off the two longest-path passes:
the head is the earliest start (forward longest path to the op, minus its own duration), and the tail
is the backward label minus the op's duration (the downstream work after it). `H` is the current
makespan, common to every candidate, so `ell = C_max_with_tails - H` ranks the machines.

```cpp
void heads_tails(vector<ll>& head, vector<ll>& tail, ll& H){
    vector<ll> fwd, bwd;
    longest_path(true, fwd);                     // forward longest paths
    longest_path(false, bwd);                    // backward longest paths
    H=0; for(int v=0;v<n_ops;++v) H=max(H,fwd[v]);
    for(int op=0;op<n_ops;++op){ head[op]=fwd[op]-dur[op]; tail[op]=bwd[op]-dur[op]; }
}
```

Solving one machine into the graph, re-solving if the new order created a cycle (find the offending
precedence and try again):

```cpp
InsertResult insert_machine(const vector<int>& ops){
    vector<pair<int,int>> precs;
    while(true){
        vector<ll> head,tail; ll H; heads_tails(head,tail,H);
        LmaxResult lr = solve_lmax(ops, head, tail, H, precs);
        for(size_t i=0;i+1<lr.order.size();++i) add_arc(lr.order[i], lr.order[i+1]);
        ll ms = makespan();
        if(ms>=0) return {lr.order, lr.ell, ms};                     // feasible: done
        auto cyc = succ_cycle();                                     // cycle -> find culprit
        for(size_t i=0;i+1<lr.order.size();++i) remove_arc(lr.order[i], lr.order[i+1]);
        precs.push_back(offending_pair(cyc, lr.order));
    }
}
```

And the driver — the shifting bottleneck loop itself: each iteration scores every unsequenced machine
by its optimal `L_max` relative to the same current bound, fixes the worst (the bottleneck), then
re-optimizes the already-sequenced machines in cycles; a final local-reoptimization grinds to a local
optimum.

```cpp
for(int it=0; it<n_mach; ++it){
    ll bestEll=LLONG_MIN; int bm=-1; vector<int> bperm;
    for(int k: to_schedule){                                         // score every unsequenced machine
        InsertResult r = insert_machine(machine_ops[k]);             // optimal single-machine L_max
        if(r.ell>bestEll){ bestEll=r.ell; bm=k; bperm=r.order; }     // bottleneck = largest L_max
        for(size_t i=0;i+1<r.order.size();++i) remove_arc(r.order[i], r.order[i+1]);  // undo trial
    }
    sol[bm]=bperm;                                                   // commit the bottleneck's order
    for(size_t i=0;i+1<bperm.size();++i) add_arc(bperm[i], bperm[i+1]);
    if(it>0 && it<n_mach-1)                                          // re-optimize previously sequenced
        for(int cyc=0; cyc<3; ++cyc){                                // ~3 worst-first cycles
            vector<pair<ll,int>> sc;
            for(int m: scheduled) sc.push_back({reopt_machine(m), m});  // pull out, re-solve, drop back
            sort(sc.rbegin(), sc.rend());                            // worst-first next sweep
            scheduled.clear(); for(auto& pr:sc) scheduled.push_back(pr.second);
        }
    to_schedule.erase(find(to_schedule.begin(),to_schedule.end(),bm)); scheduled.push_back(bm);
}
// then last_reoptimize: cycle until a full sweep yields no makespan improvement.
```
