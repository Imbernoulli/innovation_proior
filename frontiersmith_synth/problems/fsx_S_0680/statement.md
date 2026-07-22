# Shifting Bottleneck: Sequence Every Machine in a Job Shop

A shop has **M machines** and **N jobs**. Each job is a fixed route of operations,
each operation bound to one machine and taking a fixed duration. Operations of a
job must run in route order (job precedence); operations assigned to the same
machine must run one at a time in whatever order you choose (no preemption).

Write a program that reads one instance and outputs, **for every machine, the
processing order of its operations**. The evaluator combines your per-machine
orders with the fixed job routes to build the full schedule and scores the
resulting **makespan** (finish time of the last operation).

## Input (one JSON object on stdin)

```
{"name": str,
 "n_jobs": J, "n_machines": M, "n_ops": K,
 "ops": [{"id": i, "job": j, "pos": p, "machine": m, "dur": d}, ...],  # K entries
 "job_ops": [[id, id, ...], ...],       # J lists: job j's operation ids in route order
 "machine_ops": [[id, id, ...], ...]}   # M lists: ids assigned to each machine (unordered)
```

`ops[i]["id"] == i`. `job_ops[j]` lists job `j`'s operations in the order they must
run. `machine_ops[m]` lists the operations that live on machine `m`, in no
particular order — you decide the order.

## Output (one JSON object on stdout)

```
{"machine_order": [[id, id, ...], ...]}   # M lists
```

`machine_order[m]` must be a **permutation** of `machine_ops[m]` (same multiset,
no missing/duplicate ids). Given `machine_order` plus the fixed `job_ops`, each
operation's start time is `max(finish of its job-predecessor, finish of its
machine-predecessor)` (0 if it has neither), and its finish is `start + dur`. If
the combined precedence graph has a cycle (your machine order contradicts the
job routes, so no valid start times exist), or `machine_order` has the wrong
shape/length/duplicate/foreign ids, the instance scores **0**. Otherwise the
**makespan** is the maximum finish time over all operations.

## Objective — MINIMIZE makespan

The evaluator computes two references directly from the instance: `q_base`, the
makespan when every machine simply orders its operations by ascending `id`
(equivalently: job index, then position — a naive fixed-priority schedule), and
`q_lb`, the classical lower bound `max(busiest machine's total load, longest
job's total duration)` — unreachable in general because it ignores all
cross-machine contention. For your (feasible) candidate makespan `q_cand`:

```
r = clamp( 0.1 + 0.75 * (q_base - q_cand) / max(q_base - q_lb, 1e-9) , 0, 1 )
```

The id-order schedule scores exactly `0.1`; every reduction below it raises `r`,
but `q_lb` can never be reached, so headroom stays open above any solution. The
final score is the mean of `r` over 10 fixed hidden instances.

## Why "busiest machine" is a trap

A machine's total workload is a poor proxy for how *critical* it is: a machine
can carry little total work yet still force the makespan up, if its few
operations sit deep inside long job chains with little slack on either side.
Conversely a heavily loaded machine whose operations have generous slack barely
constrains anything. Several instances are built exactly this way: a handful of
operations on a low-load machine are each squeezed between long, timing-critical
chains, while decoy machines soak up most of the total processing time without
being tight anywhere. A dispatch rule that reacts to processing time alone
(shortest-first, or "keep the busiest machine moving") has no notion of how much
slack each operation actually has on either side, and frequently sequences the
pivotal machine's operations in the wrong order — inflating makespan on exactly
these instances.

## What actually moves the score

For a fixed machine, given how much time must elapse before each of its
operations can start (its **head**, from everything upstream) and how much must
still happen after it finishes (its **tail**, from everything downstream), the
one-machine subproblem — order this machine's own operations to minimize the
worst head-to-tail span — has a well known priority answer: among operations
already available, always run the one with the largest tail next. Solving this
subproblem for *every* unsequenced machine identifies which one is truly
constraining (largest one-machine bound), fixing that machine's order first and
recomputing heads/tails for the rest. Fixing one machine changes the heads/tails
everyone else sees, so the identified bottleneck **shifts** — a machine that
looked calm earlier can become the tightest constraint later. Revisiting an
already-fixed machine against the updated graph, or swapping two adjacent
operations on the current longest chain through the schedule, can still shave
off idle time. Time limit 2–5s, memory ≤512MB, deterministic scoring only.
