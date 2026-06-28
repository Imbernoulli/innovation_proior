#!/usr/bin/env python3
"""Instance generator for "Resource-Constrained Project Scheduling (RCPSP)".

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format (see context.md "Input / output
contract"):

    n R
    cap_1 cap_2 ... cap_R
    then n lines, the i-th describing task i (1-indexed):
        dur  d_{i,1} d_{i,2} ... d_{i,R}  p  pred_1 ... pred_p

Meaning: a project of n tasks (activities) to be scheduled. Time is discrete and
starts at 0. There are R renewable resources; resource k has constant capacity
cap_k available at every time unit. Task i runs without interruption for dur_i
time units and, while running (over the half-open interval [s_i, s_i + dur_i)),
consumes d_{i,k} units of resource k. A task may start only after all its p
predecessors have finished (finish-to-start precedence). The objective is to
choose a start time s_i for every task so that no resource is ever oversubscribed
and all precedences hold, MINIMIZING the makespan max_i (s_i + dur_i). See
score.py / context.md for the exact rule and the feasibility -> 0 floor.

Instance regime (deterministic from the seed):
  * n tasks in [60, 120]; R resources in [2, 4].
  * The precedence graph is a random DAG built on a topological order: each task
    draws a few predecessors from earlier tasks, giving a realistic "network
    complexity" without cycles (acyclicity is guaranteed by only linking
    backwards in the order).
  * Durations in [1, 10]. Resource capacities are set to a modest multiple of the
    typical per-task demand, so resources are genuinely scarce (a task often
    cannot start the instant its predecessors finish) -- this is the regime where
    the order in which ready tasks are scheduled (the priority list) matters and a
    naive earliest-start list wastes time.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x2C00_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    n = rng.randint(60, 120)
    R = rng.randint(2, 4)

    # Per-task durations and resource demands.
    dur = [rng.randint(1, 10) for _ in range(n)]
    # Each task demands a random amount of each resource; a fraction of demands
    # are zero so that not every task contends for every resource.
    demand = [[0] * R for _ in range(n)]
    for i in range(n):
        for k in range(R):
            if rng.random() < 0.75:
                demand[i][k] = rng.randint(1, 6)
            else:
                demand[i][k] = 0

    # Precedence DAG on the topological order 0..n-1. Each task picks up to a few
    # predecessors among strictly-earlier tasks.
    preds = [[] for _ in range(n)]
    for i in range(n):
        if i == 0:
            continue
        max_pred = min(i, 3)
        # number of predecessors (could be 0 -> a fresh source)
        npred = rng.randint(0, max_pred)
        if npred > 0:
            chosen = rng.sample(range(i), npred)
            preds[i] = sorted(chosen)

    # Capacities: scale to the demand so resources are scarce but every single
    # task is individually schedulable (cap_k >= max single-task demand_k).
    caps = []
    for k in range(R):
        col = [demand[i][k] for i in range(n)]
        mx = max(col) if col else 1
        avg = (sum(col) / n) if n else 1.0
        # capacity around 2x..3x the average per-task demand, never below the max
        # single demand (otherwise some task could never run).
        cap = max(mx, int(round(avg * rng.uniform(2.0, 3.2))))
        cap = max(cap, 1)
        caps.append(cap)

    out = [f"{n} {R}"]
    out.append(" ".join(str(c) for c in caps))
    for i in range(n):
        row = [str(dur[i])]
        row.extend(str(demand[i][k]) for k in range(R))
        row.append(str(len(preds[i])))
        # predecessors are emitted as 1-indexed task ids
        row.extend(str(p + 1) for p in preds[i])
        out.append(" ".join(row))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
