#!/usr/bin/env python3
"""Instance generator for "Job-Shop Scheduling (makespan)" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the classic job-shop format:

    n m
    <job 0 row>
    <job 1 row>
    ...
    <job n-1 row>

where each job row has m space-separated (machine duration) PAIRS, i.e. 2*m
integers. The k-th pair (machine_k, dur_k) of job j is its k-th operation: it
must run on machine `machine_k` for `dur_k` time units, and operation k can only
start after operation k-1 of the same job has finished. Each machine processes at
most one operation at a time. We want to MINIMIZE the makespan (the time the last
operation finishes); see score.py / context.md for the exact rule.

Instance regime (deterministic from the seed):
  * n jobs and m machines, both in a moderate range; every job visits every
    machine exactly once (a random machine permutation per job -- the standard
    job-shop structure that makes the disjunctive constraints bite).
  * Durations are positive integers; a fraction of operations are made "heavy"
    (large duration) so that bottleneck machines emerge and ordering matters.
  * The square-ish n ~ m regime with mixed durations is exactly where a critical-
    path local search beats a dispatch rule by a wide margin.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x4A53_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # Size: square-ish, in a range that is hard but fits a 2s budget for a strong
    # local search and an O(n*m) decode.
    n = rng.randint(10, 20)   # jobs
    m = rng.randint(8, 15)    # machines (== operations per job)

    rows = []
    for _ in range(n):
        # Random machine order: every job visits every machine exactly once.
        order = list(range(m))
        rng.shuffle(order)
        durs = []
        for _k in range(m):
            base = rng.randint(10, 99)
            # Some operations are heavy to create bottlenecks.
            if rng.random() < 0.18:
                base = rng.randint(100, 200)
            durs.append(base)
        pairs = []
        for k in range(m):
            pairs.append(f"{order[k]} {durs[k]}")
        rows.append(" ".join(pairs))

    out = [f"{n} {m}"]
    out.extend(rows)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
