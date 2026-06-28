#!/usr/bin/env python3
"""Instance generator for "Factory Job-Shop Scheduling" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the classic rectangular job-shop format:

    n m
    (job 0) mach_0 proc_0  mach_1 proc_1  ...  mach_{m-1} proc_{m-1}
    (job 1) ...
    ...
    (job n-1) ...

where there are n JOBS and m MACHINES. Each job is a chain of EXACTLY m operations
that must run in the listed order; operation t of a job runs on machine `mach_t` and
takes `proc_t` time units. Each job visits every machine EXACTLY once (the machine
list of every job is a permutation of 0..m-1), so every machine processes exactly n
operations (one per job). A machine runs one operation at a time, no preemption.

n is chosen in [15, 30] and m in [10, 20] from the seed, which is the size band where
list-scheduling leaves substantial slack for critical-block local search to recover
yet the longest-path makespan is still cheap to recompute. Processing times are drawn
in [1, 99] (the standard Taillard band).
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x10B5_3030 ^ (seed * 2654435761 & 0xFFFFFFFF))

    n = rng.randint(15, 30)   # number of jobs
    m = rng.randint(10, 20)   # number of machines (= ops per job)

    out = [f"{n} {m}"]
    for _ in range(n):
        machines = list(range(m))
        rng.shuffle(machines)  # a random machine order (permutation of 0..m-1)
        toks = []
        for mc in machines:
            p = rng.randint(1, 99)
            toks.append(f"{mc} {p}")
        out.append(" ".join(toks))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
