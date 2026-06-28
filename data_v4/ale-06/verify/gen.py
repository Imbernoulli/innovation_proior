#!/usr/bin/env python3
"""Instance generator for ale-06 "Production Line Scheduling"
(permutation flow-shop, makespan minimization).

Usage:  python3 gen.py SEED  > instance.txt

Instance format (stdout):
    line 1:  n m            (n jobs, m machines)
    next n lines: m integers   p[j][0] p[j][1] ... p[j][m-1]
                               (processing time of job j on machine k,
                                in machine order 0..m-1)

Semantics: a permutation flow shop. There are m machines in a fixed line
0,1,...,m-1. Every job must pass through ALL machines in that order. We choose
ONE permutation of the n jobs; that permutation is used on every machine (the
"permutation flow shop" restriction: no job may overtake another). A machine
processes one job at a time and never idles in the middle of a job once it
starts it. We minimize the makespan = completion time of the last job on the
last machine.

This is the classic Fm|prmu|Cmax problem (Taillard benchmarks). It is strongly
NP-hard for m >= 3. Processing times are drawn uniformly in [1, 99], the regime
used by Taillard's standard generator, so no machine trivially dominates.
Everything is a deterministic function of SEED.
"""
import sys
import random

PMIN = 1
PMAX = 99


def gen(seed: int):
    rng = random.Random(seed * 2654435761 + 12345)

    # Sizes chosen so a single accelerated NEH insertion pass is O(n*m) and the
    # iterated-greedy loop runs thousands of reconstructions inside the budget,
    # while the search space n! is astronomically large (no exact answer).
    n = rng.randint(40, 80)    # jobs
    m = rng.randint(5, 20)     # machines

    p = [[rng.randint(PMIN, PMAX) for _ in range(m)] for _ in range(n)]
    return n, m, p


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py SEED\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    n, m, p = gen(seed)
    out = [f"{n} {m}"]
    for row in p:
        out.append(" ".join(str(v) for v in row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
