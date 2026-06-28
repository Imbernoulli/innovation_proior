#!/usr/bin/env python3
"""Trivial baseline for ale-06 "Production Line Scheduling".

Reads the instance on stdin and writes a feasible permutation on stdout using
the shortest-processing-time (SPT) dispatch rule: order jobs by ascending total
processing time across all machines (ties broken by job id). This is the simple
dispatch baseline the problem normalizes against; it is always feasible (it is a
permutation) but generally far from optimal for makespan.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    tot = []
    for j in range(n):
        s = 0
        for _ in range(m):
            s += int(next(it))
        tot.append((s, j))
    tot.sort()  # ascending total time, ties by job id
    perm = [j for (_, j) in tot]
    sys.stdout.write(" ".join(str(j) for j in perm) + "\n")


if __name__ == "__main__":
    main()
