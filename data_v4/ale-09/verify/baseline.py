#!/usr/bin/env python3
"""Trivial first-fit-by-arrival baseline solver (reads instance, writes assignment).

This is exactly the baseline that score.py recomputes for normalization, emitted as
an explicit solution file so the self-verification harness can score it and confirm
the heuristic solver strictly beats it.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); C = int(next(it))
    a = [0] * N; d = [0] * N; s = [0] * N
    for i in range(N):
        a[i] = int(next(it)); d[i] = int(next(it)); s[i] = int(next(it))
    order = sorted(range(N), key=lambda i: (a[i], d[i], s[i], i))
    bins = []  # list of (departure, size)
    assign = [0] * N
    for i in order:
        ai, di, si = a[i], d[i], s[i]
        placed = False
        for k in range(len(bins)):
            load = sum(ss for (dd, ss) in bins[k] if dd > ai)
            if load + si <= C:
                bins[k].append((di, si)); assign[i] = k; placed = True; break
        if not placed:
            assign[i] = len(bins); bins.append([(di, si)])
    sys.stdout.write("\n".join(str(x) for x in assign) + "\n")


if __name__ == "__main__":
    main()
