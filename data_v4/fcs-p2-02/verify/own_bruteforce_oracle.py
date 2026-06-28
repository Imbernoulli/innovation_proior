#!/usr/bin/env python3
"""Exhaustive oracle for fcs-p2-02 from context.md.

The problem asks for the maximum total weight of pairwise non-overlapping
half-open intervals [s, e). Touching endpoints are compatible, so two intervals
overlap exactly when max(s1, s2) < min(e1, e2).

This oracle deliberately does not use the weighted-interval DP recurrence from
the submitted solution. It enumerates every subset and checks compatibility
directly, so it is only intended for small differential tests.
"""

import sys


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def solve(jobs):
    n = len(jobs)
    best = 0
    for mask in range(1 << n):
        total = 0
        ok = True
        for i in range(n):
            if not ((mask >> i) & 1):
                continue
            total += jobs[i][2]
            for j in range(i):
                if ((mask >> j) & 1) and overlaps(jobs[i], jobs[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok and total > best:
            best = total
    return best


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    jobs = []
    pos = 1
    for _ in range(n):
        s = int(data[pos])
        e = int(data[pos + 1])
        w = int(data[pos + 2])
        jobs.append((s, e, w))
        pos += 3
    print(solve(jobs))


if __name__ == "__main__":
    main()
