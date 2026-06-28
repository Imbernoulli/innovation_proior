#!/usr/bin/env python3
# Independent oracle for the weighted-completion-time scheduling problem.
# Tries ALL n! orderings and returns the minimum total sum of w[i]*C[i],
# where C[i] is the completion time (prefix sum of processing times) of job i
# in the chosen order. Obviously correct; only viable for n <= ~9.
import sys
from itertools import permutations


def solve(data):
    it = iter(data)
    n = int(next(it))
    jobs = []
    for _ in range(n):
        t = int(next(it))
        w = int(next(it))
        jobs.append((t, w))

    if n == 0:
        return 0

    best = None
    for perm in permutations(range(n)):
        clock = 0
        cost = 0
        for i in perm:
            t, w = jobs[i]
            clock += t
            cost += w * clock
        if best is None or cost < best:
            best = cost
    return best


def main():
    data = sys.stdin.read().split()
    print(solve(data))


if __name__ == "__main__":
    main()
