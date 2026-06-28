#!/usr/bin/env python3
# Independent brute-force oracle for fcs-gr-03.
# Each task is assigned to exactly one worker. Enumerate ALL W^T assignments
# and directly compute the total cost = sum of base costs + per-worker convex
# overtime cost. The per-worker overtime cost for m tasks is the running sum of
# marginal surcharges base[i]*max(0, k-q[i]) for k=1..m.
import sys
from itertools import product


def overtime_cost(m, qi, basei):
    # total surcharge for assigning m tasks to a worker with quota qi, slope basei
    tot = 0
    for k in range(1, m + 1):
        tot += basei * max(0, k - qi)
    return tot


def solve(data):
    it = iter(data)
    W = next(it)
    T = next(it)
    c = [[next(it) for _ in range(T)] for _ in range(W)]
    q = [next(it) for _ in range(W)]
    base = [next(it) for _ in range(W)]

    best = None
    # assignment[j] = worker index for task j
    for assign in product(range(W), repeat=T):
        cnt = [0] * W
        total = 0
        for j in range(T):
            i = assign[j]
            total += c[i][j]
            cnt[i] += 1
        for i in range(W):
            total += overtime_cost(cnt[i], q[i], base[i])
        if best is None or total < best:
            best = total
    if best is None:
        best = 0  # T == 0
    return best


def main():
    data = list(map(int, sys.stdin.read().split()))
    print(solve(data))


if __name__ == "__main__":
    main()
