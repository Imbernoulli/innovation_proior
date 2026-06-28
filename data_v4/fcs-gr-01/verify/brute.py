#!/usr/bin/env python3
"""Brute-force oracle for Project Selection (Max-Weight Closure).

Enumerate every subset of projects. For a chosen set of projects, the set of
machines you MUST buy is the union of their prerequisites. The value of that
choice is (sum of chosen projects' profits) - (sum of required machines' costs).
Maximize over all 2^n project subsets (the empty set gives 0).

Reads the same stdin format as sol.cpp. O(2^n * (n + E)), only for tiny n.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0

    def nxt():
        nonlocal idx
        v = int(data[idx])
        idx += 1
        return v

    n = nxt()
    m = nxt()
    profit = [nxt() for _ in range(n)]
    cost = [nxt() for _ in range(m)]
    E = nxt()
    # req[i] = bitmask of machines that project i needs
    req = [0] * n
    for _ in range(E):
        i = nxt() - 1
        j = nxt() - 1
        req[i] |= (1 << j)

    best = 0  # empty selection
    for mask in range(1 << n):
        total = 0
        machines = 0
        for i in range(n):
            if mask & (1 << i):
                total += profit[i]
                machines |= req[i]
        # subtract cost of every required machine
        mm = machines
        while mm:
            low = mm & (-mm)
            j = low.bit_length() - 1
            total -= cost[j]
            mm ^= low
        if total > best:
            best = total

    print(best)


if __name__ == "__main__":
    main()
