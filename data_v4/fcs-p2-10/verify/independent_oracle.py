#!/usr/bin/env python3
import sys


def exhaustive_composition_best(n, p):
    if n == 0:
        return 0

    best = None

    def dfs(remaining, total):
        nonlocal best
        if remaining == 0:
            if best is None or total > best:
                best = total
            return
        for first in range(1, remaining + 1):
            dfs(remaining - first, total + p[first])

    dfs(n, 0)
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    p = [0] * (n + 1)
    for i in range(1, n + 1):
        p[i] = int(data[i])

    print(exhaustive_composition_best(n, p))


if __name__ == "__main__":
    main()
