# TIER: invalid
"""Blatantly infeasible: blow straight through the weighted-degree cap of
node 0 by dumping the entire budget on its first incident cable."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    w_budget = int(next(it))
    for _ in range(m):
        next(it)
        next(it)
    for _ in range(n):
        next(it)
    w = [1] * m
    if m > 0:
        w[0] = 1 + w_budget * 1000 + 10 ** 6
    sys.stdout.write("\n".join(str(x) for x in w) + "\n")


if __name__ == "__main__":
    main()
