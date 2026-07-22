# TIER: trivial
"""Uniform-diameter baseline: give EVERY pipe the same diameter, the largest one whose
network-wide uniform cost still fits the budget. Ignores flow entirely -- reproduces
the checker's own reference construction (score ~0.1 by design)."""
import sys


def main():
    data = sys.stdin.read().split('\n')
    data = [t for t in data if t.strip() != ""]
    n = int(data[0].split()[0])
    line2 = data[1].split()
    ndiam = int(line2[0])
    diams = [int(x) for x in line2[1:1 + ndiam]]
    C = int(data[2].split()[0])
    length = [0] * n
    unit_cost = [0] * n
    for i in range(1, n):
        row = data[2 + i].split()
        length[i] = int(row[2])
        unit_cost[i] = int(row[3])

    def uniform_cost(D):
        return sum(unit_cost[v] * length[v] * D * D for v in range(1, n))

    best = 0
    for i, D in enumerate(diams):
        if uniform_cost(D) <= C:
            best = i

    print(" ".join(str(best) for _ in range(1, n)))


if __name__ == "__main__":
    main()
