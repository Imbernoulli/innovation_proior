# TIER: trivial
"""Reproduces the checker's own uniform-spread reference construction: divide
the budget as evenly as possible across all cells, giving any remainder to the
lowest-id cells. No physics, no targeting."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    k = int(toks[p]); p += 1
    cap = int(toks[p]); p += 1
    budget = int(toks[p]); p += 1
    # target cells are irrelevant to this baseline
    Nc = N * N
    base = budget // Nc
    rem = budget - base * Nc
    loads = [min(cap, base) for _ in range(Nc)]
    for i in range(rem):
        loads[i] = min(cap, loads[i] + 1)
    print(" ".join(map(str, loads)))


if __name__ == "__main__":
    main()
