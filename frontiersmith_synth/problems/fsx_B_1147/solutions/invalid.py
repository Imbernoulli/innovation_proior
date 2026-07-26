# TIER: invalid
"""Deliberately infeasible: dumps the ENTIRE season budget into week 0 for
every field, blowing straight through the weekly delivery cap."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0

    def nxt():
        nonlocal idx
        v = data[idx]
        idx += 1
        return v

    T = int(nxt()); F = int(nxt()); K = int(nxt())
    total_budget = int(nxt()); weekly_cap = int(nxt())
    for _ in range(F):
        for _ in range(5):
            nxt()
    for _ in range(K):
        for _ in range(T):
            nxt()

    # blatantly over the weekly cap (and typically over budget too)
    blast = total_budget + weekly_cap + 1000

    print(F, T)
    for f in range(F):
        row = [0] * T
        row[0] = blast
        print(" ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
