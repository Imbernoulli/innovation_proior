# TIER: trivial
"""Reproduce the checker's own uniform-split baseline: spread the season
budget evenly over every (field, week) slot, respecting both caps by
construction.  No look at rainfall / thresholds at all."""
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

    per_slot = 0
    if F * T > 0 and F > 0:
        per_slot = min(total_budget // (F * T), weekly_cap // F)
        per_slot = max(per_slot, 0)

    print(F, T)
    row = " ".join(str(per_slot) for _ in range(T))
    for _ in range(F):
        print(row)


if __name__ == "__main__":
    main()
