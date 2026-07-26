# TIER: trivial
"""Reproduces the checker's own internal baseline: book nights 1..B (ascending),
each night's FIRST-listed (common) pointing. Always feasible, always positive."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nxt():
        return next(it)

    N = int(nxt()); D = int(nxt()); K = int(nxt()); B = int(nxt()); T = int(nxt())
    night_first = {}
    for _ in range(T):
        sid = int(nxt()); night = int(nxt()); w = int(nxt()); ns = int(nxt())
        for _ in range(ns):
            nxt()
        if night not in night_first:
            night_first[night] = sid
    # families are irrelevant to the trivial construction, no need to parse further.

    nights = list(range(1, min(B, D) + 1))
    chosen = [night_first[d] for d in nights if d in night_first]
    print(len(chosen))
    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()
