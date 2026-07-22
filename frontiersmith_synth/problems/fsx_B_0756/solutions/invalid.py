# TIER: invalid
"""Deliberately infeasible: oversells past each night's printed ceiling and emits a
non-increasing ladder. Must score 0 under the checker's feasibility gate."""
import sys


def main():
    toks = sys.stdin.read().split()
    pos = 0
    n = int(toks[pos]); pos += 1
    ladder_lo = int(toks[pos])
    pos += 3
    nights = []
    for _ in range(n):
        capacity, fare, max_sell = int(toks[pos]), int(toks[pos + 1]), int(toks[pos + 2])
        pos += 3
        pos += 2 * max_sell
        nights.append((capacity, fare, max_sell))

    ladder = [ladder_lo + 5, ladder_lo + 5, ladder_lo + 3, ladder_lo + 2, ladder_lo + 1]  # not increasing
    sold = [m + 999 for (c, f, m) in nights]  # blows past max_sell every night

    out = [" ".join(map(str, ladder)), " ".join(map(str, sold))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
