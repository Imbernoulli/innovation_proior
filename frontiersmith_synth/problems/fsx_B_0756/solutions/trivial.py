# TIER: trivial
"""Sell only a small, always-safe fraction of each night's berths; never overbook.
This reproduces the checker's own weak baseline construction almost exactly."""
import sys


def main():
    toks = sys.stdin.read().split()
    pos = 0
    n = int(toks[pos]); pos += 1
    ladder_lo, ladder_hi, penalty = int(toks[pos]), int(toks[pos + 1]), int(toks[pos + 2])
    pos += 3
    nights = []
    for _ in range(n):
        capacity, fare, max_sell = int(toks[pos]), int(toks[pos + 1]), int(toks[pos + 2])
        pos += 3
        pos += 2 * max_sell  # skip passenger records, unused
        nights.append((capacity, fare, max_sell))

    ladder = [ladder_lo, ladder_lo + 1, ladder_lo + 2, ladder_lo + 3, ladder_lo + 4]
    sold = [max(1, round(c * 0.15)) for (c, f, m) in nights]

    out = [" ".join(map(str, ladder)), " ".join(map(str, sold))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
