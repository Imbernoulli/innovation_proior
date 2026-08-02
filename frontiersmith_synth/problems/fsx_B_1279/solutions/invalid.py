# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); F = int(next(it))

    # Deliberately infeasible: face values must sum to exactly F. Dump everything into
    # the last maturity but forget one unit, and pad with a stray non-integer token
    # style violation is unnecessary -- the sum mismatch alone must score 0.
    p = [0] * T
    if T > 0:
        p[-1] = max(0, F - 1)
    print(" ".join(map(str, p)))


main()
