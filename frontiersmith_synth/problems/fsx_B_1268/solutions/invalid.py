# TIER: invalid
# Emits a price far outside the rate-change-cap band for every tier -- infeasible.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    K = int(next(it))
    prices = []
    for _ in range(K):
        p0 = int(next(it)); c = int(next(it)); cap = int(next(it)); B = int(next(it))
        for _ in range(B):
            n = int(next(it)); m = int(next(it))
            for _ in range(m):
                next(it); next(it)
            next(it); next(it)
        prices.append(p0 * 50 + 100000)
    sys.stdout.write("\n".join(str(p) for p in prices) + "\n")


if __name__ == "__main__":
    main()
