# TIER: trivial
# The absolute minimum: certify only density level 1 (always trivially
# Blocker-safe by construction -- it is built from a couple of size-3 decoy
# lines) via the weight-floor rule w_i = 2^-size_i, and skip every other
# level. This is exactly the checker's own internal baseline construction.
import sys
from fractions import Fraction


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))
    pool = []
    for _ in range(P):
        sz = int(next(it))
        cells = [int(next(it)) for _ in range(sz)]
        pool.append(cells)
    c = [int(next(it)) for _ in range(K)]

    out = [str(K)]
    lvl1 = pool[:c[0]]
    weights = [f"1/{2 ** len(line)}" for line in lvl1]
    total = sum(Fraction(1, 2 ** len(line)) for line in lvl1)
    if total < Fraction(1, 2):
        out.append(f"1 B " + " ".join(weights))
    else:
        out.append("1 U")
    for j in range(2, K + 1):
        out.append(f"{j} U")
    print("\n".join(out))


if __name__ == "__main__":
    main()
