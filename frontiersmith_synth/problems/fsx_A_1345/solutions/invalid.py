# TIER: invalid
# Deliberately garbage: claims every level is a Blocker-safe certificate
# using absurd weights that violate both the per-line floor and the sum
# bound, guaranteeing the checker rejects every single level.
import sys


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
    for j in range(1, K + 1):
        m_j = c[j - 1]
        # weight 1/999999 is far below the required floor 2^-size for any
        # line of size 2 or 3, and the "sum" also fails to matter since each
        # individual line already violates the floor condition.
        weights = ["1/999999"] * m_j
        out.append(f"{j} B " + " ".join(weights))
    print("\n".join(out))


if __name__ == "__main__":
    main()
