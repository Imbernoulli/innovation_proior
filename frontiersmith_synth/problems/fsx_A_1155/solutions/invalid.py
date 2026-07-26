# TIER: invalid
"""Deliberately infeasible: endowments do not sum to the required scrip supply S, and
a tax rate is out of range. Must score Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0
    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1; return v
    N = int(nxt()); R = int(nxt())
    ALPHA_NUM = int(nxt()); ALPHA_DEN = int(nxt()); S = int(nxt()); TAX_DEN = int(nxt())
    for _ in range(R):
        for _ in range(N):
            nxt()

    # violate budget conservation: dump everything on nurse 0, ignore S
    E = [S * 5] + [0] * (N - 1)
    T = [TAX_DEN * 9] * R  # out of [0, TAX_DEN] range
    W = [[0] * N for _ in range(R)]

    out = []
    out.append(" ".join(map(str, E)))
    out.append(" ".join(map(str, T)))
    for r in range(R):
        out.append(" ".join(map(str, W[r])))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
