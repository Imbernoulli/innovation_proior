# TIER: strong
"""Block-product construction from small optimal caps.

The Cartesian product of a cap set in F_3^a and a cap set in F_3^b is a cap set
in F_3^(a+b): three distinct product points are collinear only if each block
projection is collinear-or-constant, which (with cap blocks) forces the points
to coincide. Using the extremal dim-4 cap (size 20) and dim-3 cap (size 9) as
building blocks beats the {0,1}^n heuristic substantially.

Sizes: n=3->9, 4->20, 5->40, 6->80, 7->180, 8->400."""
import sys

CAP3 = [(0, 2, 0), (1, 0, 0), (1, 0, 2), (1, 1, 0), (1, 1, 2),
        (2, 0, 2), (2, 1, 2), (2, 2, 0), (2, 2, 1)]
CAP4 = [(0, 0, 0, 2), (0, 0, 2, 0), (0, 1, 0, 2), (0, 1, 2, 0), (0, 2, 0, 0),
        (0, 2, 2, 2), (1, 0, 0, 1), (1, 0, 1, 2), (1, 1, 0, 2), (1, 1, 1, 1),
        (1, 2, 0, 2), (1, 2, 1, 1), (2, 0, 1, 0), (2, 0, 2, 1), (2, 1, 0, 0),
        (2, 1, 0, 1), (2, 1, 1, 2), (2, 1, 2, 2), (2, 2, 1, 0), (2, 2, 2, 1)]
BLK = {
    1: [(0,), (1,)],
    2: [(0, 0), (0, 1), (1, 0), (1, 1)],
    3: CAP3,
    4: CAP4,
}


def blocks_for(n):
    b = [4] * (n // 4)
    r = n % 4
    if r > 0:
        b.append(r)
    return b


def build(n):
    S = [()]
    for dim in blocks_for(n):
        blk = BLK[dim]
        S = [a + p for a in S for p in blk]
    return S


def main():
    n = int(sys.stdin.readline().split()[0])
    S = build(n)
    sys.stdout.write("\n".join(" ".join(map(str, v)) for v in S) + "\n")


if __name__ == "__main__":
    main()
