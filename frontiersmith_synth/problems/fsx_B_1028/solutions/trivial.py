# TIER: trivial
"""Reproduces the checker's own naive baseline: both guards patrol every
petal identically, treating each petal as a dead-end corridor (walk out to
the tip and back) instead of using the closing edge back to the hub. Pure
redundancy -- the second guard adds nothing -- and even the single-guard
route is inefficient (2*L_p - 2 steps per petal instead of L_p)."""
import sys


def build_blocks(Ls):
    offset = 1
    blocks = []
    for Lp in Ls:
        priv = list(range(offset, offset + Lp - 1))
        blocks.append(priv)
        offset += Lp - 1
    return blocks


def outback_tour(blocks, order):
    tour = []
    for idx in order:
        tour.append(0)
        b = blocks[idx]
        tour.extend(b)
        if len(b) >= 2:
            tour.extend(list(reversed(b))[1:])
    return tour


def main():
    data = sys.stdin.read().split()
    k = int(data[0]); P = int(data[1])
    Ls = [int(data[2 + i]) for i in range(k)]

    blocks = build_blocks(Ls)
    order = list(range(k))
    w = outback_tour(blocks, order)

    out = [str(len(w)), " ".join(str(x) for x in w),
           str(len(w)), " ".join(str(x) for x in w)]
    print("\n".join(out))


if __name__ == "__main__":
    main()
