# TIER: greedy
"""The obvious fix over pure redundancy: notice each petal is actually a
LOOP (use the closing edge, don't backtrack -- period sum(L_p) instead of
2*sum(L_p)-2k), and send the two guards around the flower in OPPOSITE
petal orders so they are not doing the exact same thing at the exact same
time. This looks like real coordination (better geography, more efficient
routes) but never reasons about TIMING: both guards still start at the hub
at time 0 with no period padding and no phase search. Reversing a
length-k petal order leaves a middle petal's position essentially
unchanged relative to the other guard when petal lengths are symmetric --
the trap the strong solution has to see past."""
import sys


def build_blocks(Ls):
    offset = 1
    blocks = []
    for Lp in Ls:
        priv = list(range(offset, offset + Lp - 1))
        blocks.append(priv)
        offset += Lp - 1
    return blocks


def canonical_tour(blocks, order):
    tour = []
    for idx in order:
        tour.append(0)
        tour.extend(blocks[idx])
    return tour


def main():
    data = sys.stdin.read().split()
    k = int(data[0]); P = int(data[1])
    Ls = [int(data[2 + i]) for i in range(k)]

    blocks = build_blocks(Ls)
    order_fwd = list(range(k))
    order_rev = list(range(k - 1, -1, -1))

    w1 = canonical_tour(blocks, order_fwd)
    w2 = canonical_tour(blocks, order_rev)

    out = [str(len(w1)), " ".join(str(x) for x in w1),
           str(len(w2)), " ".join(str(x) for x in w2)]
    print("\n".join(out))


if __name__ == "__main__":
    main()
