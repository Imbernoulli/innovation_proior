# TIER: strong
"""Batcher odd-even merge-sort network (pad to a power of two, then drop every
comparator that only touches padding wires). Fewer comparators than the bitonic
network for these n, so it strictly beats greedy while still leaving large
headroom below the (unknown) optimum -> a third distinct score vector."""
import sys

NET = []


def oe_merge(lo, n, r):
    step = r * 2
    if step < n:
        oe_merge(lo, n, step)
        oe_merge(lo + r, n, step)
        for i in range(lo + r, lo + n - r, step):
            NET.append((i, i + r))
    else:
        NET.append((lo, lo + r))


def oe_sort(lo, n):
    if n > 1:
        m = n // 2
        oe_sort(lo, m)
        oe_sort(lo + m, m)
        oe_merge(lo, n, 1)


def main():
    n = int(sys.stdin.read().split()[0])
    m = 1
    while m < n:
        m *= 2
    oe_sort(0, m)
    # drop comparators touching padding wires (they hold +inf and never move data)
    net = [(i, j) for (i, j) in NET if i < n and j < n]
    sys.stdout.write("\n".join("%d %d" % (a, b) for (a, b) in net) + ("\n" if net else ""))


if __name__ == "__main__":
    main()
