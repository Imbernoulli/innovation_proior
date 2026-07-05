# TIER: greedy
"""Batcher bitonic sorting network, generalized to arbitrary n. Sub-quadratic
(O(n log^2 n)), so it clearly beats the bubble baseline, but uses more comparators
than the odd-even merge network -> a distinct, intermediate score vector."""
import sys

NET = []


def compare(seq, i, j, up):
    a, b = seq[i], seq[j]
    NET.append((a, b) if up else (b, a))       # min -> first token


def bmerge(seq, up):
    n = len(seq)
    if n <= 1:
        return
    m = 1
    while m * 2 < n:
        m *= 2
    for i in range(n - m):
        compare(seq, i, i + m, up)
    bmerge(seq[:m], up)
    bmerge(seq[m:], up)


def bsort(seq, up):
    n = len(seq)
    if n <= 1:
        return
    m = n // 2
    bsort(seq[:m], not up)
    bsort(seq[m:], up)
    bmerge(seq, up)


def main():
    n = int(sys.stdin.read().split()[0])
    bsort(list(range(n)), True)
    sys.stdout.write("\n".join("%d %d" % (a, b) for (a, b) in NET) + ("\n" if NET else ""))


if __name__ == "__main__":
    main()
