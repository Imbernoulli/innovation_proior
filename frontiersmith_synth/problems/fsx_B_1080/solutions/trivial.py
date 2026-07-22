# TIER: trivial
"""Naive 'important symbols deserve the biggest slot' baseline: bind the
symbol with the highest final frequency to the LONGEST available codeword,
the next-highest to the next-longest, and so on -- the intuitively-plausible
but information-theoretically backwards idea that big/important things
should get more room. This is exactly the checker's internal baseline
construction (and, by the rearrangement inequality, the worst possible
binding for this fixed slot multiset)."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    f = [int(data[idx + i]) for i in range(n)]; idx += n
    L = [int(data[idx + i]) for i in range(n)]; idx += n

    order = sorted(range(n), key=lambda i: (-f[i], i))  # biggest freq first
    slots_desc = sorted(L, reverse=True)                 # longest slot first

    d = [0] * n
    for rank, sym in enumerate(order):
        d[sym] = slots_desc[rank]

    print(" ".join(str(x) for x in d))


if __name__ == "__main__":
    main()
