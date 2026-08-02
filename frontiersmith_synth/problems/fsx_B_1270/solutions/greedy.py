# TIER: greedy
#!/usr/bin/env python3
"""Obvious recipe: buy the cheapest rate-on-line (premium / width) layers first,
at 100% participation, until the premium budget runs out (with a partial
placement on whichever layer the budget cuts off mid-purchase). This maximizes
nominal coverage-width bought per premium dollar spent -- a completely
reasonable-sounding shopping rule. It never asks whether the layers it ends up
buying form a CONTIGUOUS band starting at the retention: on this catalog the
cheapest rate-on-line layers sit at the TOP of the tower, so this recipe
reliably leaves a hole in the cheapest-looking place to skip -- right above
retention, where losses are most certain to land first."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    catalog = []
    for _ in range(M):
        A = int(next(it)); W = int(next(it)); Prem = int(next(it))
        K = int(next(it)); RP = int(next(it))
        catalog.append((A, W, Prem, K, RP))
    C0 = int(next(it))
    Pmax = int(next(it))
    S = int(next(it))
    for _ in range(S):
        n = int(next(it))
        for _ in range(n):
            next(it)

    order = sorted(range(M), key=lambda j: catalog[j][2] / catalog[j][1])

    # all-integer bookkeeping: spent100 = premium spent so far, scaled by 100
    shares = [0] * M
    spent100 = 0
    for j in order:
        Prem = catalog[j][2]
        if spent100 + 100 * Prem <= 100 * Pmax:
            shares[j] = 100
            spent100 += 100 * Prem
        else:
            leftover100 = 100 * Pmax - spent100
            if leftover100 > 0 and Prem > 0:
                s = min(100, leftover100 // Prem)
                shares[j] = s
                spent100 += s * Prem

    print(" ".join(map(str, shares)))


if __name__ == "__main__":
    main()
