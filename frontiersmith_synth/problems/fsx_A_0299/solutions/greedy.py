# TIER: greedy
"""Kronecker / additive-recurrence lattice: x_i^k = frac((i+1)*sqrt(prime_k)).
A cheap, moderate-quality low-discrepancy spread that beats the pseudo-random
baseline but is not optimal."""
import sys
import math

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]


def main():
    data = sys.stdin.read().split()
    d, M = int(data[0]), int(data[1])
    alpha = [math.sqrt(PRIMES[k]) for k in range(d)]
    out = []
    for i in range(M):
        coords = []
        for k in range(d):
            v = ((i + 1) * alpha[k]) % 1.0
            coords.append("%.10f" % v)
        out.append(" ".join(coords))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
