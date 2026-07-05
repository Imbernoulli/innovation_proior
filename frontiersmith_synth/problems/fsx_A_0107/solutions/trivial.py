# TIER: trivial
# Reproduces the grader's reference: a Sidon (perfect-ruler) placement of n towers.
# All pairwise differences are (nearly) distinct, so |A-A| is large and the ratio
# sits well below 1 -> scores about 0.1.
import sys


def _is_prime(m):
    if m < 2:
        return False
    i = 2
    while i * i <= m:
        if m % i == 0:
            return False
        i += 1
    return True


def _next_prime(x):
    while not _is_prime(x):
        x += 1
    return x


def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    p = _next_prime(n)
    S = sorted({2 * p * k + (k * k) % p for k in range(p)})[:n]
    # all within M = 4 n^2 by construction
    print(" ".join(map(str, S)))


if __name__ == "__main__":
    main()
