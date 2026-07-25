# TIER: greedy
# Textbook compressed sensing: a random Rademacher (+/-pmax) measurement matrix.
# Basis-agnostic; the standard recipe an experienced coder writes first.
import sys
import numpy as np


def main():
    data = sys.stdin.read().split()
    n, m, K, s, pmax = (int(data[i]) for i in range(5))
    rng = np.random.default_rng(1234567)
    P = rng.choice([-pmax, pmax], size=(m, n))
    sys.stdout.write("\n".join(" ".join(str(int(v)) for v in row) for row in P) + "\n")


main()
