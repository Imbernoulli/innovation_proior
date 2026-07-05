# TIER: greedy
"""Greedy: a dense, mildly clustered layout. Pack the n beacons pseudo-randomly into a
narrow window of width ~2n. This is much more sum-favourable than the Sidon baseline
(rho climbs toward ~1) but is not engineered for MSTD, so it stays below the strong
construction and varies per instance."""
import sys
import random


def main():
    data = sys.stdin.read().split()
    n, V = int(data[0]), int(data[1])
    W = min(V, 2 * n)               # narrow, dense window -> interval-like
    rng = random.Random(20240624)   # fixed seed -> deterministic
    pool = list(range(W + 1))
    rng.shuffle(pool)
    A = sorted(pool[:n])
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
