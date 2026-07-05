# TIER: greedy
# Symmetric idea: place the towers as a contiguous arithmetic run 0,1,...,n-1.
# Then |A+A| = |A-A| = 2n-1 exactly, so the ratio is exactly 1 -- a clear step up
# from the Sidon baseline (whose ratio is ~0.5), but it never exceeds 1.
import sys


def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    A = list(range(n))  # exactly n distinct posts, all <= M
    print(" ".join(map(str, A)))


if __name__ == "__main__":
    main()
