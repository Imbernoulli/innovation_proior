# TIER: trivial
# The first k multiples of D: A = {0, D, 2D, ..., (k-1)D} mod M.
# Looks like a reasonable "spread out" attempt (an arithmetic progression with step D), but
# every element is congruent to 0 mod D -- it is maximally CONCENTRATED in a single coarse
# band, which is exactly the failure mode the objective punishes hardest (E_D blows up to
# roughly k^4, its worst possible value). Also mediocre on the fine term (a plain low-order
# subgroup-aligned progression). Reproduces the checker's low baseline region.
import sys


def main():
    toks = sys.stdin.read().split()
    M, D, k, W = int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3])
    A = [(i * D) % M for i in range(k)]
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
