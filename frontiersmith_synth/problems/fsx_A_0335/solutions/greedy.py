# TIER: greedy
# Uniform drive: a flat profile. The self-convolution is a triangle peaking at the
# center; c1 = 2 exactly, which already beats the boundary-loaded baseline.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    print(" ".join(["1"] * n))


if __name__ == "__main__":
    main()
