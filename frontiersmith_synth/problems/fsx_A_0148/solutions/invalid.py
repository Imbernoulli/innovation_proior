# TIER: invalid
"""Emits a single bogus rank-one act that does not reconstruct T -> score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    a, b, c = int(toks[0]), int(toks[1]), int(toks[2])
    row = [1] * (a + b + c)
    sys.stdout.write("1\n" + " ".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
