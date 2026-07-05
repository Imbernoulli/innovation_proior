# TIER: invalid
"""Emits a syntactically well-formed but WRONG split: a single all-zero term, which
reconstructs the zero tensor and fails the exact identity for any nonzero T -> 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    p = int(next(it)); q = int(next(it)); r = int(next(it))
    a = [0] * p; c = [0] * q; d = [0] * r
    sys.stdout.write("1\n" + " ".join(map(str, a + c + d)) + "\n")


if __name__ == "__main__":
    main()
