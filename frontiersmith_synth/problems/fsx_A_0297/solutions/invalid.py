# TIER: invalid
# Emits n copies of the same weight (all duplicates) -> fails the distinctness check -> scores 0.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    sys.stdout.write(" ".join(["0"] * n) + "\n")


if __name__ == "__main__":
    main()
