# TIER: invalid
"""Invalid tier: emits n identical positions (all zero) -> not pairwise distinct,
so the checker's feasibility gate rejects it and it must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    sys.stdout.write(" ".join(["0"] * n) + "\n")


if __name__ == "__main__":
    main()
