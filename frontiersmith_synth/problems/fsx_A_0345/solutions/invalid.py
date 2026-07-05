# TIER: invalid
# Emits an all-zero profile (sum <= 0), which the feasibility gate rejects.
# Must score exactly 0.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    sys.stdout.write(" ".join(["0"] * n) + "\n")


if __name__ == "__main__":
    main()
