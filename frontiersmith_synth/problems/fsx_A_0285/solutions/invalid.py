# TIER: invalid
# Deploys no relay teams at all: total strength zero -> infeasible -> score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    sys.stdout.write(" ".join(["0"] * n) + "\n")


if __name__ == "__main__":
    main()
