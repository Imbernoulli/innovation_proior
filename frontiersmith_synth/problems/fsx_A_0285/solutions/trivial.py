# TIER: trivial
# Reproduces the checker's reference baseline: a block of ceil(n/2) unit relay
# stations on the lower half of the trail. Scores ~0.1 by construction.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    b = (n + 1) // 2
    f = [1] * b + [0] * (n - b)
    sys.stdout.write(" ".join(map(str, f)) + "\n")


if __name__ == "__main__":
    main()
