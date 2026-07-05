# TIER: trivial
# Consecutive block {0,1,...,n-1}: maximal sum-frequency collisions.
# |A+A| = 2n-1 = the checker's baseline B -> Ratio == 0.1 exactly.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); M = int(toks[1])
    A = list(range(n))          # fits since M = 9n >= n
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
