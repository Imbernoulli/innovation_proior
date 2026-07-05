# TIER: invalid
# Emits a set with an obvious pairwise-sum collision (1+5 = 2+4 = 6), so it violates
# the conflict-free condition and must score 0.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    out = [v for v in [1, 2, 3, 4, 5, 6] if v <= n]
    sys.stdout.write(" ".join(map(str, out)) + "\n")


if __name__ == "__main__":
    main()
