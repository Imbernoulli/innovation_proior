# TIER: greedy
# Spread the coolant into a triangular profile: milder peak self-interference
# than a concentrated block, but still worse than a flat schedule. c1 ~ 2.6.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    f = [min(i + 1, n - i) for i in range(n)]
    sys.stdout.write(" ".join(map(str, f)) + "\n")


if __name__ == "__main__":
    main()
