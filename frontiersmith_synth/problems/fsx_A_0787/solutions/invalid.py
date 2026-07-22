# TIER: invalid
# Emits an out-of-range / malformed artifact -> checker must score 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    K = int(toks[0]); L = int(toks[1])
    m = min(5, L)
    print(m)
    # pitch class 99 is out of the valid [0,11] range -> feasibility gate fails
    print(" ".join(["99"] * m))


if __name__ == "__main__":
    main()
