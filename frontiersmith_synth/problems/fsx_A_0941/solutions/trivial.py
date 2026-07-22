# TIER: trivial
# Predict nm=0 (i.e. "just barely on the boundary") for every query. This exactly
# reproduces the checker's own internal baseline B, so it lands at Ratio ~= 0.1
# by construction.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    testId = int(next(it)); N = int(next(it)); Q = int(next(it))
    for _ in range(N):
        for _ in range(8):
            next(it)
    for _ in range(Q):
        for _ in range(7):
            next(it)
    out = ["0.0"] * Q
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
