# TIER: trivial
# Predict 0 for every query. This exactly reproduces the checker's own internal
# baseline B, so it lands at Ratio ~= 0.1 by construction.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    testId = int(next(it)); d = int(next(it)); N = int(next(it)); Q = int(next(it))
    for _ in range(N):
        next(it); next(it); next(it)
    for _ in range(Q):
        next(it); next(it)
    out = ["0"] * Q
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
