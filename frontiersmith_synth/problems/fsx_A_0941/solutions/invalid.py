# TIER: invalid
# Emits a non-finite / malformed artifact -> checker must score this ~0.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    testId = int(next(it)); N = int(next(it)); Q = int(next(it))
    # deliberately wrong: emit garbage tokens instead of Q finite floats
    out = ["nan"] * Q
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
