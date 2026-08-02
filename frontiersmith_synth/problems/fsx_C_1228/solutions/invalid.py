# TIER: invalid
# Garbage output: an out-of-range isolation "level" (5 is not a real level
# in {0,1,2}) for every transaction. Must be rejected by the checker's
# range check on every case, unconditionally (not relying on any planted
# structure being present).
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    for _ in range(N):
        next(it)
        nr = int(next(it))
        for _ in range(nr):
            next(it)
        nw = int(next(it))
        for _ in range(nw):
            next(it)
    print(" ".join(["5"] * N))


if __name__ == "__main__":
    main()
