# TIER: trivial
# Reproduces the checker's own reference construction: everyone runs fully
# SERIALIZABLE. Always safe (no cycle can survive if nobody is ever
# "exposed"), but leaves all the throughput on the table.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    for _ in range(N):
        w = int(next(it))
        nr = int(next(it))
        for _ in range(nr):
            next(it)
        nw = int(next(it))
        for _ in range(nw):
            next(it)
    print(" ".join(["2"] * N))


if __name__ == "__main__":
    main()
