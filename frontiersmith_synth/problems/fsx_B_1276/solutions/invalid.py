# TIER: invalid
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    # skip the rest of the header/body -- we don't need it to emit garbage
    _ = [next(it) for _ in range(3 + 3 + 5 * N)]

    # infeasible: doc depth out of the allowed {0,1,2,3} range, and a duplicated index
    print(2)
    print(1, 9)     # doc depth 9 -- out of range
    print(1, 0)     # duplicate index
    print(5000)


main()
