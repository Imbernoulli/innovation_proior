# TIER: trivial
"""
Place no vias anywhere. This is exactly the checker's own reference baseline construction
(x = all zeros is always feasible, since it spends 0 of the area budget), so it reproduces B
exactly -> the calibrated ~0.1 trivial reference point on every test case.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    M = int(nx())
    N = int(nx())
    nx()  # A (unused)
    nx(); nx()  # R0, Rv (unused)
    for _ in range(N):
        nx()  # area costs, unused
    for _ in range(M * N):
        nx()  # power maps, unused

    sys.stdout.write(" ".join("0" for _ in range(N)) + "\n")


if __name__ == "__main__":
    main()
