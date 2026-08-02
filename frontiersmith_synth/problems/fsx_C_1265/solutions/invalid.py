# TIER: invalid
"""
Vias every column ("max cooling everywhere") while ignoring the area budget entirely. Every
generated instance deliberately sets the budget A far below the cost of via-ing all N columns,
so this always blows the budget -> Ratio: 0.0 on every test case.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    M = int(nx())
    N = int(nx())
    nx()  # A (intentionally ignored)
    nx(); nx()  # R0, Rv
    for _ in range(N):
        nx()  # area costs, ignored
    for _ in range(M * N):
        nx()  # power maps, ignored

    sys.stdout.write(" ".join("1" for _ in range(N)) + "\n")


if __name__ == "__main__":
    main()
