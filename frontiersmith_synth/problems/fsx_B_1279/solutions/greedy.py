# TIER: greedy
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); F = int(next(it))
    L = [int(next(it)) for _ in range(T)]
    # (yields / capacities / haircut are read but not used -- the "obvious" idea)
    for _ in range(T):
        next(it)  # y
    for _ in range(T):
        next(it)  # Base
    next(it); next(it)  # haircut
    next(it); next(it)  # S, window_len

    # Exact cashflow matching: place each maturity right where the legacy liability
    # falls due. Looks unambiguously "correct" -- zero prefunding slack needed -- but
    # it reproduces the legacy schedule's own clustering verbatim.
    p = L[:]
    print(" ".join(map(str, p)))


main()
