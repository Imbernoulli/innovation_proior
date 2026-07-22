# TIER: invalid
"""Deliberately infeasible: emit a temperature level strictly above Tmax. This
violates the hard per-step range bound unconditionally (independent of the
instance's fuel budget B), so the checker's feasibility gate must reject it
on every test case."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    L = int(data[p]); p += 1
    Tmax = int(data[p]); p += 1

    n = 3
    print(n)
    print(" ".join([str(Tmax + 5)] * n))


main()
