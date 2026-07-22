# TIER: invalid
# Emits an out-of-range edge index -> the checker's feasibility gate rejects it (0.0).
import sys


def main():
    raw = sys.stdin.read().split()
    it = iter(raw)
    N = int(next(it)); M = int(next(it))
    print(M + 999999)


main()
