# TIER: trivial
"""Walk the explicitly-listed backbone valve chain. Always valid, never short."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    for _ in range(N):
        next(it)                      # capacities, unused
    V0 = int(next(it)); target = int(next(it)); Nb = int(next(it))
    for _ in range(4):
        next(it)                      # Lo_num Lo_den Hi_num Hi_den, unused

    ops = [int(next(it)) for _ in range(Nb)]
    print(len(ops))
    print(" ".join(map(str, ops)))


if __name__ == "__main__":
    main()
