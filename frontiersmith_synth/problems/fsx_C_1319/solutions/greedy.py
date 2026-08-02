# TIER: greedy
"""The obvious recipe: 'strong bonds are the stable/important ones, so lock
them in first to maximize early yield.' Sort every candidate bond -- target
and decoy alike -- by strength descending and enable them one at a time in
that order. This is exactly the kinetic-trap anti-pattern the family is
built around: decoys are always the strongest bonds here, so they always win
the race to grab their monomers, freezing before the (weaker, correct)
target bonds ever get a turn on any monomer they compete for."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    Tmax = int(data[idx]); idx += 1
    theta0 = int(data[idx]); idx += 1
    bonds = []
    for _ in range(M):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        s = int(data[idx]); idx += 1
        typ = data[idx]; idx += 1
        bonds.append((u, v, s, typ))

    order = sorted(range(M), key=lambda j: (-bonds[j][2], j))
    times = [0] * M
    for rank, j in enumerate(order):
        times[j] = min(Tmax, rank + 1)

    print(" ".join(map(str, times)))


if __name__ == "__main__":
    main()
