# TIER: greedy
# Obvious, textbook approach: press the most-frequently-cued cells onto the
# CENTRE grooves (classic "hot data near the middle" disk-layout heuristic),
# coldest cells at the rim. Uses only per-cell CUE FREQUENCY -- it never
# looks at which cells are cued close together in time, so it cannot see the
# side/cluster structure the look-ahead window actually rewards.
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    N = int(data[p]); T = int(data[p + 1]); cap = int(data[p + 2]); w = int(data[p + 3]); M = int(data[p + 4])
    p += 5
    Q = [int(data[p + k]) for k in range(M)]

    freq = [0] * N
    for q in Q:
        freq[q] += 1

    order = sorted(range(N), key=lambda i: (-freq[i], i))

    center = T // 2
    slots = []
    d = 0
    while len(slots) < N:
        for cand in ((center - d), (center + d)) if d > 0 else (center,):
            if 0 <= cand < T:
                slots.extend([cand] * cap)
        d += 1

    trk = [0] * N
    for cell, slot in zip(order, slots):
        trk[cell] = slot

    sys.stdout.write(" ".join(str(x) for x in trk) + "\n")


if __name__ == "__main__":
    main()
