# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1
        return v
    N = int(nxt()); T = int(nxt()); K = int(nxt())
    nxt(); nxt(); nxt()  # alpha, decay, gamma -- unused by the trivial strategy
    songs = []
    for _ in range(N):
        e = int(nxt()); d = int(nxt())
        for _ in range(K):
            nxt()
        songs.append((e, d))

    # Reproduce the checker's own unambitious fallback: calmest (lowest-energy) songs
    # first, and don't even use the whole time budget -- stop at 2/5 of T.
    cap = (2 * T) // 5
    idxs = sorted(range(N), key=lambda i: (songs[i][0], songs[i][1], i))
    order = []
    tot = 0
    for i in idxs:
        d = songs[i][1]
        if tot + d <= cap:
            order.append(i)
            tot += d

    print(len(order))
    print(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
