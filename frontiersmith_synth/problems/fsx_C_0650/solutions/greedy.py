# TIER: greedy
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1
        return v
    N = int(nxt()); T = int(nxt()); K = int(nxt())
    nxt(); nxt(); nxt()  # alpha, decay, gamma
    songs = []
    for _ in range(N):
        e = int(nxt()); d = int(nxt())
        for _ in range(K):
            nxt()
        songs.append((e, d))

    # The obvious first attempt: energy is what pumps the crowd, so play the highest-
    # energy songs you can afford, loudest first, in a classic fractional-knapsack-style
    # descending sweep. It ignores state dynamics/fatigue entirely.
    idxs = sorted(range(N), key=lambda i: -songs[i][0])
    order = []
    tot = 0
    for i in idxs:
        d = songs[i][1]
        if tot + d <= T:
            order.append(i)
            tot += d

    print(len(order))
    print(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
