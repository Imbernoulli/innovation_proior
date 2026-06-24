import sys

def solve(data):
    it = iter(data)
    n = int(next(it)); K = int(next(it)); L = int(next(it)); R = int(next(it))
    v = [int(next(it)) for _ in range(n)]

    # Enumerate every way to partition the unit-segments [0, n) into contiguous
    # billets whose lengths are each in [L, R]. Each billet over [j, i) costs
    # K + |sum of v on [j, i)|. Take the minimum total cost; -1 if none valid.
    INF = float('inf')
    best = [INF]

    S = [0]*(n+1)
    for i in range(n):
        S[i+1] = S[i] + v[i]

    def rec(pos, acc):
        if acc >= best[0]:
            return
        if pos == n:
            best[0] = min(best[0], acc)
            return
        for length in range(L, R+1):
            np = pos + length
            if np > n:
                break
            seg = S[np] - S[pos]
            rec(np, acc + K + abs(seg))

    rec(0, 0)
    if best[0] == INF:
        return "-1"
    return str(best[0])

def main():
    data = sys.stdin.read().split()
    print(solve(data))

main()
