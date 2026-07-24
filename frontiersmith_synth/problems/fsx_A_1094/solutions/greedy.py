# TIER: greedy
# The canonical first approach: repeatedly play the legal move that most
# increases the immediately collected weight; stop when no strictly improving
# move exists. Myopic: it never crosses a plateau or a dip, so on planted
# instances it stalls far below the reachable-class ceiling.
import sys
from collections import Counter

def coverage(cnt, w):
    return sum(w[c - 1] for c, k in cnt.items() if k > 0)

def apply(cnt, t, i):
    if t == "s":
        cnt[i] -= 2
        if cnt[i] == 0: del cnt[i]
        cnt[i - 1] = cnt.get(i - 1, 0) + 1
        cnt[i + 1] = cnt.get(i + 1, 0) + 1
    else:
        cnt[i - 1] -= 1
        if cnt[i - 1] == 0: del cnt[i - 1]
        cnt[i + 1] -= 1
        if cnt[i + 1] == 0: del cnt[i + 1]
        cnt[i] = cnt.get(i, 0) + 2

def main():
    d = sys.stdin.read().split()
    N, M = int(d[0]), int(d[1])
    w = list(map(int, d[2:2 + N]))
    cnt = Counter(map(int, d[2 + N:2 + N + M]))

    moves = []
    cur = coverage(cnt, w)
    while len(moves) < 400:
        best, bestgain = None, 0
        for i in range(2, N):
            for t in ("s", "g"):
                if t == "s" and cnt.get(i, 0) < 2:
                    continue
                if t == "g" and (cnt.get(i - 1, 0) < 1 or cnt.get(i + 1, 0) < 1):
                    continue
                snap = Counter(cnt)
                apply(snap, t, i)
                g = coverage(snap, w) - cur
                if g > bestgain or (g == bestgain and g > 0 and best is not None
                                    and (t, i) < best):
                    best, bestgain = (t, i), g
        if best is None:
            break
        apply(cnt, best[0], best[1])
        cur += bestgain
        moves.append(best)

    print(len(moves))
    for t, i in moves:
        print(t, i)

if __name__ == "__main__":
    main()
