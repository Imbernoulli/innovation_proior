# TIER: strong
# Incremental simulated annealing over slot swaps, maximizing |A+A|. Starts from the
# greedy Mian-Chowla packing, then repeatedly relocates one coupler to a free slot,
# maintaining a multiplicity table of sums so each move is O(n) and the distinct-sum
# count is tracked incrementally. Fully deterministic (RNG seeded from the instance).
import sys, random
from collections import Counter


def mian_chowla(n, M):
    A = [0]; sums = {0}; x = 1
    while len(A) < n and x <= M:
        news = set(); ok = True
        for a in A:
            v = a + x
            if v in sums or v in news:
                ok = False; break
            news.add(v)
        if ok:
            news.add(2 * x); A.append(x); sums |= news
        x += 1
    if len(A) < n:
        used = set(A)
        for y in range(M + 1):
            if len(A) >= n: break
            if y not in used:
                A.append(y); used.add(y)
    return A[:n]


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); M = int(toks[1])
    rng = random.Random(1000003 * n + M)

    A = mian_chowla(n, M)
    inset = set(A)
    while len(A) < n:
        y = rng.randint(0, M)
        if y not in inset:
            inset.add(y); A.append(y)

    cnt = Counter()
    for i in range(n):
        for j in range(i, n):
            cnt[A[i] + A[j]] += 1
    cur = len(cnt)
    best = cur
    bestA = A[:]

    iters = 60000
    for _ in range(iters):
        i = rng.randrange(n)
        old = A[i]
        new = rng.randint(0, M)
        if new in inset:
            continue
        # remove old coupler's contributions
        d = 0
        for j in range(n):
            if j == i:
                continue
            v = old + A[j]
            cnt[v] -= 1
            if cnt[v] == 0:
                d -= 1
        v = old + old
        cnt[v] -= 1
        if cnt[v] == 0:
            d -= 1
        # add new coupler's contributions
        A[i] = new
        for j in range(n):
            if j == i:
                continue
            v = new + A[j]
            if cnt[v] == 0:
                d += 1
            cnt[v] += 1
        v = new + new
        if cnt[v] == 0:
            d += 1
        cnt[v] += 1

        if d >= 0 or rng.random() < 0.02:
            cur += d
            inset.discard(old); inset.add(new)
            if cur > best:
                best = cur; bestA = A[:]
        else:
            # revert: undo new, restore old
            for j in range(n):
                if j == i:
                    continue
                cnt[new + A[j]] -= 1
            cnt[new + new] -= 1
            A[i] = old
            for j in range(n):
                if j == i:
                    continue
                cnt[old + A[j]] += 1
            cnt[old + old] += 1

    sys.stdout.write(" ".join(map(str, sorted(bestA))) + "\n")


if __name__ == "__main__":
    main()
