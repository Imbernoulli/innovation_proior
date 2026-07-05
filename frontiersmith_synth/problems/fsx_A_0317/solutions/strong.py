# TIER: strong
# Seeded random-restart hill climbing over the post set to maximize |A+A|.
# Deterministic: the RNG is seeded ONLY from the instance `seed`, so reruns are
# bit-for-bit identical. One restart is warm-started from the greedy B_2^+ set;
# the others start from random layouts. A move relocates one box to a free post
# and is accepted iff it does not decrease the distinct-sum count.
import sys
import random


def sumset_size(A):
    s = set()
    m = len(A)
    for i in range(m):
        ai = A[i]
        for j in range(i, m):
            s.add(ai + A[j])
    return len(s)


def greedy_seed(n, M):
    A = []
    sums = set()
    for c in range(0, M + 1):
        if len(A) >= n:
            break
        new = set()
        ok = True
        for a in A:
            v = a + c
            if v in sums or v in new:
                ok = False
                break
            new.add(v)
        if ok:
            v = c + c
            if v in sums or v in new:
                ok = False
            else:
                new.add(v)
        if ok:
            A.append(c)
            sums |= new
    if len(A) < n:
        used = set(A)
        c = 0
        while len(A) < n and c <= M:
            if c not in used:
                A.append(c)
                used.add(c)
            c += 1
    return A[:n]


def climb(cur, M, rnd, iters):
    curset = set(cur)
    best = sumset_size(cur)
    n = len(cur)
    for _ in range(iters):
        i = rnd.randrange(n)
        old = cur[i]
        nv = rnd.randint(0, M)
        if nv in curset:
            continue
        curset.discard(old); curset.add(nv); cur[i] = nv
        s = sumset_size(cur)
        if s >= best:
            best = s
        else:
            curset.discard(nv); curset.add(old); cur[i] = old
    return cur, best


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); M = int(data[1])
    seed = int(data[2]) if len(data) > 2 else 0
    rnd = random.Random(seed)

    # iteration budget shrinks as n grows to stay comfortably in the time limit
    iters = max(600, 24000 // n)

    best_set = list(range(n))
    best_val = sumset_size(best_set)

    # restart 0: warm start from the greedy B_2^+ packing
    start = greedy_seed(n, M)
    cand, val = climb(list(start), M, rnd, iters)
    if val > best_val:
        best_val, best_set = val, list(cand)

    # additional random restarts
    for _ in range(4):
        s = set()
        while len(s) < n:
            s.add(rnd.randint(0, M))
        cand, val = climb(sorted(s), M, rnd, iters)
        if val > best_val:
            best_val, best_set = val, list(cand)

    print(" ".join(str(x) for x in best_set))


if __name__ == "__main__":
    main()
