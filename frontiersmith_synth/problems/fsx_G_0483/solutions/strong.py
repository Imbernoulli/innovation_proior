# TIER: strong
# Greedy-insertion seed, then a bounded, deterministically-seeded hill climb of
# random transpositions that keeps any non-worsening swap. This finds much
# lower-coincidence permutations than the plain greedy pass but, with a fixed
# small iteration budget, still leaves genuine headroom on the largest orders
# (where the true minimum is open) -- it does NOT implement any proven-optimal
# algebraic Costas construction.
import sys
import random
from collections import Counter


def coincidences(p):
    n = len(p)
    c = Counter()
    for i in range(n):
        pi = p[i]
        for j in range(i + 1, n):
            c[(j - i, p[j] - pi)] += 1
    return sum(v - 1 for v in c.values())


def greedy_insert(n):
    used = [False] * n
    p = []
    cnt = Counter()
    for col in range(n):
        best_v = None
        best_add = None
        best_new = None
        for v in range(n):
            if used[v]:
                continue
            add = 0
            newv = Counter()
            for k in range(col):
                vec = (col - k, v - p[k])
                if cnt[vec] + newv[vec] >= 1:
                    add += 1
                newv[vec] += 1
            if best_add is None or add < best_add:
                best_add = add
                best_v = v
                best_new = newv
        p.append(best_v)
        used[best_v] = True
        cnt.update(best_new)
    return p


def main():
    n = int(sys.stdin.read().split()[0])
    p = greedy_insert(n)
    cur = coincidences(p)
    rng = random.Random(20240607)  # fixed seed -> deterministic
    ITERS = 400  # bounded budget: leaves headroom on the hardest orders
    for _ in range(ITERS):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        p[i], p[j] = p[j], p[i]
        c2 = coincidences(p)
        if c2 <= cur:
            cur = c2
        else:
            p[i], p[j] = p[j], p[i]
    print(" ".join(str(x) for x in p))


if __name__ == "__main__":
    main()
