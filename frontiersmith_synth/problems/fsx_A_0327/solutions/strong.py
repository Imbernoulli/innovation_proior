# TIER: strong
"""Strong: engineer a "more sums than differences" (MSTD) set with rho > 1.

Strategy: tile a hand-tuned high-ratio dense MSTD seed (rho ~ 1.068 on 16 points) to reach
n beacons, then hill-climb with small local shifts to push rho as high as possible within a
fixed iteration budget. Deterministic (seeded); stays well inside the time limit."""
import sys
import random

# A dense 16-point seed with |A+A| > |A-A| (rho ~ 1.068), spread over [1, 32].
SEED = [1, 2, 3, 6, 9, 11, 12, 13, 19, 21, 22, 24, 28, 29, 31, 32]


def rho_parts(A):
    s = set()
    d = set()
    for a in A:
        for b in A:
            s.add(a + b)
            d.add(a - b)
    return len(s), len(d)


def rho(A):
    ns, nd = rho_parts(A)
    return ns / nd


def build(n, V):
    L = max(SEED) + 2
    A = []
    j = 0
    while len(A) < n:
        for s in SEED:
            A.append(s + L * j)
            if len(A) >= n:
                break
        j += 1
    A = sorted(set(A))[:n]
    rng = random.Random(12345)
    while len(A) < n:
        y = rng.randrange(V + 1)
        if y not in A:
            A.append(y)
    return set(A), rng


def main():
    data = sys.stdin.read().split()
    n, V = int(data[0]), int(data[1])
    A, rng = build(n, V)
    cur = rho(A)
    best = set(A)
    best_r = cur
    Al = list(A)
    iters = 4000 if n <= 48 else 2500
    for _ in range(iters):
        i = rng.randrange(len(Al))
        old = Al[i]
        y = old + rng.choice([-3, -2, -1, 1, 2, 3])
        if y < 0 or y > V or y in A:
            continue
        A.discard(old)
        A.add(y)
        Al[i] = y
        r = rho(A)
        if r >= cur:
            cur = r
            if r > best_r:
                best_r = r
                best = set(A)
        else:
            A.discard(y)
            A.add(old)
            Al[i] = old
    out = sorted(best)
    sys.stdout.write(" ".join(map(str, out)) + "\n")


if __name__ == "__main__":
    main()
