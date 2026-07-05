# TIER: strong
# Best-of several deterministic constructions for maximizing F = |A+A| + |A-A|
# under a size budget k and corridor [0,V]:
#   (1) Mian-Chowla greedy Sidon layout (scan from 0);
#   (2) algebraic Erdos-Turan Sidon set {2*p*i + (i*i mod p)} scaled to fit;
#   (3) several seeded marginal-gain restarts: from a random seed milepost,
#       repeatedly add the candidate milepost giving the largest marginal gain in
#       distinct sums + distinct differences, until k stations or no gain.
# Deterministic (fixed RNG seed). Returns the best F found.
import sys
import random


def compute_F(A):
    sums = set()
    diffs = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            d = ai - aj
            diffs.add(d)
            diffs.add(-d)
    return len(sums) + len(diffs)


def mian_chowla(k, V):
    A = []
    sums = set()
    for x in range(V + 1):
        if len(A) >= k:
            break
        cand = set()
        ok = True
        for a in A:
            s = a + x
            if s in sums or s in cand:
                ok = False
                break
            cand.add(s)
        if not ok:
            continue
        d = x + x
        if d in sums or d in cand:
            continue
        cand.add(d)
        sums |= cand
        A.append(x)
    return A if A else [0]


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def erdos_turan(k, V):
    # try the largest prime p <= k such that the Sidon set fits in [0,V]
    best = None
    p = k
    while p >= 2:
        if is_prime(p):
            S = sorted(set((2 * p * i + (i * i) % p) for i in range(p)))
            if len(S) <= k and (max(S) <= V):
                if best is None or len(S) > len(best):
                    best = S
                    break  # largest fitting prime is good enough
        p -= 1
    return best


def marginal_restart(k, V, rng):
    # candidate pool
    if V + 1 <= 500:
        pool_full = list(range(V + 1))
    else:
        pool_full = rng.sample(range(V + 1), 500)
    start = rng.choice(pool_full)
    A = [start]
    sums = {2 * start}
    diffs = {0}
    chosen = {start}
    while len(A) < k:
        best_gain = -1
        best_x = None
        for x in pool_full:
            if x in chosen:
                continue
            add_s = set()
            g = 0
            for a in A:
                s = a + x
                if s not in sums and s not in add_s:
                    add_s.add(s)
                    g += 1
            s2 = 2 * x
            if s2 not in sums and s2 not in add_s:
                g += 1
            add_d = set()
            for a in A:
                d1 = x - a
                if d1 not in diffs and d1 not in add_d:
                    add_d.add(d1)
                    g += 1
                d2 = a - x
                if d2 not in diffs and d2 not in add_d:
                    add_d.add(d2)
                    g += 1
            # tie-break randomly for restart diversity
            if g > best_gain or (g == best_gain and rng.random() < 0.5):
                best_gain = g
                best_x = x
        if best_x is None or best_gain <= 0:
            break
        # commit best_x
        for a in A:
            sums.add(a + best_x)
            diffs.add(best_x - a)
            diffs.add(a - best_x)
        sums.add(2 * best_x)
        A.append(best_x)
        chosen.add(best_x)
    return A


def main():
    data = sys.stdin.read().split()
    k = int(data[0])
    V = int(data[1])

    candidates = []
    candidates.append(mian_chowla(k, V))
    et = erdos_turan(k, V)
    if et:
        candidates.append(et)

    rng = random.Random(0xC0FFEE ^ (k * 1000003) ^ (V * 97))
    restarts = 6
    for _ in range(restarts):
        candidates.append(marginal_restart(k, V, rng))

    best = None
    best_F = -1
    for A in candidates:
        if not A:
            continue
        # ensure validity (distinct, in range, size <= k)
        A2 = sorted(set(a for a in A if 0 <= a <= V))[:k]
        F = compute_F(A2)
        if F > best_F:
            best_F = F
            best = A2

    if not best:
        best = [0]
    out = [str(len(best))]
    out += [str(x) for x in best]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
