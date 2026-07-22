# TIER: strong
# Insight: coverage factors through the CRT.  With B = B1 x B2 x B3 (residue product),
#   D(B) intersect T  =  (D(B1) cap A1) x (D(B2) cap A2) x (D(B3) cap A3),
# so |covered| = prod_i |D(Bi) cap Ai|.  Hence:
#   (1) SPLIT the budget k = k1*k2*k3 across the coprime factors,
#   (2) DESIGN a near-perfect per-factor difference cover Bi (tiny search in Z_{m_i}),
#   (3) take the CRT product.
# This multiplies clean per-factor coverage instead of searching the huge ring directly.
import sys, random, itertools

def dset(B, m):
    s = set()
    for x in B:
        for y in B:
            s.add((x - y) % m)
    return s

def best_factor(m, k, As, rng):
    best = None; bc = -1
    for _ in range(60):
        B = rng.sample(range(m), k)
        c = len(dset(B, m) & As)
        if c > bc:
            bc = c; best = B[:]
    pool = list(range(m))
    for _ in range(3):
        improved = False
        for idx in range(k):
            rng.shuffle(pool)
            for cand in pool[:min(m, 300)]:
                if cand in best:
                    continue
                tr = best[:]; tr[idx] = cand
                c = len(dset(tr, m) & As)
                if c > bc:
                    bc = c; best = tr[:]; improved = True
        if not improved:
            break
    return best, bc

def crt_point(rs, ms):
    x = rs[0] % ms[0]; M = ms[0]
    for v, mi in zip(rs[1:], ms[1:]):
        inv = pow(M % mi, -1, mi)
        tt = ((v - x) * inv) % mi
        x = x + M * tt; M *= mi
    return x

def main():
    tk = sys.stdin.buffer.read().split()
    it = iter(tk)
    t = int(next(it)); ms = [int(next(it)) for _ in range(t)]; k = int(next(it))
    A = []
    for i in range(t):
        s = int(next(it)); A.append(set(int(next(it)) % ms[i] for _ in range(s)))

    # candidate factorizations of k into t=3 factors (each >= 2)
    facs = []
    for a in range(2, k + 1):
        if k % a:
            continue
        for b in range(a, k // a + 1):
            if (k // a) % b:
                continue
            c = k // a // b
            if c >= b:
                facs.append((a, b, c))
    small = [f for f in facs if max(f) <= 6]
    if small:
        facs = small
    kvals = set()
    for f in facs:
        for v in f:
            kvals.add(v)

    rng = random.Random(7)
    cache = {}
    for i in range(t):
        for v in kvals:
            if 2 <= v <= ms[i]:
                cache[(i, v)] = best_factor(ms[i], v, A[i], rng)

    bestprod = -1; bestsets = None
    for f in facs:
        for perm in set(itertools.permutations(f)):
            if any((i, perm[i]) not in cache for i in range(t)):
                continue
            prod = 1; sets = []
            for i in range(t):
                B, c = cache[(i, perm[i])]
                prod *= c; sets.append(B)
            if prod > bestprod:
                bestprod = prod; bestsets = sets

    pts = []
    seen = set()
    for combo in itertools.product(*bestsets):
        x = crt_point(list(combo), ms)
        if x not in seen:
            seen.add(x); pts.append(x)
    print(len(pts))
    sys.stdout.write("\n".join(map(str, pts)) + "\n")

if __name__ == "__main__":
    main()
