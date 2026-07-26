import sys

# gen.py <testId> -- prints ONE gauss-period-basis-sparsity instance to stdout.
#
# An instance is a finite field F_{p^k}, presented as F_p[x]/(f(x)) for a monic
# irreducible degree-k polynomial f found by deterministic seeded search. Whether
# an OPTIMAL Gauss-period normal basis exists for (p,k) is a number-theoretic
# property of (p,k) ALONE (a "type-1/2" condition -- see statement); it does not
# depend on which f we happened to find. Difficulty (field size k) grows with
# testId, and cases are curated so that >=3 of the 10 land in the "type-1/2 holds"
# regime where a principled construction dramatically outperforms any basis a
# solver could stumble onto without checking the regime first.
#
# (p, k, find_irreducible-seed) per testId -- fixed by problem authoring/tuning.
SPECS = {
    1:  (2, 9,  18),
    2:  (2, 10, 16),
    3:  (2, 11, 22),
    4:  (2, 12, 17),
    5:  (2, 13, 6),
    6:  (2, 14, 32),
    7:  (2, 8,  14),
    8:  (2, 16, 4),
    9:  (3, 16, 38),
    10: (2, 17, 6),
}


def is_zero(a):
    return all(c == 0 for c in a)


def reduce_full(raw, f, p, k):
    a = raw[:]
    for deg_ in range(len(a) - 1, k - 1, -1):
        c = a[deg_]
        if c:
            shift = deg_ - k
            for i in range(k):
                a[shift + i] = (a[shift + i] - c * f[i]) % p
            a[deg_] = 0
    return a[:k]


def frob_step(a, f, p, k):
    raw = [0] * ((k - 1) * p + 1)
    for i, c in enumerate(a):
        if c:
            raw[i * p] = (raw[i * p] + c) % p
    return reduce_full(raw, f, p, k)


def deg(a):
    d = len(a) - 1
    while d > 0 and a[d] == 0:
        d -= 1
    return d


def trim_deg(a):
    a = a[:]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def poly_mod_full(a, g, p):
    a = a[:]
    dg_ = deg(g)
    lc_inv = pow(g[dg_], p - 2, p)
    while deg(a) >= dg_ and not (len(a) == 1 and a[0] == 0):
        da = deg(a)
        if da < dg_:
            break
        c = (a[da] * lc_inv) % p
        shift = da - dg_
        for i in range(dg_ + 1):
            a[shift + i] = (a[shift + i] - c * g[i]) % p
        a = trim_deg(a)
    return a


def poly_coprime(a, f, p, k):
    A = trim_deg(a[:])
    F = trim_deg(f[:] + [1])
    while not (len(A) == 1 and A[0] == 0):
        if deg(A) < deg(F):
            A, F = F, A
            continue
        A = poly_mod_full(A, F, p)
    return deg(F) == 0 and F[0] != 0


def factorize(m):
    fac = {}
    d = 2
    x = m
    while d * d <= x:
        while x % d == 0:
            fac[d] = fac.get(d, 0) + 1
            x //= d
        d += 1
    if x > 1:
        fac[x] = fac.get(x, 0) + 1
    return fac


def is_irreducible(f, p, k):
    if k == 1:
        return True
    x = [0] * k
    x[1] = 1
    cur = x[:]
    for _ in range(k):
        cur = frob_step(cur, f, p, k)
    diff = cur[:]
    diff[1] = (diff[1] - 1) % p
    if not is_zero(diff):
        return False
    for q in factorize(k):
        d = k // q
        cur2 = x[:]
        for _ in range(d):
            cur2 = frob_step(cur2, f, p, k)
        diff2 = cur2[:]
        diff2[1] = (diff2[1] - 1) % p
        if not poly_coprime(diff2, f, p, k):
            return False
    return True


def find_irreducible(p, k, seed, tries=6000):
    import random
    rng = random.Random(seed)
    for _ in range(tries):
        f = [rng.randrange(p) for _ in range(k)]
        if f[0] == 0:
            f[0] = rng.randrange(1, p)
        if is_irreducible(f, p, k):
            return f
    return None


def main():
    tid = int(sys.argv[1])
    p, k, sd = SPECS[tid]
    f = find_irreducible(p, k, seed=5000 + 100 * sd + k * 7 + p)
    if f is None:
        raise SystemExit("no irreducible polynomial found (author error)")
    print(p, k)
    print(" ".join(map(str, f)))


if __name__ == "__main__":
    main()
