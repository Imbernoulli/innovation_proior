import sys, random

# gen.py <testId>  --  "Skeleton-Key Polynomials over F_p"
#
# Family: subgroup-annihilator-sparse-poly.
#
# PLANT: pick a prime p and, inside F_p^*, a "target set" S built as the union of
# k in {2,3,4} cosets of DISTINCT multiplicative subgroups (each subgroup order a
# genuine divisor of p-1), plus a little noise (extra scattered points, and a few
# holes punched out of the cosets).  One of the k cosets is deliberately a SMALL
# order (12..25) -- the "unit" coset u -- and the rest are meaningfully BIGGER
# ("big" cosets, strictly above 25).  This asymmetry is exactly what separates a
# solver that only ever finds ONE coset (misses most of the big mass, or only
# stumbles on the small one) from a solver that detects and COMPOSES several.
#
# x^d - c has root set exactly one coset of the order-d subgroup of F_p^* (for a
# divisor d of p-1 and c a d-th power residue).  A product of such binomials
# (expanded to its sparse monomial form) is therefore a "skeleton key": it opens
# (vanishes at) exactly the union of those cosets, using only O(1) terms.
#
# Deterministic: everything seeded from testId only.

# (p_lo, p_hi, k)  -- prime search window and number of planted cosets, ladder
# small/easy -> large/adversarial.  k>=3 on most cases forces the trap: a solver
# that only ever finds a SINGLE best coset (the natural "greedy" move) leaves most
# of S uncaptured whenever k>=3.
SPECS = {
    1:  (1000,   1600,  2),
    2:  (1600,   2600,  2),
    3:  (2600,   4200,  3),
    4:  (4200,   6200,  3),
    5:  (6200,   9000,  3),
    6:  (9000,  12000,  4),
    7:  (12000, 15500,  4),
    8:  (15500, 19000,  4),
    9:  (19000, 23000,  4),
    10: (23000, 27000,  4),
}

SMALL_LO, SMALL_HI = 12, 25   # band for the "unit" coset u
BIG_LO = 30                    # big cosets strictly above the small band


def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for sp in small_primes:
        if n % sp == 0:
            return n == sp
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small_primes:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


def divisors(n):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return sorted(ds)


def prime_factors(n):
    fs = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return sorted(fs)


def find_primitive_root(p, pf):
    for g in range(2, p):
        ok = True
        for q in pf:
            if pow(g, (p - 1) // q, p) == 1:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("no primitive root found")


def pick_structure(rng, p_lo, p_hi, k):
    """Search upward from a seeded start for a prime p with:
       - a divisor u in [SMALL_LO, SMALL_HI]
       - at least (k-1) divisors in [BIG_LO, ...] whose SUM can be steered close
         to a target multiple of u (keeps F_strong/F_baseline in a sane range).
    Returns (p, g, pf, u, bigs) with bigs a sorted list of k-1 big orders.
    """
    start = rng.randint(p_lo, p_hi)
    span = (p_hi - p_lo) + 4000
    for delta in range(span):
        p = start + delta
        if p > p_hi + 4000:
            continue
        if not is_prime(p):
            continue
        n = p - 1
        divs = divisors(n)
        small_cands = [d for d in divs if SMALL_LO <= d <= SMALL_HI]
        if not small_cands:
            continue
        u = max(small_cands)  # prefer the larger end of the small band (steadier baseline)
        big_pool = [d for d in divs if d >= BIG_LO and d <= 60 * u and d <= n // 2]
        if len(big_pool) < (k - 1):
            continue
        # target: total mass of the (k-1) big cosets ~ 6.0*u, spread evenly
        target_each = max(BIG_LO, int(round(6.0 * u / (k - 1))))
        big_pool_sorted = sorted(big_pool, key=lambda d: abs(d - target_each))
        chosen = []
        for d in big_pool_sorted:
            if d not in chosen:
                chosen.append(d)
            if len(chosen) == k - 1:
                break
        if len(chosen) < k - 1:
            continue
        if sum(chosen) > 8.0 * u:
            # trim: try to swap in smaller pool members
            chosen = sorted(big_pool)[:k - 1]
            if len(chosen) < k - 1 or sum(chosen) > 9.5 * u:
                continue
        pf = prime_factors(n)
        g = find_primitive_root(p, pf)
        return p, g, pf, u, sorted(chosen)
    raise RuntimeError("no suitable prime found in window")


def main():
    tid = int(sys.argv[1])
    p_lo, p_hi, k = SPECS[tid]
    rng = random.Random(19348411 + 1000003 * tid)

    p, g, pf, u, bigs = pick_structure(rng, p_lo, p_hi, k)
    n = p - 1
    orders = [u] + bigs  # k subgroup orders

    def subgroup_coset(order, rep_exp):
        step = n // order
        rep = pow(g, rep_exp, p)
        return {(rep * pow(g, step * j, p)) % p for j in range(order)}

    union = set()
    for order in orders:
        rep_exp = rng.randrange(n)
        union |= subgroup_coset(order, rep_exp)

    union_list = sorted(union)
    m = len(union_list)

    # holes: remove a few points from the union (unavoidable algebraic false
    # positives for anyone who reconstructs the exact cosets)
    n_holes = max(0, round(0.03 * m))
    holes = set(rng.sample(union_list, min(n_holes, m))) if n_holes > 0 else set()

    S = union - holes

    # extra noise: scattered points NOT in any coset (unreachable by any O(1)
    # binomial product -- these cap the achievable score below 1.0)
    n_extra = max(1, round(0.05 * m))
    pool = [x for x in range(1, p) if x not in union]
    extra = set(rng.sample(pool, min(n_extra, len(pool))))

    S |= extra
    S = sorted(S)

    # term budget: enough for a product of up to 4 binomials (<=16 monomials)
    # plus slack, but nowhere near |S| (rules out plain interpolation)
    T = 24

    out = []
    out.append("%d %d" % (p, T))
    out.append("%d" % len(S))
    out.append(" ".join(str(x) for x in S))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
