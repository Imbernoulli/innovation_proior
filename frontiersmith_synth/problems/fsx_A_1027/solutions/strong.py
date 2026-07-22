# TIER: strong
import sys


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


def poly_mul(A, B, p):
    """Multiply two sparse polys (dict exp->coeff mod p)."""
    C = {}
    for ea, ca in A.items():
        if ca == 0:
            continue
        for eb, cb in B.items():
            if cb == 0:
                continue
            e = ea + eb
            C[e] = (C.get(e, 0) + ca * cb) % p
    return {e: c for e, c in C.items() if c != 0}


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it))
    T = int(next(it))
    m = int(next(it))
    S = [int(next(it)) for _ in range(m)]

    LAMBDA = 2
    n = p - 1

    # --- INSIGHT: don't stop at the single best coset -- score EVERY candidate
    # subgroup order, then COMPOSE the several independently-good ones into a
    # single product polynomial.  x^d - c has root set exactly one coset of the
    # order-d subgroup, and (over a field) roots of a PRODUCT of such binomials
    # are exactly the UNION of the individual coset root sets -- so multiplying
    # 2-4 winning binomials together captures several cosets of S at once, in a
    # still-tiny number of expanded monomial terms. ---
    cand = [d for d in divisors(n) if 2 <= d <= n // 2]

    scored = []  # (F, d, c, cnt)
    for d in cand:
        counts = {}
        for x in S:
            v = pow(x, d, p)
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            continue
        c = max(counts, key=lambda k: counts[k])
        cnt = counts[c]
        F = cnt - LAMBDA * (d - cnt)
        if F > 0:
            scored.append((F, d, c, cnt))

    scored.sort(key=lambda t: -t[0])

    # Greedily compose up to 4 factors, skipping ones that would blow the
    # exponent range or that duplicate an already-chosen subgroup order, and
    # tracking which points of S each factor's coset actually explains so we
    # don't "pay" for heavily overlapping candidates.
    chosen = []
    covered = set()
    total_exp = 0
    for F, d, c, cnt in scored:
        if len(chosen) >= 4:
            break
        # points of S this candidate coset would newly explain
        new_pts = {x for x in S if pow(x, d, p) == c}
        gain = len(new_pts - covered)
        if gain <= 0:
            continue
        if total_exp + d > n - 1:
            continue
        chosen.append((d, c))
        covered |= new_pts
        total_exp += d

    if not chosen:
        print(0)
        return

    poly = {0: 1}
    for d, c in chosen:
        factor = {d: 1, 0: (-c) % p}
        poly = poly_mul(poly, factor, p)

    terms = [(e, a) for e, a in poly.items() if a != 0]
    if len(terms) > T:
        # extremely defensive fallback (shouldn't trigger with <=4 factors
        # and T=24): keep only the single best factor
        d, c = chosen[0]
        terms = [(d, 1), (0, (-c) % p)]

    print(len(terms))
    for e, a in terms:
        print(e, a)


if __name__ == "__main__":
    main()
