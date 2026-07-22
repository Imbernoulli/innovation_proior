import sys, random

# gen.py <testId> -- prints ONE degree-n polynomial P(x) = a_0 + a_1 x + ... + a_n x^n
# to stdout (n even, 4 <= n <= 44).
#
# PLANTED STRUCTURE (never told to the solver): every instance is constructed as
# P(x) = g(h(x)) with h(x) = x^2 + b*x + c  (b != 0, so the raw coefficient list
# shows NO visible symmetry -- all a_i are generically nonzero) and g a degree-k
# polynomial (n = 2k) with a nonzero leading coefficient.
#
# This means P, after the affine "completing the square" substitution x' = x + b/2,
# becomes an EVEN polynomial in x' -- so P(x) can be evaluated as
#   y = (x + b/2)^2        (1 multiplication)
#   Horner-evaluate A(y)   (k multiplications, A = the shifted even coefficients)
# for a total of k+1 = n/2+1 multiplications, versus Horner's n on the raw
# coefficients. Recovering b, and the Taylor shift needed to find A, requires
# reading the actual coefficient values -- an average coder who just runs
# textbook Horner never notices any of this (raw a_i show no pattern at all,
# since b != 0 destroys the naive "check for zero odd coefficients" shortcut).
#
# Difficulty ladder: k (half-degree) grows with testId; the composition weights
# also grow, so both the multiplication-count gap AND the coefficient magnitude
# grow toward the adversarial/large end of the ladder.

SCHEDULE = {
    1:  3,
    2:  4,
    3:  5,
    4:  6,
    5:  7,
    6:  8,
    7:  9,
    8:  10,
    9:  11,
    10: 12,
}


def poly_mul(A, B):
    C = [0] * (len(A) + len(B) - 1)
    for i, av in enumerate(A):
        if av == 0:
            continue
        for j, bv in enumerate(B):
            if bv:
                C[i + j] += av * bv
    return C


def poly_add(A, B):
    m = max(len(A), len(B))
    A = A + [0] * (m - len(A))
    B = B + [0] * (m - len(B))
    return [x + y for x, y in zip(A, B)]


def compose(g_coeffs, h_coeffs):
    """Return coefficients (low->high) of g(h(x)), given g_coeffs[i] = coeff of y^i in g(y)."""
    result = [0]
    hp = [1]
    for i, gi in enumerate(g_coeffs):
        if i > 0:
            hp = poly_mul(hp, h_coeffs)
        if gi != 0:
            term = [gi * c for c in hp]
            result = poly_add(result, term)
    return result


def build(tid, k, rng):
    b = rng.choice([v for v in range(-3, 4) if v != 0])
    c = rng.randint(-3, 3)
    gk = rng.randint(1, 2)
    g_low = [rng.randint(-3, 3) for _ in range(k)]
    g = g_low + [gk]
    h = [c, b, 1]
    P = compose(g, h)
    # trim/pad defensively to exactly degree 2k (leading coeff of h is monic, so
    # deg(P) is exactly 2k as long as gk != 0, which it is by construction)
    n = 2 * k
    while len(P) < n + 1:
        P.append(0)
    assert len(P) == n + 1 and P[n] != 0
    return P


def main():
    tid = int(sys.argv[1])
    k = SCHEDULE[tid]
    rng = random.Random(20260718 * 1009 + tid * 97 + 3)
    P = build(tid, k, rng)
    n = len(P) - 1
    out = [str(n), " ".join(str(v) for v in P)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
