# TIER: strong
# The insight: a memoryless/fixed-order NUMERICAL fit (greedy) can't tell
# "residual is tiny" from "residual is exactly zero" -- so it happily accepts
# a low-order approximation that looks fine on the visible window but is
# structurally wrong.  The right move is EXACT algebraic recovery of the
# MINIMAL linear generator:
#   1. Run Berlekamp-Massey over the RATIONALS (exact Fraction arithmetic) on
#      the visible integer sequence.  Because the window is long enough
#      (>= 2*true_order), BM certifies the true minimal-order recurrence --
#      not an approximation, an exact algebraic fact.
#   2. Factor its characteristic polynomial by exact trial division over a
#      bounded range of small integer candidates (the generating function's
#      poles are the sequence's roots), deflating exactly with Fractions.
#   3. Solve the resulting exact Vandermonde system for the (exact, integer)
#      amplitude of every root -- i.e. the closed-form rational-GF partial
#      fraction decomposition, a(n) = sum c_i * r_i^n.
# Emitting this exact closed form costs a few extra terms (parsimony penalty)
# but reproduces the true generator bit-for-bit, so it nails every held-out
# extrapolation, however far past the training window.
import sys
from fractions import Fraction as F


def berlekamp_massey(seq):
    seq = [F(x) for x in seq]
    ls, cur = [], []
    lf, ld = 0, F(0)
    for i in range(len(seq)):
        t = F(0)
        for j in range(len(cur)):
            t += cur[j] * seq[i - 1 - j]
        delta = seq[i] - t
        if delta == 0:
            continue
        if not cur:
            cur = [F(0)] * (i + 1)
            lf, ld = i, delta
            continue
        k = delta / ld
        c = [F(0)] * (i - lf - 1) + [k] + [-k * x for x in ls]
        if len(c) < len(cur):
            c += [F(0)] * (len(cur) - len(c))
        for j in range(len(cur)):
            c[j] += cur[j]
        if i - len(cur) >= lf - len(ls):
            ls, lf, ld = cur, i, delta
        cur = c
    return cur


def find_integer_roots(rec_coeffs, rmax=200):
    k = len(rec_coeffs)
    poly = [F(1)] + [-c for c in rec_coeffs]  # x^k - c0 x^{k-1} - ... - c_{k-1}
    roots = []
    for r in range(2, rmax):
        while True:
            v = F(0)
            for co in poly:
                v = v * r + co
            if v != 0:
                break
            newpoly = [poly[0]]
            for co in poly[1:]:
                newpoly.append(newpoly[-1] * r + co)
            poly = newpoly[:-1]
            roots.append(r)
            if len(poly) == 1:
                break
        if len(poly) == 1:
            break
    return roots, poly


def solve_amplitudes(roots_found, seq):
    k = len(roots_found)
    M = [[F(r) ** j for r in roots_found] + [F(seq[j])] for j in range(k)]
    for col in range(k):
        piv = next(r for r in range(col, k) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(k):
            if r != col and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [M[r][i2] - factor * M[col][i2] for i2 in range(k + 1)]
    return [M[i][k] for i in range(k)]


def fmt_frac(x):
    if x.denominator == 1:
        return str(x.numerator)
    return "(%d/%d)" % (x.numerator, x.denominator)


def main():
    data = sys.stdin.read().split()
    T, t = int(data[0]), int(data[1])
    rows = data[2:]
    seq = [0] * T
    for i in range(T):
        n = int(rows[2 * i])
        v = int(rows[2 * i + 1])
        seq[n] = v

    rec = berlekamp_massey(seq)
    if not rec:
        # degenerate (constant zero) sequence -> trivial fallback
        print("0")
        return
    found_roots, remainder = find_integer_roots(rec)
    if len(remainder) != 1 or not found_roots:
        # fallback: exact factorisation failed (shouldn't happen on-spec);
        # emit the plain recurrence's dominant term via a crude ratio guess
        ratio = seq[-1] / seq[-2] if seq[-2] else 1.0
        A = seq[-1] / (ratio ** (T - 1))
        print("( %r ) * ( %r ) ** n" % (A, ratio))
        return
    amps = solve_amplitudes(found_roots, seq)

    terms = []
    for r, c in zip(found_roots, amps):
        terms.append("( %s ) * ( %d ) ** n" % (fmt_frac(c), r))
    print(" + ".join(terms))


if __name__ == "__main__":
    main()
