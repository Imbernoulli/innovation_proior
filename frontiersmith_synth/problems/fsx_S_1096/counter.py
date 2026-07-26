import sys

# Format D checker -- "Gauss-period basis sparsity" (max-lane structure-constant load).
#
#   1) Parse (p, k, f) from <in>: f is a monic irreducible degree-k poly over F_p
#      defining the field F_p[x]/(f(x)).
#   2) Parse the participant's k x k matrix M from <out> (row i = coordinates, in
#      the monomial basis 1,x,...,x^{k-1}, of a candidate NEW basis vector b_i).
#   3) EXACT feasibility gate: M must be invertible mod p (that is precisely the
#      condition for {b_i} to be a basis of the k-dim F_p-vector space F_p[x]/(f)) ;
#      any parse error / wrong shape / out-of-range entry / singular M -> Ratio 0.0.
#   4) Objective (minimize): build the exact structure-constant tensor
#         b_i * b_j = sum_l c[i][j][l] * b_l   (field multiplication, reduced mod f)
#      For each output coordinate l ("lane"), let w_l = #{(i,j): 0<=i<=j<k,
#      c[i][j][l] != 0}  (how many cross terms a lane must accumulate if all k
#      output lanes are evaluated in parallel). F = max_l w_l  (the busiest lane
#      bounds the whole circuit's latency).  Baseline B = the SAME quantity for
#      the checker's own trivial construction: the monomial basis itself (M = I).
#      Ratio = min(1, 0.1*B/F).
#
# This directly rewards the classical fact that an OPTIMAL Gauss-period normal
# basis loads every lane EQUALLY (Frobenius-shift symmetry), while a basis chosen
# without exploiting that symmetry (monomial or an arbitrary normal basis) has an
# uneven load whose worst lane bottlenecks the whole circuit.

MAXTOK = 5_000_000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def reduce_full(raw, f, p, k):
    a = raw[:]
    for deg in range(len(a) - 1, k - 1, -1):
        c = a[deg]
        if c:
            shift = deg - k
            for i in range(k):
                a[shift + i] = (a[shift + i] - c * f[i]) % p
            a[deg] = 0
    return a[:k]


def pmulmod(a, b, f, p, k):
    raw = [0] * (2 * k - 1)
    for i in range(k):
        ai = a[i]
        if not ai:
            continue
        for j in range(k):
            bj = b[j]
            if bj:
                raw[i + j] = (raw[i + j] + ai * bj) % p
    return reduce_full(raw, f, p, k)


def mat_inv_mod_p(M, p, k):
    A = [row[:] + [1 if i == j else 0 for j in range(k)] for i, row in enumerate(M)]
    for col in range(k):
        piv = None
        for r in range(col, k):
            if A[r][col] % p != 0:
                piv = r
                break
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        inv = pow(A[col][col], p - 2, p)
        A[col] = [(x * inv) % p for x in A[col]]
        for r in range(k):
            if r != col and A[r][col] % p != 0:
                c = A[r][col]
                A[r] = [(A[r][j] - c * A[col][j]) % p for j in range(2 * k)]
    return [row[k:] for row in A]


def max_lane(M, f, p, k):
    Minv = mat_inv_mod_p(M, p, k)
    if Minv is None:
        return None
    per_l = [0] * k
    for i in range(k):
        for j in range(i, k):
            prod = pmulmod(M[i], M[j], f, p, k)
            c = [sum(prod[t] * Minv[t][l] for t in range(k)) % p for l in range(k)]
            for l in range(k):
                if c[l] != 0:
                    per_l[l] += 1
    return max(per_l)


def main():
    in_toks = open(sys.argv[1]).read().split()
    out_text = open(sys.argv[2]).read()

    it = iter(in_toks)
    try:
        p = int(next(it)); k = int(next(it))
    except Exception:
        fail("bad header")
    if not (2 <= p <= 100000 and 1 <= k <= 64):
        fail("bad p,k")
    try:
        f = [int(next(it)) for _ in range(k)]
    except Exception:
        fail("bad f")
    if any(c < 0 or c >= p for c in f):
        fail("f coeff out of range")

    # ---- parse participant output: exactly k*k tokens, ints in [0,p-1] ----
    out_toks = out_text.split()
    if len(out_toks) > MAXTOK:
        fail("output too large")
    if len(out_toks) != k * k:
        fail("wrong token count (got %d, need %d)" % (len(out_toks), k * k))
    vals = []
    for t in out_toks:
        # reject non-finite / non-integer tokens (nan, inf, 1e3, fractions, ...)
        if not (t.lstrip("-").isdigit()):
            fail("non-integer token %r" % t)
        v = int(t)
        if v < 0 or v >= p:
            fail("entry out of range [0,%d): %d" % (p, v))
        vals.append(v)
    M = [vals[i * k:(i + 1) * k] for i in range(k)]

    F = max_lane(M, f, p, k)
    if F is None:
        fail("submitted matrix is singular mod p (not a valid basis)")
    if F <= 0:
        fail("degenerate objective")

    B = max_lane([[1 if i == j else 0 for j in range(k)] for i in range(k)], f, p, k)
    if B is None or B <= 0:
        fail("internal baseline degenerate (author error)")

    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("p=%d k=%d B=%d F=%d Ratio: %.6f" % (p, k, B, F, ratio))


if __name__ == "__main__":
    main()
