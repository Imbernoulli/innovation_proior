#!/usr/bin/env python3
"""counter.py <in> <out> <ans>  -- deterministic scorer (format D, eval_form=flops).

Reads the coefficient tensor T from <in> and a candidate BILINEAR SPLIT from <out>.

A candidate split is a list of R scalar-multiplication "terms".  Term t is a rank-1
bilinear product

        m_t = ( sum_i a_t[i] x_i ) * ( sum_j c_t[j] y_j )

and the r output forms are recovered as  b_k = sum_t d_t[k] * m_t.  Equivalently the
split claims

        T[k][i][j] == sum_t d_t[k] * a_t[i] * c_t[j]     for all k,i,j.

The checker FIRST verifies this identity EXACTLY over the rationals (any mismatch,
non-integer/rational token, non-finite value, or schema violation -> Ratio 0.0),
THEN counts the number of scalar products R (each term = one multiplication).  Only
the bilinear products are charged; the linear combinations a.x, c.y and the output
mixing d are free (cheap shifts/adds), exactly as in Karatsuba/Toom accounting.

Objective = minimize R.  Internal baseline B = p*q (the schoolbook split that forms
every product x_i*y_j).  Score:  sc = min(1000, 100 * B / R);  Ratio = sc/1000.
"""
import sys
import re
from fractions import Fraction

TOKEN_RE = re.compile(r"^-?\d+(?:/\d+)?$")   # integer or n/d rational only
MAXABS = 10 ** 7                             # numerator/denominator magnitude bound


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    p = int(next(it)); q = int(next(it)); r = int(next(it))
    T = [[[0] * q for _ in range(p)] for _ in range(r)]
    for k in range(r):
        for i in range(p):
            for j in range(q):
                T[k][i][j] = int(next(it))
    return p, q, r, T


def parse_frac(tok):
    if not TOKEN_RE.match(tok):
        return None
    f = Fraction(tok)
    if abs(f.numerator) > MAXABS or f.denominator > MAXABS:
        return None
    return f


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    p, q, r, T = read_instance(inf)

    cap = 4 * p * q                      # generous upper bound on #terms
    width = p + q + r

    try:
        raw = open(outf).read()
    except Exception:
        fail("cannot read output")

    toks = raw.split()
    if not toks:
        fail("empty output")

    # first token: R (number of terms)
    if not re.match(r"^-?\d+$", toks[0]):
        fail("R is not an integer")
    R = int(toks[0])
    if R < 1 or R > cap:
        fail(f"R={R} out of range [1,{cap}]")

    body = toks[1:]
    if len(body) != R * width:
        fail(f"expected {R*width} coefficient tokens, got {len(body)}")

    terms = []
    idx = 0
    for _ in range(R):
        vals = []
        for _w in range(width):
            f = parse_frac(body[idx]); idx += 1
            if f is None:
                fail("non-integer/rational/oversized/non-finite coefficient")
            vals.append(f)
        a = vals[:p]
        c = vals[p:p + q]
        d = vals[p + q:p + q + r]
        terms.append((a, c, d))

    # exact reconstruction over the rationals
    Tp = [[[Fraction(0)] * q for _ in range(p)] for _ in range(r)]
    for a, c, d in terms:
        for k in range(r):
            dk = d[k]
            if dk == 0:
                continue
            for i in range(p):
                ai = a[i]
                if ai == 0:
                    continue
                aik = dk * ai
                for j in range(q):
                    cj = c[j]
                    if cj != 0:
                        Tp[k][i][j] += aik * cj
    for k in range(r):
        for i in range(p):
            for j in range(q):
                if Tp[k][i][j] != T[k][i][j]:
                    fail(f"identity violated at k={k} i={i} j={j}")

    B = p * q
    sc = min(1000.0, 100.0 * B / max(1, R))
    print(f"terms R={R} baseline_pq={B}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
