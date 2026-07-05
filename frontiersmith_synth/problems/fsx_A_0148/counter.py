#!/usr/bin/env python3
"""counter.py <in> <out> <ans>   (Format D, eval_form=flops)

Verifies EXACT reconstruction of the festival vibe tensor T from a submitted
CP decomposition (a list of rank-one 'acts'), then scores by the number of
acts R (fewer is better).

Submission format (participant stdout), token stream:
    line 1:  R                              (number of rank-one acts, >=1)
    then R rows, each with a+b+c numbers:
             u[0..a-1]  v[0..b-1]  w[0..c-1]
    reconstructed  That[i,j,k] = sum_r u_r[i]*v_r[j]*w_r[k]  must EXACTLY equal T.
    entries may be integers or rationals written as p/q (also plain decimals).

Feasibility failures (bad schema, non-finite, wrong reconstruction, R out of
range) -> "Ratio: 0.0".  Baseline B = a*b (the canonical mode-3 slab rank).
Score: ratio = min(1.0, 0.1 * B / R).
"""
import sys
from fractions import Fraction as F

MAX_TERMS = 5000


def fail(msg):
    print("reason:", msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints(path):
    with open(path) as fh:
        toks = fh.read().split()
    return toks


def parse_frac(tok):
    # rejects nan/inf (Fraction raises), accepts int / p/q / decimal
    return F(tok)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    itoks = read_ints(in_path)
    try:
        a = int(itoks[0]); b = int(itoks[1]); c = int(itoks[2])
    except Exception:
        fail("bad instance header")
    need = a * b * c
    body = itoks[3:]
    if len(body) < need:
        fail("truncated instance")
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    idx = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                T[i][j][k] = int(body[idx]); idx += 1

    # ---- read submission ----
    otoks = read_ints(out_path)
    if len(otoks) == 0:
        fail("empty output")
    try:
        R = int(otoks[0])
    except Exception:
        fail("R not an integer")
    if R < 1:
        fail("R < 1")
    if R > MAX_TERMS:
        fail("R exceeds MAX_TERMS")

    per = a + b + c
    rest = otoks[1:]
    if len(rest) != R * per:
        fail("expected %d term-tokens, got %d" % (R * per, len(rest)))

    terms = []
    p = 0
    for _ in range(R):
        try:
            vals = [parse_frac(rest[p + t]) for t in range(per)]
        except Exception:
            fail("unparseable / non-finite token")
        p += per
        u = vals[0:a]
        v = vals[a:a + b]
        w = vals[a + b:a + b + c]
        terms.append((u, v, w))

    # ---- exact reconstruction & equality ----
    That = [[[F(0) for _ in range(c)] for _ in range(b)] for _ in range(a)]
    for (u, v, w) in terms:
        for i in range(a):
            ui = u[i]
            if ui == 0:
                continue
            for j in range(b):
                uv = ui * v[j]
                if uv == 0:
                    continue
                for k in range(c):
                    if w[k] != 0:
                        That[i][j][k] += uv * w[k]

    for i in range(a):
        for j in range(b):
            for k in range(c):
                if That[i][j][k] != T[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    # ---- score (minimize R) ----
    B = a * b  # canonical mode-3 slab rank
    sc = min(1000.0, 100.0 * B / max(1e-9, R))
    print("R=%d baseline=%d" % (R, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
