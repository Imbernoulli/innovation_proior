import sys
from fractions import Fraction

# Format D checker -- minimum-multiplier CP decomposition of an integer crosstalk tensor.
#
#   1) Parse the target integer tensor T (a x b x c) from <in>  (i outer, j middle rows).
#   2) Parse the participant's rank-R stage list from <out>:
#         R
#         then R stages, each  a + b + c  rationals:  u[0..a-1] v[0..b-1] w[0..c-1]
#      Each stage is the separable rank-1 term  u (x) v (x) w.
#   3) EXACT-EQUALITY gate: the sum of the R stages must reproduce T exactly using
#      rational arithmetic (any mismatch, wrong shape, non-finite or non-rational
#      entry  ->  Ratio 0.0).
#   4) Objective (minimise) = R = number of scalar multipliers.
#      Internal baseline B = number of nonzero coefficients (one multiplier each).
#      Ratio = min(1, 0.1 * B / R).   trivial (per-entry) -> 0.1 ; 10x fewer -> 1.0.

MAXR = 20000
DMAX = 12


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        a = int(next(it)); b = int(next(it)); c = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= a <= DMAX and 1 <= b <= DMAX and 1 <= c <= DMAX):
        fail("bad dims")

    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    try:
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    T[i][j][k] = int(next(it))
    except Exception:
        fail("bad tensor")

    B = sum(1 for i in range(a) for j in range(b) for k in range(c) if T[i][j][k] != 0)
    if B == 0:
        fail("degenerate zero tensor")

    # ---- parse participant output ----
    if not out:
        fail("empty output")
    try:
        R = int(out[0])
    except Exception:
        fail("bad R")
    if R < 1:
        fail("R < 1")
    if R > MAXR:
        fail("R too large")

    per = a + b + c
    need = 1 + R * per
    if len(out) != need:
        fail("wrong token count (got %d, need %d)" % (len(out), need))

    toks = out[1:need]
    # Fraction() rejects nan / inf / 1e9-style floats -> exception -> fail (non-finite guard).
    try:
        vals = [Fraction(t) for t in toks]
    except Exception:
        fail("non-rational / non-finite entry")

    stages = []
    p = 0
    for _ in range(R):
        u = vals[p:p + a]; p += a
        v = vals[p:p + b]; p += b
        w = vals[p:p + c]; p += c
        stages.append((u, v, w))

    # ---- exact reconstruction (sparse over nonzero factor entries) ----
    recon = [[[Fraction(0)] * c for _ in range(b)] for _ in range(a)]
    for (u, v, w) in stages:
        nzk = [(k, w[k]) for k in range(c) if w[k] != 0]
        for i in range(a):
            ui = u[i]
            if ui == 0:
                continue
            for j in range(b):
                uv = ui * v[j]
                if uv == 0:
                    continue
                row = recon[i][j]
                for k, wk in nzk:
                    row[k] += uv * wk

    for i in range(a):
        for j in range(b):
            for k in range(c):
                if recon[i][j][k] != T[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    ratio = min(1.0, 0.1 * B / R)
    print("R=%d B=%d Ratio: %.6f" % (R, B, ratio))


if __name__ == "__main__":
    main()
