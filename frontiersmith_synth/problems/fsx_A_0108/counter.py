import sys
from fractions import Fraction

# Format D checker -- minimal-mode CP decomposition of an integer thermal-coupling
# tensor H (n x n x n).
#   1) Parse target integer tensor H from <in>  (i outer, j inner rows, k along a line).
#   2) Parse participant's rank-R decomposition from <out>:
#          R
#          then R modes, each  a+b+c = 3n  rational tokens:  u[0..n-1] v[0..n-1] w[0..n-1]
#   3) EXACT-equality gate: sum of modes must reproduce H exactly (rational math).
#      Any parse error, wrong token count, non-finite value, R<1, or mismatch -> Ratio 0.0.
#   4) Objective (minimize) = R.
#      Internal baseline B = number of nonzero mode-3 fibers (i,j) -- the "one separable
#      mode per (injection,production) pair" construction the checker can always build.
#      minimization score:  sc = min(1000, 100 * B / R);  Ratio = sc/1000.
#      trivial (fiber) decomposition -> R == B -> Ratio 0.1.


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read instance")

    it = iter(inp)
    try:
        a = int(next(it)); b = int(next(it)); c = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= a <= 40 and 1 <= b <= 40 and 1 <= c <= 40):
        fail("bad dims")

    H = [[[0] * c for _ in range(b)] for _ in range(a)]
    try:
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    H[i][j][k] = int(next(it))
    except Exception:
        fail("bad tensor")

    # internal baseline: nonzero mode-3 fibers
    B = 0
    for i in range(a):
        for j in range(b):
            if any(H[i][j][k] != 0 for k in range(c)):
                B += 1
    if B == 0:
        fail("degenerate zero tensor")

    # ---- participant output (bounded reads) ----
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
    if not out:
        fail("empty output")
    try:
        R = int(out[0])
    except Exception:
        fail("bad R")
    if R < 1:
        fail("R < 1")
    MAXR = a * b * c  # any tensor has a <=nnz decomposition; more is never useful
    if R > MAXR:
        fail("R too large (> a*b*c)")

    per = a + b + c
    need = 1 + R * per
    if len(out) != need:
        fail("wrong token count (got %d, need %d)" % (len(out), need))

    # Fraction() rejects nan/inf -> exception -> fail (non-finite guard).
    try:
        vals = [Fraction(t) for t in out[1:need]]
    except Exception:
        fail("non-rational / non-finite entry")

    modes = []
    p = 0
    for _ in range(R):
        u = vals[p:p + a]; p += a
        v = vals[p:p + b]; p += b
        w = vals[p:p + c]; p += c
        modes.append((u, v, w))

    # ---- exact reconstruction ----
    recon = [[[Fraction(0)] * c for _ in range(b)] for _ in range(a)]
    for (u, v, w) in modes:
        for i in range(a):
            ui = u[i]
            if ui == 0:
                continue
            for j in range(b):
                uv = ui * v[j]
                if uv == 0:
                    continue
                row = recon[i][j]
                for k in range(c):
                    wk = w[k]
                    if wk != 0:
                        row[k] += uv * wk

    for i in range(a):
        for j in range(b):
            Hij = H[i][j]
            ri = recon[i][j]
            for k in range(c):
                if ri[k] != Hij[k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    sc = min(1000.0, 100.0 * B / max(1e-9, R))
    print("R=%d B=%d Ratio: %.6f" % (R, B, sc / 1000.0))


if __name__ == "__main__":
    main()
