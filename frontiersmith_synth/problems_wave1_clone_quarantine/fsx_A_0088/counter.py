import sys
from fractions import Fraction

# Format D checker -- minimal-multiplier CP decomposition of an integer gain tensor.
#   1) Parse target integer tensor G (a x b x c) from <in>  (i outer, j inner rows).
#   2) Parse participant's rank-R stage list from <out>:
#         R
#         R stages, each a+b+c rationals:  u[0..a-1] v[0..b-1] w[0..c-1]
#   3) EXACT-equality gate: sum of stages must reproduce G exactly (rational math).
#   4) Objective (minimize) = R.  Baseline B = # nonzero entries (naive per-entry rank).
#      Ratio = min(1, 0.1 * B / R).

MAXR = 1000

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
    if not (1 <= a <= 5 and 1 <= b <= 5 and 1 <= c <= 5):
        fail("bad dims")
    G = [[[0] * c for _ in range(b)] for _ in range(a)]
    try:
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    G[i][j][k] = int(next(it))
    except Exception:
        fail("bad tensor")

    B = sum(1 for i in range(a) for j in range(b) for k in range(c) if G[i][j][k] != 0)
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
    # Fraction() rejects nan/inf/1e3 -> exception -> fail (non-finite guard).
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

    # ---- exact reconstruction ----
    recon = [[[Fraction(0)] * c for _ in range(b)] for _ in range(a)]
    for (u, v, w) in stages:
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
                    if w[k] != 0:
                        row[k] += uv * w[k]

    for i in range(a):
        for j in range(b):
            for k in range(c):
                if recon[i][j][k] != G[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    ratio = min(1.0, 0.1 * B / R)
    print("R=%d B=%d Ratio: %.6f" % (R, B, ratio))

if __name__ == "__main__":
    main()
