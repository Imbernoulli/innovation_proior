import sys
from fractions import Fraction

# Format D checker -- minimal-multiplication CP decomposition of the integer
# quay-transfer tensor T (I x J x K).
#
#   1) Parse the target tensor T from <in>  (order i, then j, then k).
#   2) Parse the participant's rank-R crane program from <out>:
#          R
#          then R stages, each I+J+K rationals:  u[0..I-1] v[0..J-1] w[0..K-1]
#      A stage contributes the separable term  u (x) v (x) w.
#   3) EXACT-equality gate: the sum of the R stages must reproduce T exactly,
#      using rational arithmetic.  Any mismatch / malformed / non-finite output
#      scores Ratio: 0.0.
#   4) Objective (MINIMIZE) = R = number of scalar multiplications the program
#      needs to evaluate the bilinear map.  Baseline B = number of nonzero
#      entries of T (the naive one-multiplication-per-entry program).
#      Ratio = min(1, 0.1 * B / R)  -> a 10x-shorter program caps at 1.0.

MAXR = 20000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    it = iter(inp)
    try:
        I = int(next(it)); J = int(next(it)); K = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= I <= 5 and 1 <= J <= 5 and 1 <= K <= 5):
        fail("bad dims")

    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    try:
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    T[i][j][k] = int(next(it))
    except Exception:
        fail("bad tensor entries")

    B = sum(1 for i in range(I) for j in range(J) for k in range(K) if T[i][j][k] != 0)
    if B == 0:
        fail("degenerate zero tensor")

    # ---- parse participant output ----
    if not out:
        fail("empty output")
    try:
        R = int(out[0])
    except Exception:
        fail("bad R token")
    if R < 1:
        fail("R < 1")
    if R > MAXR:
        fail("R too large")

    per = I + J + K
    need = 1 + R * per
    if len(out) != need:
        fail("wrong token count (got %d, need %d)" % (len(out), need))

    toks = out[1:need]
    # Fraction() rejects nan / inf / 1e5 -> exception -> fail (non-finite guard).
    try:
        vals = [Fraction(t) for t in toks]
    except Exception:
        fail("non-rational / non-finite entry")

    stages = []
    p = 0
    for _ in range(R):
        u = vals[p:p + I]; p += I
        v = vals[p:p + J]; p += J
        w = vals[p:p + K]; p += K
        stages.append((u, v, w))

    # ---- exact reconstruction ----
    recon = [[[Fraction(0)] * K for _ in range(J)] for _ in range(I)]
    for (u, v, w) in stages:
        for i in range(I):
            ui = u[i]
            if ui == 0:
                continue
            for j in range(J):
                uv = ui * v[j]
                if uv == 0:
                    continue
                row = recon[i][j]
                for k in range(K):
                    wk = w[k]
                    if wk != 0:
                        row[k] += uv * wk

    for i in range(I):
        for j in range(J):
            for k in range(K):
                if recon[i][j][k] != T[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    ratio = min(1.0, 0.1 * B / R)
    print("R=%d B=%d Ratio: %.6f" % (R, B, ratio))


if __name__ == "__main__":
    main()
