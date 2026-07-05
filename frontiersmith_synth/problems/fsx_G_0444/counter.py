import sys
from fractions import Fraction

# Format D checker -- minimal-multiplier bilinear algorithm for a short-convolution
# accelerator (Winograd-style rank of the structure tensor T, shape s x p x q).
#
#   1) Parse target integer tensor T from <in>:
#         header  p q s
#         then p*q lines; line (i*q + j) = [ T[0][i][j] ... T[s-1][i][j] ].
#   2) Parse participant's rank-R bilinear algorithm from <out>:
#         R
#         R terms, each  p + q + s  rationals:  u[0..p-1] v[0..q-1] w[0..s-1]
#      Term r realises one scalar multiplier  m_r = (u_r . a) * (v_r . b)  whose
#      product is distributed to the outputs by w_r.
#   3) EXACT-equality gate: sum_r w_r[k] u_r[i] v_r[j] must reproduce T exactly
#      (rational arithmetic) at every (k,i,j).  Any mismatch -> Ratio 0.
#   4) Objective (minimize) = R = number of scalar multipliers.
#      Baseline B = number of product fibers a[i]*b[j] that are actually used
#      (nonzero for some output tap) -- the naive "one multiplier per used product"
#      algorithm always achieves rank B.   Ratio = min(1, 0.1 * B / R).

MAXR = 100000

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        p = int(next(it)); q = int(next(it)); s = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= p <= 12 and 1 <= q <= 12 and 1 <= s <= 8):
        fail("bad dims")

    # T[k][i][j]
    T = [[[0] * q for _ in range(p)] for _ in range(s)]
    try:
        for i in range(p):
            for j in range(q):
                for k in range(s):
                    T[k][i][j] = int(next(it))
    except Exception:
        fail("bad tensor")

    # B = number of used product fibers (i,j) : nonzero for some output tap k.
    B = 0
    for i in range(p):
        for j in range(q):
            if any(T[k][i][j] != 0 for k in range(s)):
                B += 1
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
    per = p + q + s
    need = 1 + R * per
    if len(out) != need:
        fail("wrong token count (got %d, need %d)" % (len(out), need))

    toks = out[1:need]
    # Fraction() rejects nan / inf / 1e3 -> exception -> fail (non-finite guard).
    try:
        vals = [Fraction(t) for t in toks]
    except Exception:
        fail("non-rational / non-finite entry")

    terms = []
    idx = 0
    for _ in range(R):
        u = vals[idx:idx + p]; idx += p
        v = vals[idx:idx + q]; idx += q
        w = vals[idx:idx + s]; idx += s
        terms.append((u, v, w))

    # ---- exact reconstruction: recon[k][i][j] = sum_r w_r[k] u_r[i] v_r[j] ----
    recon = [[[Fraction(0)] * q for _ in range(p)] for _ in range(s)]
    for (u, v, w) in terms:
        for i in range(p):
            ui = u[i]
            if ui == 0:
                continue
            for j in range(q):
                uv = ui * v[j]
                if uv == 0:
                    continue
                for k in range(s):
                    wk = w[k]
                    if wk != 0:
                        recon[k][i][j] += wk * uv

    for k in range(s):
        for i in range(p):
            for j in range(q):
                if recon[k][i][j] != T[k][i][j]:
                    fail("reconstruction mismatch at (k=%d,i=%d,j=%d)" % (k, i, j))

    ratio = min(1.0, 0.1 * B / R)
    print("R=%d B=%d Ratio: %.6f" % (R, B, ratio))

if __name__ == "__main__":
    main()
