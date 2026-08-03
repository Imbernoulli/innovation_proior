import sys
from fractions import Fraction

# Format D checker for the tide-pool interaction-tensor decomposition problem.
# 1) Parse the target integer tensor T (shape a x b x c) from <in>.
# 2) Parse the participant's rank-R channel list from <out>:
#       R
#       R lines, each with (a + b + c) rationals: u[0..a-1] v[0..b-1] w[0..c-1]
#    Each channel r contributes u_r (x) v_r (x) w_r to the reconstruction.
# 3) EXACT-equality gate: sum of channels must reproduce T exactly (rational math).
#    Any violation -> "Ratio: 0.0".
# 4) Objective (minimize) = R (number of interaction channels / scalar multiplies).
#    Baseline B = number of nonzero tensor entries (the naive one-channel-per-entry
#    construction). ratio = min(1, 0.1 * B / R).

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
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    try:
        for k in range(c):
            for i in range(a):
                for j in range(b):
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
    per = a + b + c
    cap = 10 * a * b * c + 10
    if R > cap:
        fail("R too large")
    need = 1 + R * per
    if len(out) < need:
        fail("not enough channel numbers")

    toks = out[1:need]
    try:
        vals = [Fraction(t) for t in toks]
    except Exception:
        fail("non-rational entry")

    channels = []
    p = 0
    for _ in range(R):
        u = vals[p:p + a]; p += a
        v = vals[p:p + b]; p += b
        w = vals[p:p + c]; p += c
        channels.append((u, v, w))

    # ---- exact reconstruction ----
    recon = [[[Fraction(0) for _ in range(c)] for _ in range(b)] for _ in range(a)]
    for (u, v, w) in channels:
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
                        recon[i][j][k] += uv * w[k]

    for i in range(a):
        for j in range(b):
            for k in range(c):
                if recon[i][j][k] != T[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    ratio = min(1.0, 0.1 * B / R)
    print("R=%d B=%d Ratio: %.6f" % (R, B, ratio))

if __name__ == "__main__":
    main()
