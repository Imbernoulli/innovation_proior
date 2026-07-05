import sys
from fractions import Fraction


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def parse_instance(tok):
    idx = 0
    n1 = int(tok[idx]); n2 = int(tok[idx + 1]); n3 = int(tok[idx + 2]); idx += 3
    T = [[[0] * n3 for _ in range(n2)] for _ in range(n1)]
    for k in range(n3):
        for i in range(n1):
            for j in range(n2):
                T[i][j][k] = int(tok[idx]); idx += 1
    return n1, n2, n3, T


def fail(msg):
    print("REJECT: " + msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inp, out = sys.argv[1], sys.argv[2]
    itok = read_tokens(inp)
    n1, n2, n3, T = parse_instance(itok)

    # Internal trivial baseline B: one bilinear multiplication per (i,j) pair that
    # is nonzero in some output slice k (a valid rank-<support> bilinear algorithm).
    B = 0
    for i in range(n1):
        for j in range(n2):
            if any(T[i][j][k] != 0 for k in range(n3)):
                B += 1
    if B == 0:
        B = 1

    otok = read_tokens(out)
    if len(otok) == 0:
        fail("empty output")
    try:
        R = int(otok[0])
    except Exception:
        fail("first token is not an integer R")
    MAXR = 3 * n1 * n2 * n3 + 100
    if R < 1 or R > MAXR:
        fail("R out of range [1,%d]" % MAXR)

    per = n1 + n2 + n3
    body = otok[1:]
    if len(body) != R * per:
        fail("expected %d rational tokens after R, got %d" % (R * per, len(body)))

    us, vs, ws = [], [], []
    p = 0
    for _ in range(R):
        try:
            u = [Fraction(body[p + t]) for t in range(n1)]; p += n1
            v = [Fraction(body[p + t]) for t in range(n2)]; p += n2
            w = [Fraction(body[p + t]) for t in range(n3)]; p += n3
        except Exception:
            fail("non-rational / non-finite token (nan/inf/float-exp not allowed)")
        us.append(u); vs.append(v); ws.append(w)

    # Verify the bilinear algorithm computes T EXACTLY:
    #   sum_r u_r[i] * v_r[j] * w_r[k] == T[i][j][k]   for all i,j,k.
    rec = [[[Fraction(0) for _ in range(n3)] for _ in range(n2)] for _ in range(n1)]
    for r in range(R):
        u, v, w = us[r], vs[r], ws[r]
        ui = [(i, u[i]) for i in range(n1) if u[i] != 0]
        vj = [(j, v[j]) for j in range(n2) if v[j] != 0]
        wk = [(k, w[k]) for k in range(n3) if w[k] != 0]
        for i, a in ui:
            reci = rec[i]
            for j, b in vj:
                ab = a * b
                rr = reci[j]
                for k, c in wk:
                    rr[k] += ab * c
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if rec[i][j][k] != T[i][j][k]:
                    fail("bilinear algorithm does not equal target tensor at (%d,%d,%d)" % (i, j, k))

    sc = min(1000.0, 100.0 * B / max(1e-9, float(R)))
    print("baseline_support=%d multiplications=%d" % (B, R))
    print("Ratio: %.6f" % (sc / 1000.0))


main()
