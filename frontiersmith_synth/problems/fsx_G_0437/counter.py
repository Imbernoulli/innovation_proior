import sys
from fractions import Fraction

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_tensor(path):
    tok = open(path).read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for k in range(c):
        for i in range(a):
            for j in range(b):
                T[i][j][k] = int(next(it))
    return a, b, c, T

def parse_frac(s):
    # strict: reject nan/inf and anything Fraction can't parse exactly
    low = s.lower()
    if low in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
        raise ValueError("nonfinite")
    if "e" in low or "E" in s:  # no scientific notation (would hide inf/overflow)
        raise ValueError("scientific")
    return Fraction(s)

def main():
    a, b, c, T = read_tensor(sys.argv[1])

    # ---- internal baseline B: trivial mode-c flattening = # nonzero fibres ----
    B = 0
    for i in range(a):
        for j in range(b):
            if any(T[i][j][k] != 0 for k in range(c)):
                B += 1
    B = max(1, B)

    # ---- parse participant decomposition ----
    tok = open(sys.argv[2]).read().split()
    if not tok:
        fail("empty")
    try:
        R = int(tok[0])
    except Exception:
        fail("bad R")
    if R < 1:
        fail("R<1")
    MAXR = 4 * a * b * c + 16
    if R > MAXR:
        fail("R too large")
    need = R * (a + b + c)
    body = tok[1:]
    if len(body) != need:
        fail("token count mismatch (got %d need %d)" % (len(body), need))

    terms = []
    idx = 0
    try:
        for _ in range(R):
            u = [parse_frac(body[idx + t]) for t in range(a)]; idx += a
            v = [parse_frac(body[idx + t]) for t in range(b)]; idx += b
            w = [parse_frac(body[idx + t]) for t in range(c)]; idx += c
            terms.append((u, v, w))
    except Exception as e:
        fail("bad number (%s)" % e)

    # ---- exact reconstruction check ----
    # accumulate per (i,j): sum_r u[i]*v[j] * w[k]  ==  T[i][j][k]
    for i in range(a):
        for j in range(b):
            coef = [Fraction(0)] * c
            for (u, v, w) in terms:
                uv = u[i] * v[j]
                if uv == 0:
                    continue
                for k in range(c):
                    if w[k] != 0:
                        coef[k] += uv * w[k]
            for k in range(c):
                if coef[k] != T[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    F = R  # number of scalar multiplications
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
