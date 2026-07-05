import sys
from fractions import Fraction

# Format D checker -- minimal-multiplier CP (rank) decomposition of an integer
# traffic-signal response tensor.
#
#   1) Parse target integer tensor T (a x b x c) from <in>  (i outer, then j rows
#      of c phase values each).
#   2) Parse the participant's rank-R stage list from <out>:
#         R
#         then R stages, each a+b+c rationals:  u[0..a-1] v[0..b-1] w[0..c-1]
#      (rationals: integers, "p/q", or plain decimals; nan/inf/scientific rejected).
#   3) EXACT-equality gate: sum_r u_r (x) v_r (x) w_r must reproduce T exactly,
#      using rational arithmetic (wrong -> Ratio: 0.0).
#   4) Objective (minimize) = R = number of shared multiplier units.
#      Internal baseline B = number of nonzero entries (the naive one-multiply-
#      per-nonzero table).  Ratio = min(1, 0.1 * B / R).

MAXR = 5000          # cap participant stage count (bounded compute)
MAXMAG = 10 ** 9     # cap |numerator|,|denominator| of any factor


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_rational(tok):
    # Strict rational parse; rejects nan / inf / scientific notation.
    low = tok.lower()
    if ("inf" in low) or ("nan" in low) or ("e" in low) or ("x" in low):
        raise ValueError("non-finite/scientific token")
    f = Fraction(tok)  # accepts ints, p/q, and plain decimals like -0.5
    if abs(f.numerator) > MAXMAG or abs(f.denominator) > MAXMAG:
        raise ValueError("factor magnitude too large")
    return f


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    # ---- parse target tensor ----
    it = iter(inp)
    try:
        a = int(next(it)); b = int(next(it)); c = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= a <= 5 and 1 <= b <= 5 and 1 <= c <= 5):
        fail("bad dims")
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    try:
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    T[i][j][k] = int(next(it))
    except Exception:
        fail("bad tensor body")

    B = sum(1 for i in range(a) for j in range(b) for k in range(c) if T[i][j][k] != 0)
    if B == 0:
        fail("degenerate zero tensor")

    # ---- parse participant output ----
    if not out:
        fail("empty output")
    if len(out) > 2 + MAXR * (a + b + c):
        fail("output too large")
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

    try:
        vals = [parse_rational(t) for t in out[1:need]]
    except Exception as e:
        fail("bad factor token (%s)" % e)

    U = [[Fraction(0)] * a for _ in range(R)]
    V = [[Fraction(0)] * b for _ in range(R)]
    W = [[Fraction(0)] * c for _ in range(R)]
    p = 0
    for r in range(R):
        for i in range(a):
            U[r][i] = vals[p]; p += 1
        for j in range(b):
            V[r][j] = vals[p]; p += 1
        for k in range(c):
            W[r][k] = vals[p]; p += 1

    # ---- exact reconstruction gate ----
    for i in range(a):
        for j in range(b):
            for k in range(c):
                s = Fraction(0)
                for r in range(R):
                    s += U[r][i] * V[r][j] * W[r][k]
                if s != T[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    # ---- score (minimize R) ----
    F = R
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("ops_yours=%d baseline_nnz=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
