import sys

# Format D checker -- minimal-count TERNARY rank-1 decomposition of an integer matrix.
#   1) Parse target integer matrix M (n x m) from <in>.
#   2) Parse participant's decomposition from <out>:
#         r
#         then r terms, each = one line of n ints (u_k) followed by one line of m ints (v_k)
#      (whitespace-insensitive; we just read the token stream).
#   3) FEASIBILITY GATE:
#        - every u/v entry must be in the TERNARY alphabet {-1, 0, 1}
#        - sum_k u_k v_k^T must reproduce M EXACTLY (integer arithmetic)
#      any violation / non-integer / non-finite token -> Ratio: 0.0
#   4) Objective (MINIMIZE) = r (number of rank-1 terms).
#      Internal baseline B = row-wise unary construction the checker builds itself:
#        for each row i, need max_j |M[i][j]| ternary terms of the form e_i (x) level-vector,
#        so B = sum_i max_j |M[i][j]|  (a guaranteed feasible upper bound, positive).
#      Ratio = min(1, 0.1 * B / r).

MAX_TERMS = 200000

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        n = int(next(it)); m = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= n <= 2000 and 1 <= m <= 2000):
        fail("bad dims")
    M = [[0] * m for _ in range(n)]
    try:
        for i in range(n):
            for j in range(m):
                M[i][j] = int(next(it))
    except Exception:
        fail("bad matrix")

    # ---- internal baseline (row-wise unary) ----
    B = sum(max((abs(M[i][j]) for j in range(m)), default=0) for i in range(n))
    if B == 0:
        fail("degenerate zero matrix")

    # ---- parse participant output ----
    if not out:
        fail("empty output")
    try:
        r = int(out[0])
    except Exception:
        fail("bad r")
    if r < 1:
        fail("r < 1")
    if r > MAX_TERMS:
        fail("r too large")
    need = 1 + r * (n + m)
    if len(out) != need:
        fail("wrong token count (got %d, need %d)" % (len(out), need))

    # strict integer parse (rejects nan/inf/floats -> non-finite guard)
    toks = out[1:need]
    vals = []
    try:
        for t in toks:
            x = int(t)
            if x not in (-1, 0, 1):
                fail("entry outside ternary alphabet {-1,0,1}")
            vals.append(x)
    except SystemExit:
        raise
    except Exception:
        fail("non-integer / non-finite entry")

    # ---- exact reconstruction ----
    recon = [[0] * m for _ in range(n)]
    p = 0
    for _ in range(r):
        u = vals[p:p + n]; p += n
        v = vals[p:p + m]; p += m
        for i in range(n):
            ui = u[i]
            if ui == 0:
                continue
            row = recon[i]
            for j in range(m):
                vj = v[j]
                if vj:
                    row[j] += ui * vj

    for i in range(n):
        for j in range(m):
            if recon[i][j] != M[i][j]:
                fail("reconstruction mismatch at (%d,%d)" % (i, j))

    ratio = min(1.0, 0.1 * B / r)
    print("r=%d B=%d Ratio: %.6f" % (r, B, ratio))

if __name__ == "__main__":
    main()
