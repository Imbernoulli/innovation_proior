import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def bareiss_det(M):
    """Exact integer determinant via fraction-free Bareiss elimination (no floats)."""
    n = len(M)
    M = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = None
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    sw = r
                    break
            if sw is None:
                return 0
            M[k], M[sw] = M[sw], M[k]
            sign = -sign
        for r in range(k + 1, n):
            for c in range(k + 1, n):
                M[r][c] = (M[r][c] * M[k][k] - M[r][k] * M[k][c]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]

def main():
    # ---- parse instance ----
    try:
        tok = open(sys.argv[1]).read().split()
        n = int(tok[0]); nf = int(tok[1])
        fixed = []
        idx = 2
        for _ in range(nf):
            r = int(tok[idx]); c = int(tok[idx + 1]); v = int(tok[idx + 2]); idx += 3
            fixed.append((r, c, v))
    except Exception:
        fail("bad instance")

    # ---- parse participant artifact strictly (reject non-integer / non-finite / out-of-range) ----
    raw = open(sys.argv[2]).read().split()
    if len(raw) != n * n:
        fail("expected %d tokens, got %d" % (n * n, len(raw)))
    vals = []
    for t in raw:
        try:
            x = int(t)                 # rejects 'nan', 'inf', '1.0', etc.
        except Exception:
            fail("non-integer token %r" % t)
        if x != 1 and x != -1:
            fail("entry not in {-1,+1}: %r" % t)
        vals.append(x)
    M = [vals[i * n:(i + 1) * n] for i in range(n)]

    # ---- feasibility: every pre-wired cell must match ----
    for (r, c, v) in fixed:
        if not (0 <= r < n and 0 <= c < n):
            fail("fixed cell out of range")
        if M[r][c] != v:
            fail("pre-wired cell (%d,%d) must be %d, got %d" % (r, c, v, M[r][c]))

    # ---- objective: bit-length of exact |det| ----
    d = bareiss_det(M)
    F = abs(d).bit_length()            # 0 if singular

    # ---- internal baseline B: the minimum non-zero |det| of an NxN +/-1 matrix is
    #      2^(N-1), whose bit-length is exactly N.  B = N. ----
    B = n

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d |det|_bits=%d Ratio: %.6f" % (F, B, F, sc / 1000.0))

if __name__ == "__main__":
    main()
