import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def bareiss_det(M):
    """Exact integer determinant via fraction-free Bareiss elimination.
    No floating point anywhere -- the score is a bit-length of an exact big int."""
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
    try:
        n = int(open(sys.argv[1]).read().split()[0])
    except Exception:
        fail("bad input")

    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) != n * n:
        fail("expected %d tokens, got %d" % (n * n, len(raw)))

    vals = []
    for t in raw:
        # Reject non-integers, nan/inf, and out-of-alphabet symbols.
        try:
            v = int(t)
        except Exception:
            fail("non-integer token %r" % t)
        if v != 1 and v != -1:
            fail("signal cell not in {-1,+1}: %r" % t)
        vals.append(v)

    M = [vals[i * n:(i + 1) * n] for i in range(n)]

    # ---- Feasibility: the arterial reference column (column 0) must be all +1.
    # Every intersection is normalized so its coupling to the reference arterial
    # is "green-primary" (+1). Row sign-flips preserve |det|, so this loses no
    # generality -- but a submission that ignores it is rejected.
    for i in range(n):
        if M[i][0] != 1:
            fail("reference column (col 0) must be all +1; row %d has %d" % (i, M[i][0]))

    # ---- Objective: bit-length of the EXACT |det|. The determinant measures how
    # linearly-independent (mutually decorrelated) the N signal schedules are.
    d = bareiss_det(M)
    F = abs(d).bit_length()  # 0 if the plan is degenerate (singular)

    # ---- Internal baseline B: the "corridor" arrow plan. Column 0 is all +1,
    # row 0 is all +1, and cell (i,i) = -1 for i>=1. |det| = 2^(N-1) -- the
    # minimum non-zero determinant, so B = N bits. Trivial reproduces this -> 0.1.
    A = [[1] * n for _ in range(n)]
    for i in range(1, n):
        A[i][i] = -1
    B = abs(bareiss_det(A)).bit_length()
    B = max(1, B)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d |det|_bits=%d Ratio: %.6f" % (F, B, F, sc / 1000.0))

if __name__ == "__main__":
    main()
