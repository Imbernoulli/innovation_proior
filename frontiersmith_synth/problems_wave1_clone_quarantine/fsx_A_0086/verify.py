import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def bareiss_det(M):
    """Exact integer determinant via fraction-free Bareiss elimination."""
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

    raw = open(sys.argv[2]).read().split()
    if len(raw) != n * n:
        fail("expected %d tokens, got %d" % (n * n, len(raw)))

    vals = []
    for t in raw:
        try:
            v = int(t)
        except Exception:
            fail("non-integer token %r" % t)
        if v != 1 and v != -1:
            fail("entry not in {-1,+1}: %r" % t)
        vals.append(v)

    M = [vals[i * n:(i + 1) * n] for i in range(n)]

    # ---- objective: bit-length of exact |det| ----
    d = bareiss_det(M)
    F = abs(d).bit_length()  # 0 if singular

    # ---- internal baseline B: arrow array, |det| = 2^(n-1), so B = n bits ----
    A = [[1] * n for _ in range(n)]
    for i in range(1, n):
        A[i][i] = -1
    B = abs(bareiss_det(A)).bit_length()
    B = max(1, B)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d |det|_bits=%d Ratio: %.6f" % (F, B, F, sc / 1000.0))

if __name__ == "__main__":
    main()
