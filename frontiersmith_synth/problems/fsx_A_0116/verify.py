import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def bareiss_det(M):
    n = len(M)
    M = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = -1
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    sw = i
                    break
            if sw == -1:
                return 0
            M[k], M[sw] = M[sw], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]

def default_fill(n, forced):
    A = [[(1 if j >= i else -1) for j in range(n)] for i in range(n)]
    for (r, c), v in forced.items():
        A[r][c] = v
    return A

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    try:
        it = iter(inp)
        n = int(next(it))
        k = int(next(it))
        forced = {}
        for _ in range(k):
            r = int(next(it)); c = int(next(it)); v = int(next(it))
            forced[(r, c)] = v
    except Exception:
        fail("bad input")

    # ---- parse participant matrix ----
    try:
        vals = [int(x) for x in out]
    except Exception:
        fail("non-integer output")
    if len(vals) != n * n:
        fail("expected %d entries, got %d" % (n * n, len(vals)))
    A = [vals[i * n:(i + 1) * n] for i in range(n)]

    # ---- feasibility: entries are +/-1 and terrain-fixed links respected ----
    for i in range(n):
        for j in range(n):
            if A[i][j] not in (-1, 1):
                fail("entry not +/-1 at (%d,%d)" % (i, j))
    for (r, c), v in forced.items():
        if A[r][c] != v:
            fail("terrain link (%d,%d) violated" % (r, c))

    # ---- objective: exact |det| via Bareiss integer elimination ----
    F = abs(bareiss_det(A))

    # ---- internal baseline: |det| of the terrain-default completion ----
    B = abs(bareiss_det(default_fill(n, forced)))
    B = max(1, B)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
