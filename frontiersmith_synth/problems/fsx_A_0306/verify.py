import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

# ---- exact integer determinant magnitude via Bareiss elimination ----
def bareiss_absdet(M):
    n = len(M)
    A = [row[:] for row in M]
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            piv = None
            for r in range(k + 1, n):
                if A[r][k] != 0:
                    piv = r
                    break
            if piv is None:
                return 0
            A[k], A[piv] = A[piv], A[k]
        akk = A[k][k]
        for i in range(k + 1, n):
            aik = A[i][k]
            Ai = A[i]
            Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return abs(A[n - 1][n - 1])

def main():
    # ---- read instance ----
    try:
        inp = open(sys.argv[1]).read().split()
        it = iter(inp)
        N = int(next(it))
        K = int(next(it))
        fixed = {}
        for _ in range(K):
            r = int(next(it)); c = int(next(it)); v = int(next(it))
            fixed[(r, c)] = v
    except Exception:
        fail("bad instance")

    # ---- internal baseline B: the triangular reference completion (guaranteed non-singular) ----
    base = [[fixed.get((i, j), 1 if i >= j else -1) for j in range(N)] for i in range(N)]
    B = bareiss_absdet(base)
    if B <= 0:
        fail("degenerate baseline")

    # ---- read participant matrix (bounded) ----
    try:
        raw = open(sys.argv[2]).read(20000000).split()
    except Exception:
        fail("no output")
    if len(raw) != N * N:
        fail("expected %d entries, got %d" % (N * N, len(raw)))

    M = []
    idx = 0
    for i in range(N):
        row = []
        for j in range(N):
            tok = raw[idx]; idx += 1
            try:
                x = int(tok)
            except Exception:
                fail("non-integer entry %r" % tok)
            if x != 1 and x != -1:
                fail("entry not in {-1,+1}: %r" % tok)
            if (i, j) in fixed and x != fixed[(i, j)]:
                fail("violates pre-committed route at (%d,%d)" % (i, j))
            row.append(x)
        M.append(row)

    # ---- objective: exact |det|, scored by per-dimension magnitude |det|^(1/N) ----
    F_det = bareiss_absdet(M)
    if F_det <= 0:
        print("det=0 Ratio: 0.0 (singular ledger)")
        sys.exit(0)

    F = float(F_det) ** (1.0 / N)
    Bp = float(B) ** (1.0 / N)
    sc = min(1000.0, 100.0 * F / max(1e-9, Bp))
    print("det=%d B=%d Ratio: %.6f" % (F_det, B, sc / 1000.0))

if __name__ == "__main__":
    main()
