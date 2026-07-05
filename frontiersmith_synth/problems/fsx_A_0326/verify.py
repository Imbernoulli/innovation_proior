import sys

EXP = 2.0  # size-normalizing geometric exponent (see statement Scoring)

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def bareiss_det(M):
    """Exact integer determinant via Bareiss fraction-free elimination."""
    n = len(M)
    M = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            swap = -1
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    swap = i
                    break
            if swap == -1:
                return 0
            M[k], M[swap] = M[swap], M[k]
            sign = -sign
        akk = M[k][k]
        for i in range(k + 1, n):
            aik = M[i][k]
            Mi = M[i]
            Mk = M[k]
            for j in range(k + 1, n):
                # exact division in Bareiss
                Mi[j] = (Mi[j] * akk - aik * Mk[j]) // prev
        prev = akk
    return sign * M[n - 1][n - 1]

def main():
    try:
        inp = open(sys.argv[1]).read().split()
        N = int(inp[0])
    except Exception:
        fail("bad input")
    if N < 1 or N > 200 or N % 2 == 0:
        fail("bad N")

    # ---- parse participant matrix strictly: exactly N*N tokens, each in {-1,+1} ----
    raw = open(sys.argv[2]).read().split()
    if len(raw) != N * N:
        fail("expected %d tokens, got %d" % (N * N, len(raw)))
    vals = []
    for tok in raw:
        # reject nan/inf/floats/out-of-range: only the literal ints 1 and -1 allowed
        if tok not in ("1", "-1", "+1"):
            fail("illegal entry %r (must be 1 or -1)" % tok)
        vals.append(1 if tok in ("1", "+1") else -1)
    M = [vals[r * N:(r + 1) * N] for r in range(N)]

    # ---- objective: exact |det| ----
    detM = abs(bareiss_det(M))

    # ---- internal reference D0: trivial design M0[i][j] = +1 if i==j else -1 ----
    M0 = [[1 if i == j else -1 for j in range(N)] for i in range(N)]
    D0 = abs(bareiss_det(M0))
    if D0 <= 0:
        D0 = 1

    if detM <= 0:
        # singular / degenerate design -> no information volume
        print("Ratio: 0.000000 (singular det=0)")
        sys.exit(0)

    F = float(detM) ** (EXP / N)
    B = float(D0) ** (EXP / N)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("|det|=%d D0=%d  Ratio: %.6f" % (detM, D0, sc / 1000.0))

if __name__ == "__main__":
    main()
