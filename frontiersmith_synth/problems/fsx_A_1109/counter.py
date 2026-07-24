import sys
import math

# Format D checker -- metered Jacobi polishing ledger.
#
#   1) Read instance from <in>:  n B m / n diagonal integers / m triples (i j v), 1<=i<j<=n.
#   2) Read participant schedule from <out>:  k, then k pairs (i j), 1-indexed, i != j.
#      Strict schema: exactly 1 + 2k integer tokens, 0 <= k <= 2000.  Any violation,
#      non-integer or out-of-range token -> Ratio: 0.0.
#   3) Simulate: pivots are executed in order.  Pivot (i,j) costs nnz(row i) + nnz(row j)
#      (counting diagonal) at the CURRENT state.  The first pivot whose cost would push
#      total spend over B, and everything after it, is ignored.  Each executed pivot
#      applies the exact Jacobi rotation that zeroes A[i][j] (stable tan formula below),
#      in IEEE double precision with a fixed loop order -- bit-for-bit reproducible.
#   4) Objective (minimize): E = sum_{i<j} A[i][j]^2 after the budget is spent.
#      Internal baseline B0 = E of the untouched matrix (the do-nothing schedule).
#      ratio = min(1, 0.1 * B0 / max(E, 1e-9));  do-nothing -> exactly 0.1.


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def rotate(A, i, j):
    # Exact Jacobi rotation zeroing A[i][j]; A symmetric, updated in place.
    p = A[i][j]
    if p == 0.0:
        return
    a = A[i][i]
    d = A[j][j]
    tau = (d - a) / (2.0 * p)
    if tau == 0.0:
        t = 1.0
    elif tau > 0.0:
        t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
    else:
        t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
    c = 1.0 / math.sqrt(1.0 + t * t)
    s = t * c
    n = len(A)
    Ai = A[i]
    Aj = A[j]
    for k in range(n):
        if k == i or k == j:
            continue
        aik = Ai[k]
        ajk = Aj[k]
        if aik == 0.0 and ajk == 0.0:
            continue
        ni = c * aik - s * ajk
        nj = s * aik + c * ajk
        Ai[k] = A[k][i] = ni
        Aj[k] = A[k][j] = nj
    A[i][i] = a - t * p
    A[j][j] = d + t * p
    Ai[j] = 0.0
    Aj[i] = 0.0


def nnz(row):
    c = 0
    for v in row:
        if v != 0.0:
            c += 1
    return c


def main():
    # ---- instance ----
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read instance")
    pos = 0
    try:
        n = int(inp[pos]); B = int(inp[pos + 1]); m = int(inp[pos + 2]); pos += 3
    except Exception:
        fail("bad header")
    if not (2 <= n <= 400 and 1 <= B <= 10 ** 9 and 0 <= m <= 200000):
        fail("bad header ranges")
    try:
        diag = [int(inp[pos + i]) for i in range(n)]
        pos += n
    except Exception:
        fail("bad diagonal")
    if any(d == 0 for d in diag):
        fail("zero diagonal")
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = float(diag[i])
    try:
        for _ in range(m):
            i = int(inp[pos]) - 1
            j = int(inp[pos + 1]) - 1
            v = int(inp[pos + 2])
            pos += 3
            if not (0 <= i < j < n) or v == 0:
                fail("bad entry")
            A[i][j] = float(v)
            A[j][i] = float(v)
    except IndexError:
        fail("truncated entries")

    E0 = 0.0
    for i in range(n):
        Ai = A[i]
        for j in range(i + 1, n):
            E0 += Ai[j] * Ai[j]
    if E0 <= 0.0:
        fail("already diagonal")

    # ---- participant schedule (bounded, strict) ----
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
    if not out:
        fail("empty output")
    try:
        k = int(out[0])
    except Exception:
        fail("bad pivot count")          # also catches nan/inf tokens
    if k < 0 or k > 2000:
        fail("pivot count out of range")
    if len(out) != 1 + 2 * k:
        fail("wrong token count (got %d, need %d)" % (len(out), 1 + 2 * k))
    piv = []
    try:
        for q in range(k):
            i = int(out[1 + 2 * q]) - 1
            j = int(out[2 + 2 * q]) - 1
            if not (0 <= i < n and 0 <= j < n) or i == j:
                fail("pivot %d out of range or i==j" % (q + 1))
            piv.append((i, j))
    except ValueError:
        fail("non-integer pivot token")

    # ---- simulate under the budget ----
    spent = 0
    for (i, j) in piv:
        c = nnz(A[i]) + nnz(A[j])
        if spent + c > B:
            break
        spent += c
        rotate(A, i, j)

    E = 0.0
    for i in range(n):
        Ai = A[i]
        for j in range(i + 1, n):
            E += Ai[j] * Ai[j]

    ratio = min(1.0, 0.1 * E0 / max(E, 1e-9))
    print("spent=%d E0=%.6f E=%.6f Ratio: %.6f" % (spent, E0, E, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
