# TIER: greedy
# Deterministic hill-climb: start from the triangular completion, flip free cells (row-major)
# that strictly increase |det|, for a couple of passes.
import sys

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
            aik = A[i][k]; Ai = A[i]; Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return abs(A[n - 1][n - 1])

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); K = int(next(it))
    fixed = {}
    for _ in range(K):
        r = int(next(it)); c = int(next(it)); v = int(next(it))
        fixed[(r, c)] = v
    M = [[fixed.get((i, j), 1 if i >= j else -1) for j in range(N)] for i in range(N)]
    free = [(i, j) for i in range(N) for j in range(N) if (i, j) not in fixed]
    cur = bareiss_absdet(M)
    for _p in range(2):
        improved = False
        for (i, j) in free:
            M[i][j] = -M[i][j]
            dd = bareiss_absdet(M)
            if dd > cur:
                cur = dd; improved = True
            else:
                M[i][j] = -M[i][j]
        if not improved:
            break
    sys.stdout.write("\n".join(" ".join(str(M[i][j]) for j in range(N)) for i in range(N)) + "\n")

main()
