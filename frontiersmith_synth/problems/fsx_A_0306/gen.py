import sys, random

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

def tri(N, fixed):
    return [[fixed.get((i, j), 1 if i >= j else -1) for j in range(N)] for i in range(N)]

def main():
    t = int(sys.argv[1])
    # difficulty ladder: N = 7, 9, 11, ..., 25 (all ODD -> no Hadamard matrix exists)
    N = 5 + 2 * t
    K = round(0.12 * N * N)          # number of pre-committed (fixed) route directions
    base_seed = 90000 + t
    # resample the fixed cells until the reference (triangular) completion is NON-SINGULAR,
    # so the checker's internal baseline determinant is always positive.
    fixed = None
    for attempt in range(500):
        rng = random.Random(base_seed + attempt * 100003)
        f = {}
        while len(f) < K:
            r = rng.randrange(N)
            c = rng.randrange(N)
            if (r, c) in f:
                continue
            f[(r, c)] = rng.choice([-1, 1])
        if bareiss_absdet(tri(N, f)) > 0:
            fixed = f
            break
    if fixed is None:
        fixed = {}
    items = sorted(fixed.items())
    out = ["%d %d" % (N, len(items))]
    for (r, c), v in items:
        out.append("%d %d %d" % (r, c, v))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
