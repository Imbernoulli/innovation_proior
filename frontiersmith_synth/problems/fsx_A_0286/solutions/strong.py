# TIER: strong
# Sherman-Morrison rank-1 incremental hill climbing on |det| over the FREE (non-wired)
# cells, with many random restarts. For a single-entry sign flip A -> A + delta*e_i e_j^T
# the determinant scales by (1 + delta * Ainv[j][i]); we repeatedly apply the free-cell
# flip that maximizes |scale| > 1 and update the inverse in O(N^2), until a local optimum.
# Float arithmetic is used ONLY to guide the search; the checker rescoring is exact (Bareiss).
import sys, random, math

def bareiss_det(M):
    n = len(M); M = [r[:] for r in M]; sign = 1; prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = None
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    sw = r; break
            if sw is None:
                return 0
            M[k], M[sw] = M[sw], M[k]; sign = -sign
        for r in range(k + 1, n):
            for c in range(k + 1, n):
                M[r][c] = (M[r][c] * M[k][k] - M[r][k] * M[k][c]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]

def parse():
    tok = sys.stdin.read().split()
    n = int(tok[0]); nf = int(tok[1]); fixed = {}
    idx = 2
    for _ in range(nf):
        r = int(tok[idx]); c = int(tok[idx + 1]); v = int(tok[idx + 2]); idx += 3
        fixed[(r, c)] = v
    return n, fixed

def completion(n, fixed, rng):
    M = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n)]
    for (i, j), v in fixed.items():
        M[i][j] = v
    return M

def inverse(M):
    n = len(M)
    A = [[float(M[i][j]) for j in range(n)] + [1.0 if k == i else 0.0 for k in range(n)]
         for i in range(n)]
    for k in range(n):
        p = k
        for r in range(k + 1, n):
            if abs(A[r][k]) > abs(A[p][k]):
                p = r
        if abs(A[p][k]) < 1e-9:
            return None
        if p != k:
            A[k], A[p] = A[p], A[k]
        inv = 1.0 / A[k][k]
        for c in range(2 * n):
            A[k][c] *= inv
        for r in range(n):
            if r == k:
                continue
            f = A[r][k]
            if f != 0.0:
                Ar = A[r]; Ak = A[k]
                for c in range(k, 2 * n):
                    Ar[c] -= f * Ak[c]
    return [row[n:] for row in A]

def logabsdet(M):
    n = len(M); A = [[float(x) for x in r] for r in M]; s = 0.0
    for k in range(n):
        p = k
        for r in range(k + 1, n):
            if abs(A[r][k]) > abs(A[p][k]):
                p = r
        if abs(A[p][k]) < 1e-12:
            return -1e18
        if p != k:
            A[k], A[p] = A[p], A[k]
        s += math.log(abs(A[k][k]))
        piv = A[k][k]
        for r in range(k + 1, n):
            f = A[r][k] / piv
            Ar = A[r]; Ak = A[k]
            for c in range(k, n):
                Ar[c] -= f * Ak[c]
    return s

def hillclimb(n, fixed, rng, restarts, max_flips):
    free = [(i, j) for i in range(n) for j in range(n) if (i, j) not in fixed]
    best = None; best_ld = -1e18
    for _ in range(restarts):
        M = completion(n, fixed, rng)
        Ainv = inverse(M)
        if Ainv is None:
            continue
        flips = 0
        while flips < max_flips:
            bestm = 1.0 + 1e-9; bi = bj = -1; bdelta = 0.0
            for (i, j) in free:
                delta = -2.0 * M[i][j]
                m = 1.0 + delta * Ainv[j][i]
                am = m if m >= 0 else -m
                if am > bestm:
                    bestm = am; bi, bj = i, j; bdelta = delta
            if bi < 0:
                break
            i, j = bi, bj; delta = bdelta
            denom = 1.0 + delta * Ainv[j][i]
            coef = delta / denom
            col = [Ainv[a][i] for a in range(n)]
            rowv = [Ainv[j][b] for b in range(n)]
            for a in range(n):
                ca = coef * col[a]
                Ara = Ainv[a]
                for b in range(n):
                    Ara[b] -= ca * rowv[b]
            M[i][j] = -M[i][j]
            flips += 1
            if flips % 15 == 0:            # periodic exact refresh to fight float drift
                ni = inverse(M)
                if ni is None:
                    break
                Ainv = ni
        ld = logabsdet(M)
        if ld > best_ld:
            best_ld = ld; best = [r[:] for r in M]
    return best

def main():
    n, fixed = parse()
    rng = random.Random(80081 + n)
    M = hillclimb(n, fixed, rng, restarts=10, max_flips=250)
    if M is None:                          # extreme fallback: any non-singular completion
        for _ in range(500):
            M = completion(n, fixed, rng)
            if bareiss_det(M) != 0:
                break
    sys.stdout.write("\n".join(" ".join(map(str, row)) for row in M) + "\n")

if __name__ == "__main__":
    main()
