# TIER: strong
# Legendre baseline + full single-flip hill-climbing to a local optimum, wrapped in seeded
# random-restart perturbations. Each restart perturbs the best-so-far by a few random flips and
# re-climbs; the best EXACT determinant (Bareiss) is kept. Strictly dominates the greedy climb.
import sys
import random


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def baseline(p):
    M = [[1 if i == j else (legendre((j - i) % p, p) or 1) for j in range(p)] for i in range(p)]
    for t in range(2):
        i = (2 * t + 1) % p
        j = (5 * t + 3) % p
        M[i][j] = -M[i][j]
    return M


def bareiss_det(M):
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = None
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    sw = i
                    break
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        akk = A[k][k]
        for i in range(k + 1, n):
            Aik = A[i][k]
            Ai = A[i]
            Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - Aik * Ak[j]) // prev
        prev = akk
    return sign * A[n - 1][n - 1]


def inverse(M):
    n = len(M)
    A = [[float(M[i][j]) for j in range(n)] + [1.0 if i == c else 0.0 for c in range(n)] for i in range(n)]
    for c in range(n):
        pr = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[pr][c]) < 1e-9:
            return None
        A[c], A[pr] = A[pr], A[c]
        iv = 1.0 / A[c][c]
        for j in range(2 * n):
            A[c][j] *= iv
        for r in range(n):
            if r != c and A[r][c] != 0.0:
                f = A[r][c]
                Ar = A[r]
                Ac = A[c]
                for j in range(2 * n):
                    Ar[j] -= f * Ac[j]
    return [row[n:] for row in A]


def climb_from(cur, maxflips):
    n = len(cur)
    inv = inverse(cur)
    if inv is None:
        return
    for _ in range(maxflips):
        bg = 1.0 + 1e-9
        bi = bj = -1
        for i in range(n):
            ci = cur[i]
            for j in range(n):
                fac = 1.0 + (-2.0 * ci[j]) * inv[j][i]
                if abs(fac) > bg:
                    bg = abs(fac)
                    bi = i
                    bj = j
        if bi < 0:
            break
        delta = -2.0 * cur[bi][bj]
        den = 1.0 + delta * inv[bj][bi]
        if abs(den) < 1e-9:
            break
        cur[bi][bj] = -cur[bi][bj]
        col = [inv[r][bi] for r in range(n)]
        rw = inv[bj][:]
        for r in range(n):
            cr = delta * col[r] / den
            ir = inv[r]
            for c in range(n):
                ir[c] -= cr * rw[c]


def main():
    N = int(sys.stdin.read().split()[0])
    rng = random.Random(7)
    best = baseline(N)
    climb_from(best, 8 * N)
    bs = abs(bareiss_det(best))
    perturb = max(2, N // 6)
    for t in range(16):
        cur = [row[:] for row in best]
        for _ in range(perturb):
            i = rng.randrange(N)
            j = rng.randrange(N)
            cur[i][j] = -cur[i][j]
        climb_from(cur, 8 * N)
        s = abs(bareiss_det(cur))
        if s > bs:
            bs = s
            best = cur
    out = "\n".join(" ".join(str(x) for x in row) for row in best)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
