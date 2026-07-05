# TIER: greedy
# Start from the Legendre baseline, then apply a LIMITED number of single-entry sign flips,
# each chosen to maximize the exact multilinear rescaling of the determinant (tracked via a
# Sherman-Morrison inverse update). No restarts -> a modest, bounded improvement over baseline.
import sys


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


def climb(M, maxflips):
    n = len(M)
    cur = [row[:] for row in M]
    inv = inverse(cur)
    if inv is None:
        return cur
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
    return cur


def main():
    N = int(sys.stdin.read().split()[0])
    M = climb(baseline(N), max(2, N // 4))
    out = "\n".join(" ".join(str(x) for x in row) for row in M)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
