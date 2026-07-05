# TIER: strong
# Structured optimum: build an exact Hadamard matrix, attaining |det| = N^(N/2),
# the theoretical maximum. Two constructions cover every test size:
#   * Sylvester-Hadamard for N a power of two (32, 64);
#   * Paley type I (Legendre-symbol Jacobsthal matrix) for N = q + 1 with q an
#     odd prime == 3 mod 4 (24, 44, 48, 60).
# If neither applies, fall back to float-guided hill climbing from a random start.
import sys
import random
from math import log

def is_pow2(n):
    return n & (n - 1) == 0

def sylvester(n):
    H = [[1]]
    while len(H) < n:
        m = len(H)
        H = [[H[i][j] for j in range(m)] + [H[i][j] for j in range(m)] for i in range(m)] + \
            [[H[i][j] for j in range(m)] + [-H[i][j] for j in range(m)] for i in range(m)]
    return H

def is_prime(q):
    if q < 2:
        return False
    i = 2
    while i * i <= q:
        if q % i == 0:
            return False
        i += 1
    return True

def paley1(n):
    q = n - 1
    # Legendre symbol chi(a) over GF(q)
    def chi(a):
        a %= q
        if a == 0:
            return 0
        return 1 if pow(a, (q - 1) // 2, q) == 1 else -1
    H = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == 0:
                H[i][j] = 1
            elif j == 0:
                H[i][j] = -1
            elif i == j:
                H[i][j] = 1          # identity term on the interior diagonal
            else:
                H[i][j] = chi((i - 1) - (j - 1))
    return H

def slogdet(A):
    n = len(A)
    A = [row[:] for row in A]
    ld = 0.0
    for k in range(n):
        p = k
        best = abs(A[k][k])
        for i in range(k + 1, n):
            v = abs(A[i][k])
            if v > best:
                best = v
                p = i
        if best < 1e-9:
            return float("-inf")
        if p != k:
            A[k], A[p] = A[p], A[k]
        piv = A[k][k]
        ld += log(abs(piv))
        Ak = A[k]
        for i in range(k + 1, n):
            Ai = A[i]
            f = Ai[k] / piv
            if f != 0.0:
                for j in range(k + 1, n):
                    Ai[j] -= f * Ak[j]
    return ld

def hill_climb(n):
    rng = random.Random(7 + n)
    best = None
    best_ld = float("-inf")
    for _ in range(8):
        M = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
        ld = slogdet([[float(x) for x in r] for r in M])
        if ld > best_ld:
            best_ld, best = ld, M
    M = best
    cur = best_ld
    for _ in range(6):
        improved = False
        bi = bj = -1
        bld = cur
        for i in range(n):
            for j in range(n):
                M[i][j] = -M[i][j]
                ld = slogdet([[float(x) for x in r] for r in M])
                if ld > bld + 1e-9:
                    bld, bi, bj = ld, i, j
                M[i][j] = -M[i][j]
        if bi >= 0:
            M[bi][bj] = -M[bi][bj]
            cur = bld
            improved = True
        if not improved:
            break
    return M

def build(n):
    if is_pow2(n):
        return sylvester(n)
    if is_prime(n - 1) and (n - 1) % 4 == 3:
        return paley1(n)
    return hill_climb(n)

def main():
    n = int(sys.stdin.read().split()[0])
    H = build(n)
    out = "\n".join(" ".join(map(str, row)) for row in H)
    sys.stdout.write(out + "\n")

if __name__ == "__main__":
    main()
