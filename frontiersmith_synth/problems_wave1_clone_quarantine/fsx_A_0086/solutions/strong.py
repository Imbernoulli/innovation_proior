# TIER: strong
# Structured optimum: Sylvester-Hadamard for power-of-two N, Paley type-I for
# N = q+1 with q prime and q = 3 (mod 4). Both attain |det| = N^(N/2) exactly
# (Hadamard bound). Candidates are verified (H H^T = N I); if none is valid the
# solver falls back to seeded multi-restart random search.
import sys, random

def bareiss_det(M):
    n = len(M)
    M = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = None
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    sw = r
                    break
            if sw is None:
                return 0
            M[k], M[sw] = M[sw], M[k]
            sign = -sign
        for r in range(k + 1, n):
            for c in range(k + 1, n):
                M[r][c] = (M[r][c] * M[k][k] - M[r][k] * M[k][c]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]

def is_pow2(n):
    return n & (n - 1) == 0

def sylvester(n):
    H = [[1]]
    while len(H) < n:
        H = [row + row for row in H] + [row + [-x for x in row] for row in H]
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
    if not (is_prime(q) and q % 4 == 3):
        return None
    QR = set((x * x) % q for x in range(1, q))
    def chi(a):
        a %= q
        if a == 0:
            return 0
        return 1 if a in QR else -1
    Q = [[chi(i - j) for j in range(q)] for i in range(q)]
    H = [[0] * n for _ in range(n)]
    H[0][0] = 1
    for j in range(1, n):
        H[0][j] = 1
    for i in range(1, n):
        H[i][0] = -1
    for i in range(1, n):
        for j in range(1, n):
            H[i][j] = Q[i - 1][j - 1] + (1 if i == j else 0)
    return H

def is_hadamard(H, n):
    if H is None or len(H) != n:
        return False
    for r in H:
        if len(r) != n or any(v not in (-1, 1) for v in r):
            return False
    for i in range(n):
        for j in range(i, n):
            s = sum(H[i][k] * H[j][k] for k in range(n))
            if (i == j and s != n) or (i != j and s != 0):
                return False
    return True

def random_best(n):
    rng = random.Random(999 + n)
    best = None
    best_d = -1
    for _ in range(200):
        M = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n)]
        d = abs(bareiss_det(M))
        if d > best_d:
            best_d = d
            best = M
    return best

def main():
    n = int(sys.stdin.read().split()[0])
    H = None
    if is_pow2(n):
        c = sylvester(n)
        if is_hadamard(c, n):
            H = c
    if H is None:
        c = paley1(n)
        if is_hadamard(c, n):
            H = c
    if H is None:
        H = random_best(n)
    out = "\n".join(" ".join(map(str, row)) for row in H)
    sys.stdout.write(out + "\n")

if __name__ == "__main__":
    main()
