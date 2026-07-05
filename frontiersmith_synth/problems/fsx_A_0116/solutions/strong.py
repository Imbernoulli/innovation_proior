# TIER: strong
import sys, random

def bareiss_det(M):
    n = len(M)
    M = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = -1
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    sw = i
                    break
            if sw == -1:
                return 0
            M[k], M[sw] = M[sw], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]

def ascent(A, free):
    best = abs(bareiss_det(A))
    improved = True
    while improved:
        improved = False
        for (i, j) in free:
            A[i][j] = -A[i][j]
            d2 = abs(bareiss_det(A))
            if d2 > best:
                best = d2
                improved = True
            else:
                A[i][j] = -A[i][j]
    return best

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); k = int(next(it))
    forced = {}
    for _ in range(k):
        r = int(next(it)); c = int(next(it)); v = int(next(it))
        forced[(r, c)] = v
    free = [(i, j) for i in range(n) for j in range(n) if (i, j) not in forced]
    rng = random.Random(90210 + n * 131 + k * 7 + sum(v for v in forced.values()))

    def make(fn):
        A = [[fn(i, j) for j in range(n)] for i in range(n)]
        for (r, c), v in forced.items():
            A[r][c] = v
        return A

    # deterministic multi-restart coordinate ascent on |det|
    bestA = make(lambda i, j: 1 if j >= i else -1)
    bestv = ascent(bestA, free)
    for _ in range(40):
        A = make(lambda i, j: rng.choice([-1, 1]))
        v = ascent(A, free)
        if v > bestv:
            bestv = v
            bestA = A
    print("\n".join(" ".join(map(str, row)) for row in bestA))

main()
