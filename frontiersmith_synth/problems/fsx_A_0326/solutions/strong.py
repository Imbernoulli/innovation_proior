# TIER: strong
# Near-D-optimal design from Sylvester-Hadamard submatrices: build a large
# Sylvester-Hadamard matrix of order 2^k >= N and pick the N-row/N-col submatrix
# with the largest |det| over several seeded selections, then a light hill-climb
# of single-entry flips.  This is a heuristic (NOT a proven optimum): odd-order
# maximal-determinant is an open search problem, so headroom always remains.
import sys, random

def bareiss_det(M):
    n = len(M); M = [row[:] for row in M]; sign = 1; prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = -1
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    sw = i; break
            if sw == -1:
                return 0
            M[k], M[sw] = M[sw], M[k]; sign = -sign
        akk = M[k][k]
        for i in range(k + 1, n):
            aik = M[i][k]; Mi = M[i]; Mk = M[k]
            for j in range(k + 1, n):
                Mi[j] = (Mi[j] * akk - aik * Mk[j]) // prev
        prev = akk
    return sign * M[n - 1][n - 1]

def sylvester(P):
    H = [[1]]
    while len(H) < P:
        H = [row + row for row in H] + [row + [-x for x in row] for row in H]
    return H

def main():
    N = int(sys.stdin.read().split()[0])
    P = 1
    while P < N:
        P *= 2
    H = sylvester(P)

    best = None; bestd = -1
    for s in range(8):
        rng = random.Random(555 + N * 13 + s)
        rows = rng.sample(range(P), N)
        cols = rng.sample(range(P), N)
        M = [[H[i][j] for j in cols] for i in rows]
        d = abs(bareiss_det(M))
        if d > bestd:
            bestd = d; best = M
    # leading submatrix as an extra candidate
    M = [[H[i][j] for j in range(N)] for i in range(N)]
    d = abs(bareiss_det(M))
    if d > bestd:
        bestd = d; best = M

    # light deterministic hill-climb: try flipping single entries, keep improvements
    rng = random.Random(97 + N)
    cur = [row[:] for row in best]
    curd = bestd
    budget = min(120, 4 * N)
    for _ in range(budget):
        i = rng.randrange(N); j = rng.randrange(N)
        cur[i][j] = -cur[i][j]
        d = abs(bareiss_det(cur))
        if d > curd:
            curd = d
        else:
            cur[i][j] = -cur[i][j]  # revert
    if curd > bestd:
        best = cur

    out = [" ".join(str(x) for x in row) for row in best]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
