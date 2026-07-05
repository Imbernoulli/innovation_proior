# TIER: strong
# Seeded local search over the Sylvester-submatrix seed plus random restarts:
# single-entry sign flips that increase the EXACT |det| (Bareiss), keeping the first
# row/column pinned to +1.  Deterministic (fixed seed).  Beats the greedy seed on the
# "gap" sizes where no exact Hadamard submatrix exists.
import sys, random


def bareiss_absdet(M):
    n = len(M)
    A = [row[:] for row in M]
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            piv = -1
            for r in range(k + 1, n):
                if A[r][k] != 0:
                    piv = r
                    break
            if piv < 0:
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


def normalize(M):
    n = len(M)
    for j in range(n):
        if M[0][j] == -1:
            for i in range(n):
                M[i][j] = -M[i][j]
    for i in range(1, n):
        if M[i][0] == -1:
            for j in range(n):
                M[i][j] = -M[i][j]
    return M


def sylvester(k):
    H = [[1]]
    for _ in range(k):
        m = len(H)
        N = [[0] * (2 * m) for _ in range(2 * m)]
        for i in range(m):
            for j in range(m):
                v = H[i][j]
                N[i][j] = v
                N[i][j + m] = v
                N[i + m][j] = v
                N[i + m][j + m] = -v
        H = N
    return H


def greedy_seed(N):
    k = 0
    while (1 << k) < N:
        k += 1
    H = sylvester(k)
    return normalize([[H[i][j] for j in range(N)] for i in range(N)])


N = int(sys.stdin.read().split()[0])
rng = random.Random(20240624)

best = greedy_seed(N)
bestd = bareiss_absdet(best)
budget = max(150, 9000 // max(1, N))

for restart in range(12):
    if restart == 0:
        M = [row[:] for row in greedy_seed(N)]
    else:
        # random +/-1 with first row/column pinned to +1
        M = [[1] * N]
        for _ in range(N - 1):
            M.append([1] + [rng.choice((-1, 1)) for _ in range(N - 1)])
    d = bareiss_absdet(M)
    cnt = 0
    improved = True
    while improved and cnt < budget:
        improved = False
        for i in range(1, N):
            for j in range(1, N):
                M[i][j] = -M[i][j]
                nd = bareiss_absdet(M)
                cnt += 1
                if nd > d:
                    d = nd
                    improved = True
                else:
                    M[i][j] = -M[i][j]
                if cnt >= budget:
                    break
            if cnt >= budget:
                break
    if d > bestd:
        bestd = d
        best = [row[:] for row in M]

normalize(best)
out = "\n".join(" ".join(str(v) for v in row) for row in best)
sys.stdout.write(out + "\n")
