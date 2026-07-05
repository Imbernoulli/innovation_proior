# TIER: greedy
# Sylvester-Hadamard submatrix: build the smallest Sylvester Hadamard matrix of order
# 2^k >= N, take its top-left N x N block, normalise.  Exact & optimal when N is a power
# of two; a good but sub-optimal design otherwise.
import sys


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


N = int(sys.stdin.read().split()[0])
k = 0
while (1 << k) < N:
    k += 1
H = sylvester(k)
M = [[H[i][j] for j in range(N)] for i in range(N)]
normalize(M)
out = "\n".join(" ".join(str(v) for v in row) for row in M)
sys.stdout.write(out + "\n")
