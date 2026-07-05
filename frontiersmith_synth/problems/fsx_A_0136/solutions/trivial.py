# TIER: trivial
# Reproduces the checker's internal baseline: the normalised triangular +/-1 matrix
# with |det| = 2^(N-1).  Scores exactly the calibrated floor (~0.1).
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


N = int(sys.stdin.read().split()[0])
T = [[1 if j >= i else -1 for j in range(N)] for i in range(N)]
normalize(T)
out = "\n".join(" ".join(str(v) for v in row) for row in T)
sys.stdout.write(out + "\n")
