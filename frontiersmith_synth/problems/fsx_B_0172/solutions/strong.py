# TIER: strong
# Two-variable power law: log T ~ log A + p*log n + r*log c  (r ~ -q).
# Least-squares fit (3x3 normal equations) recovers the joint (n, c) scaling and
# extrapolates well. Residual = observation noise + a small ignored-offset bias,
# so it beats greedy substantially but stays below the noise ceiling.
import sys, math

d = sys.stdin.read().split()
M = int(d[0])
rows = []
for i in range(M):
    n = float(d[1 + 3 * i]); c = float(d[2 + 3 * i]); t = float(d[3 + 3 * i])
    rows.append((math.log(n), math.log(c), math.log(t)))

# design vectors phi = [1, ln n, ln c]; solve (Phi^T Phi) w = Phi^T y
ATA = [[0.0] * 3 for _ in range(3)]
ATy = [0.0] * 3
for (ln_n, ln_c, ln_t) in rows:
    phi = [1.0, ln_n, ln_c]
    for a in range(3):
        ATy[a] += phi[a] * ln_t
        for b in range(3):
            ATA[a][b] += phi[a] * phi[b]

# Gaussian elimination on the 3x3 system
Aug = [ATA[i][:] + [ATy[i]] for i in range(3)]
for col in range(3):
    piv = max(range(col, 3), key=lambda r: abs(Aug[r][col]))
    Aug[col], Aug[piv] = Aug[piv], Aug[col]
    pv = Aug[col][col]
    for j in range(col, 4):
        Aug[col][j] /= pv
    for r in range(3):
        if r != col and Aug[r][col] != 0.0:
            f = Aug[r][col]
            for j in range(col, 4):
                Aug[r][j] -= f * Aug[col][j]
w = [Aug[i][3] for i in range(3)]
A = math.exp(w[0]); p = w[1]; r = w[2]
print("%.6f * n ** %.6f * c ** %.6f" % (A, p, r))
