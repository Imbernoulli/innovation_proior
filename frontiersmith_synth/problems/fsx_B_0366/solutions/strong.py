# TIER: strong
# Two-variable power law: log P ~ log A + p*log k + r*log V.
# Least-squares fit (3x3 normal equations) recovers the joint (k, V) scaling and
# extrapolates well into the large-base regime. Residual = telemetry noise plus a
# small ignored baseline-overhead bias, so it beats greedy substantially but stays
# below the noise ceiling.
import sys, math

d = sys.stdin.read().split()
M = int(d[0])
rows = []
for i in range(M):
    k = float(d[1 + 3 * i]); V = float(d[2 + 3 * i]); p = float(d[3 + 3 * i])
    rows.append((math.log(k), math.log(V), math.log(p)))

# design vectors phi = [1, ln k, ln V]; solve (Phi^T Phi) w = Phi^T y
ATA = [[0.0] * 3 for _ in range(3)]
ATy = [0.0] * 3
for (ln_k, ln_V, ln_p) in rows:
    phi = [1.0, ln_k, ln_V]
    for a in range(3):
        ATy[a] += phi[a] * ln_p
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
print("%.6f * k ** %.6f * V ** %.6f" % (A, p, r))
