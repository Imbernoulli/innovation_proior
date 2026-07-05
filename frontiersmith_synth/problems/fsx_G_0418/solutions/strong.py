# TIER: strong
# Full log-linear Arrhenius fit. Linearise k = A * exp(-E/T) * C**n as
#   ln k = ln A - E*(1/T) + n*ln C,
# which is linear in features [1, 1/T, ln C]. Solve the 3x3 normal equations by
# Gaussian elimination to recover (A, E, n) jointly, then emit the closed form.
# Captures BOTH the temperature exponential and the concentration order, so it
# extrapolates far better than the temperature-only greedy fit. Residual = bench
# noise + amplified extrapolation error, so it stays below the noise ceiling.
import sys, math

d = sys.stdin.read().split()
M = int(d[0])
rows = []
for i in range(M):
    T = float(d[1 + 3 * i]); C = float(d[2 + 3 * i]); k = float(d[3 + 3 * i])
    rows.append((1.0 / T, math.log(C), math.log(k)))

# design phi = [1, 1/T, ln C]; solve (Phi^T Phi) w = Phi^T y
ATA = [[0.0] * 3 for _ in range(3)]
ATy = [0.0] * 3
for (invT, lnC, lnk) in rows:
    phi = [1.0, invT, lnC]
    for a in range(3):
        ATy[a] += phi[a] * lnk
        for b in range(3):
            ATA[a][b] += phi[a] * phi[b]

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
A = math.exp(w[0]); negE = w[1]; n = w[2]     # w[1] ~ -E
print("%.10e * exp(%.8f / T) * C ** %.8f" % (A, negE, n))
