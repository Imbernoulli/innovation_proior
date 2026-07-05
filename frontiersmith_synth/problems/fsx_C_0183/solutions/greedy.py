# TIER: greedy
# Constant step tuned to the dominant mode: eta = mu / (mu^2 + sigma_max^2),
# which minimizes the per-step contraction |1 - eta*(mu +/- i*sigma_max)|.
import sys, json, math
inst = json.load(sys.stdin)
A = inst["A"]; d = inst["d"]; mu = inst["mu"]; T = inst["T"]

# power iteration on A^T A to get sigma_max (no numpy dependency needed).
def matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]
def matTvec(M, v):
    return [sum(M[j][i] * v[j] for j in range(len(M))) for i in range(len(M[0]))]
v = [1.0] * d
for _ in range(200):
    w = matTvec(A, matvec(A, v))
    n = math.sqrt(sum(t * t for t in w))
    if n == 0:
        break
    v = [t / n for t in w]
Av = matvec(A, v)
sigma = math.sqrt(sum(t * t for t in Av))
eta = mu / (mu * mu + sigma * sigma)
print(json.dumps({"steps": [eta] * T}))
