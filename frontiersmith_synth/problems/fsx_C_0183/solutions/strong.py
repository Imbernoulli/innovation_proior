# TIER: strong
# Start from the dominant-mode constant step, then coordinate-wise line search over the
# T step sizes to minimize the exactly-recomputed final gradient norm. Deterministic.
import sys, json, math
inst = json.load(sys.stdin)
A = inst["A"]; d = inst["d"]; mu = inst["mu"]; T = inst["T"]
b = inst["b"]; c = inst["c"]; z0 = inst["z0"]; n = 2 * d

def Gop(z):
    x = z[:d]; y = z[d:]
    gx = [mu * x[i] + b[i] + sum(A[i][j] * y[j] for j in range(d)) for i in range(d)]
    gy = [mu * y[i] + c[i] - sum(A[j][i] * x[j] for j in range(d)) for i in range(d)]
    return gx + gy

def obj(steps):
    z = list(z0)
    for eta in steps:
        g = Gop(z)
        z = [z[i] - eta * g[i] for i in range(n)]
        for zi in z:
            if not (abs(zi) < 1e150):
                return float("inf")
    g = Gop(z)
    return math.sqrt(sum(t * t for t in g))

# sigma_max via power iteration
def matvec(M, v): return [sum(M[i][j] * v[j] for j in range(d)) for i in range(d)]
def matTvec(M, v): return [sum(M[j][i] * v[j] for j in range(d)) for i in range(d)]
v = [1.0] * d
for _ in range(200):
    w = matTvec(A, matvec(A, v))
    nn = math.sqrt(sum(t * t for t in w))
    if nn == 0: break
    v = [t / nn for t in w]
Av = matvec(A, v)
sigma = math.sqrt(sum(t * t for t in Av))
eta0 = mu / (mu * mu + sigma * sigma)

steps = [eta0] * T
cur = obj(steps)
mults = [0.4, 0.6, 0.8, 1.25, 1.6, 2.5]
abses = [eta0 * g for g in (0.3, 0.7, 1.0, 1.5, 2.2, 3.5)]
for _ in range(4):
    for k in range(T):
        best = steps[k]; bestv = cur
        for cand in [steps[k] * f for f in mults] + abses:
            steps[k] = cand
            v2 = obj(steps)
            if v2 < bestv:
                bestv = v2; best = cand
        steps[k] = best; cur = bestv
print(json.dumps({"steps": steps}))
