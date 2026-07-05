# TIER: greedy
# Tune ONLY the constant GDA step: sweep a 1-D grid of step sizes (multiples of the
# reference step) and keep whichever pure descent-ascent step drives the final
# residual lowest.  A bigger-than-conservative step converges faster than the naive
# baseline, but with no optimism/momentum the rotational coupling still limits it,
# so it beats trivial only modestly.
import sys, json, math

inst = json.load(sys.stdin)
K = inst["budget"]
M = inst["M"]
d = inst["d"]
z0 = inst["z0"]
eta_ref = inst["ref_step"]
dim = inst["dim"]


def matvec(v):
    return [sum(M[i][j] * v[j] for j in range(dim)) for i in range(dim)]


def final_residual(alpha, beta, gamma):
    z_prev = list(z0)
    z = list(z0)
    F_prev = [matvec(z0)[i] + d[i] for i in range(dim)]
    for k in range(K):
        Fz = [matvec(z)[i] + d[i] for i in range(dim)]
        znew = [z[i] - alpha[k] * Fz[i] - beta[k] * F_prev[i]
                + gamma[k] * (z[i] - z_prev[i]) for i in range(dim)]
        for val in znew:
            if val != val or abs(val) == float("inf"):
                return float("inf")
        z_prev, z, F_prev = z, znew, Fz
    Fz = [matvec(z)[i] + d[i] for i in range(dim)]
    return math.sqrt(sum(x * x for x in Fz))


best_q = float("inf")
best_eta = eta_ref
for ei in range(1, 25):
    eta = ei * eta_ref * 0.25
    q = final_residual([eta] * K, [0.0] * K, [0.0] * K)
    if q < best_q:
        best_q = q
        best_eta = eta

print(json.dumps({"alpha": [best_eta] * K, "beta": [0.0] * K, "gamma": [0.0] * K}))
