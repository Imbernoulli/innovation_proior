# TIER: strong
# Design a constant "optimistic + momentum" update and tune its three scalars by a
# deterministic grid search that simulates the exact round budget:
#     alpha_k = eta,  beta_k = -opt * eta   (optimism / extrapolation),
#     gamma_k = mom                        (Polyak heavy-ball momentum).
# Optimism cancels the rotational drift of the non-symmetric coupling B and momentum
# accelerates the strongly-monotone contraction, so within K rounds this drives the
# optimality residual far below the naive baseline.  It is still only a CONSTANT
# schedule over a class that cannot reach the equilibrium in K < dim rounds, so it
# stays well short of the theoretical (per-round-tuned) optimum -> headroom remains.
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
best = (eta_ref, 0.0, 0.0)
for ei in range(1, 25):
    eta = ei * eta_ref * 0.25
    for gi in range(0, 9):
        mom = gi * 0.1
        for oi in range(0, 7):
            opt = oi * 0.15
            q = final_residual([eta] * K, [-opt * eta] * K, [mom] * K)
            if q < best_q:
                best_q = q
                best = (eta, mom, opt)

eta, mom, opt = best
print(json.dumps({"alpha": [eta] * K, "beta": [-opt * eta] * K, "gamma": [mom] * K}))
