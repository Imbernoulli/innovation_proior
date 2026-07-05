# TIER: strong
# Spectrum-aware accelerated schedule.  Read off the extreme eigenvalues (mu, L) of the
# SYMMETRIC part S=(M+M^T)/2 of the saddle operator and apply the classic optimal
# heavy-ball (Polyak) parameters for a mu-strongly / L-smoothly conditioned problem:
#     alpha = 4 / (sqrt(L)+sqrt(mu))^2 ,   beta = ((sqrt(L)-sqrt(mu))/(sqrt(L)+sqrt(mu)))^2
# then add a mild OPTIMISTIC (negative-probe) term o = alpha/2 to damp the rotation that
# the bilinear coupling A injects into the dynamics, and cap the step at 1/sigma_max(M)
# so the acceleration never destabilizes on rotation-heavy instances.  This is a fixed,
# committed schedule (constant per step) -- a genuine convergence-rate design, far better
# than plain GDA but still short of the (unreachable) degree-T polynomial optimum because
# the budget T is smaller than the dimension.
import sys, json, math
import numpy as np

inst = json.load(sys.stdin)
M = np.array(inst["M"], dtype=float)
T = inst["T"]

S = (M + M.T) / 2.0
ev = np.linalg.eigvalsh(S)
mu = max(float(ev.min()), 1e-6)
L = float(ev.max())
smax = float(np.linalg.norm(M, 2))

alpha = 4.0 / (math.sqrt(L) + math.sqrt(mu)) ** 2
beta = ((math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))) ** 2
alpha = min(alpha, 1.0 / smax)
o = alpha * 0.5

print(json.dumps({"a": [alpha] * T, "m": [beta] * T, "o": [o] * T}))
