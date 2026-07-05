# TIER: strong
# Chebyshev step-size schedule: place the N step sizes at the reciprocals of the
# roots of the degree-N minimax residual polynomial on the curvature interval
# [mu, L].  This is the optimal FIXED schedule for the worst-case mode and gives
# accelerated convergence at rate ((sqrt(kappa)-1)/(sqrt(kappa)+1))^N -- many orders
# of magnitude better than any constant step.  The product over steps is
# order-independent, so we emit the roots in index order without stability tricks.
# It is near-optimal but not the per-instance optimum (it ignores the actual
# gradient weights g_i), so it stays below the unreachable ideal -> headroom.
import sys, json, math

inst = json.load(sys.stdin)
N = inst["n_steps"]
curv = inst["curv"]
L = max(curv)
mu = min(curv)

lr = []
for k in range(1, N + 1):
    root = (L + mu) / 2.0 - (L - mu) / 2.0 * math.cos((2 * k - 1) * math.pi / (2 * N))
    lr.append(1.0 / root)

print(json.dumps({"lr": lr}))
