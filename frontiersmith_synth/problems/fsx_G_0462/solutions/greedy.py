# TIER: greedy
# The classic optimal CONSTANT step size for gradient descent on a strongly convex
# quadratic: eta = 2/(L+mu), balancing the contraction of the fastest and slowest
# modes.  It converges at rate ((kappa-1)/(kappa+1))^N -- strictly better than the
# 1/L baseline, but far short of an accelerated (varying) schedule, and on
# edge-heavy spectra the fixed step can even oscillate.  A solid but unremarkable
# middle of the ladder.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_steps"]
curv = inst["curv"]
L = max(curv)
mu = min(curv)
eta = 2.0 / (L + mu)
print(json.dumps({"lr": [eta] * N}))
