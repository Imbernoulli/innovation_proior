# TIER: greedy
# The obvious textbook recipe: estimate the convergence rate |lambda| from the
# ratio of the last two step-norms, ASSUME the dominant mode is real, and apply
# the classic single-mode acceleration -- over-relax with omega = 1/(1-|lambda|)
# and always run Aitken's Delta^2. Great on benign real instances; but on a
# near-critical OSCILLATORY instance this over-relaxes a complex mode and
# diverges (and Aitken on an oscillation is unstable), landing far from strong.
import sys, json, math

inst = json.load(sys.stdin)
its = inst["iterates"]
lo, hi = inst["omega_min"], inst["omega_max"]
n = inst["n"]

# step vectors dx_k = x_{k+1} - x_k
dx = [[its[k + 1][i] - its[k][i] for i in range(n)] for k in range(len(its) - 1)]
norms = [math.sqrt(sum(v * v for v in d)) for d in dx]

# convergence rate from the last usable step-norm ratio
rate = 0.5
for k in range(len(norms) - 1, 0, -1):
    if norms[k - 1] > 1e-14:
        rate = norms[k] / norms[k - 1]
        break
rate = min(max(rate, 0.0), 0.999)

omega = 1.0 / (1.0 - rate)          # textbook over-relaxation for a real mode
omega = min(max(omega, lo), hi)     # clamp into the legal band

print(json.dumps({"omega": omega, "aitken_period": 3}))
