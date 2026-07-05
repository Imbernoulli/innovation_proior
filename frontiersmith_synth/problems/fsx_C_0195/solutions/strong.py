# TIER: strong
"""Structure-adaptive schedule for the frozen extragradient+momentum template.

Reads the class descriptors (L, mu, type) and designs the coefficients from
convergence-rate reasoning:

  * Step size near the stability edge to spend the budget aggressively;
    extragradient look-ahead (alpha=1) keeps rotation-dominated dynamics stable
    where plain GDA diverges.
  * Momentum is only useful where the dynamics are (near-)symmetric and
    ill-conditioned.  For a strongly-monotone operator the accelerated heavy-ball
    coefficient is beta* = ((1-sqrt(q))/(1+sqrt(q)))**2 with q = mu/L, which turns
    the (1-q) rate into roughly (1-sqrt(q)) -- a large gain on ill-conditioned
    symmetric relays.  A cap keeps the simple heavy-ball form stable.
  * On rotation-dominated classes (bilinear / mixed) heavy-ball momentum
    amplifies the oscillation and diverges, so there use extragradient with a
    near-edge step and little or no momentum.

Because no single (eta, beta) wins every class, the schedule BRANCHES on the
descriptor 'type' rather than committing to one constant."""
import sys, json
import math


def main():
    inst = json.load(sys.stdin)
    L = float(inst["L"])
    mu = float(inst["mu"])
    typ = str(inst.get("type", ""))

    alpha = 1.0
    if typ == "strong":
        eta = 0.95 / L
        q = max(min(mu / L, 1.0), 0.0)
        sq = math.sqrt(q)
        beta = min(((1.0 - sq) / (1.0 + sq)) ** 2, 0.60)
    elif typ == "illcond":
        eta = 0.90 / L
        q = max(min(mu / L, 1.0), 1e-4)
        sq = math.sqrt(q)
        beta = min(((1.0 - sq) / (1.0 + sq)) ** 2, 0.55)
    elif typ == "bilinear":
        eta = 0.98 / L
        beta = 0.0
    else:  # mixed: rotation present -> extragradient, only a whisper of momentum
        eta = 0.95 / L
        beta = 0.10

    print(json.dumps({"eta": eta, "alpha": alpha, "beta": beta}))


main()
