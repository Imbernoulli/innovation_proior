# TIER: greedy
"""Naive-but-valid heuristic: push the extragradient step size close to the
stability edge (0.9/L) with no momentum.  A bigger constant step converges
faster than the conservative 1/(2L) reference on well-conditioned relays, so it
beats the trivial baseline -- but it wastes the budget on rotation-dominated and
ill-conditioned classes where momentum / acceleration is what actually helps."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    L = float(inst["L"])
    eta = 0.9 / L
    print(json.dumps({"eta": eta, "alpha": 1.0, "beta": 0.0}))


main()
