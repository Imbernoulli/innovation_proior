# TIER: trivial
"""Reproduce the evaluator's reference method exactly: extragradient with the
constant step 1/(2L), alpha=1, beta=0.  This matches the baseline schedule, so
every instance normalises to ~0.1 (no headroom, no design work)."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    L = float(inst["L"])
    eta = 0.5 / L
    print(json.dumps({"eta": eta, "alpha": 1.0, "beta": 0.0}))


main()
