# TIER: greedy
"""Per-feature standardized distance to the mean (z-score anomaly).  Fixes the
wildly-scaled fleets and stays fine on a single global cloud, but -- being a
global centroid method -- it flags whole outer operating regimes on multi-modal
fleets and is blind to correlation-breaking faults, so it collapses there."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["X"], dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    scores = np.sqrt(np.sum(Z ** 2, axis=1))
    print(json.dumps(scores.tolist()))


main()
