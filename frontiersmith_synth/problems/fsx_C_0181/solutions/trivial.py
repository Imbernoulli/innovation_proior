# TIER: trivial
"""Baseline detector: raw Euclidean distance to the global mean.  This
reproduces the evaluator's internal baseline exactly, so every fleet normalizes
to ~0.1."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["X"], dtype=np.float64)
    mu = X.mean(axis=0)
    scores = np.sqrt(np.sum((X - mu) ** 2, axis=1))
    print(json.dumps(scores.tolist()))


main()
