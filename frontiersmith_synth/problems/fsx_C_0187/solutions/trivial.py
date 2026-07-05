# TIER: trivial
"""Baseline clustering: a rank-quantile split of the FIRST usage feature into k
contiguous groups.  This reproduces the evaluator's internal baseline exactly, so
every resort instance normalizes to ~0.1."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    X = np.asarray(inst["X"], dtype=np.float64)
    k = int(inst["k"])
    f = X[:, 0]
    n = f.shape[0]
    order = np.argsort(f, kind="mergesort")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    lab = (rank * k) // n
    np.clip(lab, 0, k - 1, out=lab)
    print(json.dumps(lab.tolist()))


main()
