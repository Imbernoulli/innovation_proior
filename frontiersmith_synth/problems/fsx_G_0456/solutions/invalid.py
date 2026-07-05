# TIER: invalid
"""Broken reducer: emits a projection with the WRONG row count (d+3 rows instead
of d) and a non-finite entry, so the evaluator rejects it on every atlas -> 0."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    D = int(inst["n_genes"])
    d = int(inst["d_target"])
    W = np.zeros((d + 3, D))
    W[0, 0] = float("nan")
    b = np.zeros(D)
    print(json.dumps({"W": W.tolist(), "b": b.tolist()}))


main()
