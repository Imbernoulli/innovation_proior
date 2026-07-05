# TIER: trivial
"""Baseline imputer: per-channel MEAN imputation.  Fill each hole with the observed
mean of its column.  This reproduces the evaluator's internal baseline exactly, so
every garden instance normalizes to ~0.1."""
import sys, json
import numpy as np


def main():
    inst = json.load(sys.stdin)
    n, d = int(inst["n"]), int(inst["d"])
    Xr = inst["X"]
    miss = inst["missing"]

    # observed column sums / counts
    col_sum = np.zeros(d, dtype=np.float64)
    col_cnt = np.zeros(d, dtype=np.float64)
    for i in range(n):
        row = Xr[i]
        for j in range(d):
            v = row[j]
            if v is not None:
                col_sum[j] += float(v)
                col_cnt[j] += 1.0
    col_mean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1.0), 0.0)

    fill = [float(col_mean[j]) for (_i, j) in miss]
    print(json.dumps(fill))


main()
