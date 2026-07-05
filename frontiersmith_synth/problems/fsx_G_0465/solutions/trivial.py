# TIER: trivial
# Column-mean imputation: predict each masked cell as the mean of the observed
# entries in its column. This is exactly the evaluator's baseline construction, so
# it maps to ~0.1. It ignores all between-variable correlation.
import sys, json

inst = json.load(sys.stdin)
N, D = inst["N"], inst["D"]
M = inst["matrix"]
mask = inst["masked"]

colmean = [0.0] * D
for j in range(D):
    vals = [M[i][j] for i in range(N) if M[i][j] is not None]
    colmean[j] = sum(vals) / len(vals) if vals else 0.0

preds = [colmean[j] for (i, j) in mask]
print(json.dumps({"preds": preds}))
