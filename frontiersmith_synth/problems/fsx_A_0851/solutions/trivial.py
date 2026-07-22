# TIER: trivial
# Genuinely-online do-nothing: on round 0, commit the mean of round 0's own points
# (repeated K times) and NEVER revisit it -- every later round just echoes back
# whatever codebook it was handed (zero movement, trivially inside any budget). This
# exactly reproduces the evaluator's own F_base reference construction.
import sys, json

inst = json.load(sys.stdin)
D, K = inst["D"], inst["K"]
pts = inst["points"]
n = len(pts)
prev = inst.get("prev_codebook")

if prev is None:
    mean = [0.0] * D
    for p in pts:
        for d in range(D):
            mean[d] += p[d]
    for d in range(D):
        mean[d] /= n
    codebook = [list(mean) for _ in range(K)]
else:
    codebook = [list(row) for row in prev]

assign = [0] * n
print(json.dumps({"codebook": codebook, "assign": assign}))
