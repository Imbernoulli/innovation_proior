# TIER: trivial
# Sequential block-packing: group cars by block id and fill tracks to capacity in
# order.  Whole blocks stay contiguous (little splitting) and tracks are packed to
# capacity (no overflow), but the construction is blind to the mix-affinity matrix.
# This is exactly the evaluator's reference plan, so it scores ~0.1 everywhere.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_cars"]
K = inst["n_tracks"]
C = inst["cap"]
block = inst["block"]

order = sorted(range(N), key=lambda i: block[i])
assign = [0] * N
t = 0
used = 0
for i in order:
    if used >= C and t < K - 1:
        t += 1
        used = 0
    assign[i] = t
    used += 1

print(json.dumps({"assign": assign}))
