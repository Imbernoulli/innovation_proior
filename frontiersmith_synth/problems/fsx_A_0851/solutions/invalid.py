# TIER: invalid
# Deliberately out-of-bound codewords (|value| > BOUND) -- must be rejected -> 0.0.
import sys, json

inst = json.load(sys.stdin)
D, K = inst["D"], inst["K"]
n = len(inst["points"])

codebook = [[999999.0] * D for _ in range(K)]
assign = [0] * n

print(json.dumps({"codebook": codebook, "assign": assign}))
