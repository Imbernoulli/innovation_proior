# TIER: invalid
# Collapses every feature to the same constant vector: all embeddings identical,
# so cosine similarity is constant and verification is at chance -> score 0.
import sys, json
inst = json.load(sys.stdin)
d = inst["d"]
W = [[0.0] * d for _ in range(d)]  # zero transform -> zero (degenerate) embeddings
print(json.dumps({"W": W}))
