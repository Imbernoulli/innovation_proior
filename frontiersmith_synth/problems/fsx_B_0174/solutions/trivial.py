# TIER: trivial
# Naive reference: assemble modules in natural module-id order (0,1,...,K-1).
# This is exactly the baseline reference used for normalisation -> maps to 0.1.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
print(json.dumps({"order": list(range(K))}))
