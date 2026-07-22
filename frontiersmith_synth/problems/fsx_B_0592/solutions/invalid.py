# TIER: invalid
# Violates the budget: keeps every distinct token (|keep| > K) -> rejected -> 0.
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({"keep": sorted(set(inst["stream"]))}))
