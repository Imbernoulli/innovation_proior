# TIER: invalid
import sys, json
inst = json.load(sys.stdin)
n = inst["n"]; B = inst["budget"]
# grossly exceeds the buffer budget (sum = n * B > B) -> infeasible -> scored 0
print(json.dumps({"stock": [B] * n}))
