# TIER: invalid
# Always negotiate a supplier index equal to M (one past the valid range
# 0..M-1), every round. This is a malformed answer under the statement's
# contract (out-of-range supplier index) and must be rejected -> score 0 on
# every instance.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]; M = inst["M"]

actions = [{"type": "negotiate", "supplier": M, "qty": inst["Q"]} for _ in range(T)]
print(json.dumps({"actions": actions}))
