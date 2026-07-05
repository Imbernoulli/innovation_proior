# TIER: trivial
import sys, json
inst = json.load(sys.stdin)
n = inst["n"]; B = inst["budget"]
# equal split of the buffer budget across all stations, ignoring demand scale
print(json.dumps({"stock": [B / n] * n}))
