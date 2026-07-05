# TIER: trivial
import sys, json
inst = json.load(sys.stdin)
m = len(inst["members"])
# fattest possible pipe everywhere: safe but heavy -> normalized to ~0.1
print(json.dumps({"areas": [inst["Amax"]] * m}))
