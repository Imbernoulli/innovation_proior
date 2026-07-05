# TIER: invalid
import sys,json
inst=json.load(sys.stdin); n=len(inst["points"]); print(json.dumps({"tour":[0]*n}))
