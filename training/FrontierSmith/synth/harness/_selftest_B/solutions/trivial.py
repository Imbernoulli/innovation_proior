# TIER: trivial
import sys,json
inst=json.load(sys.stdin); n=len(inst["points"]); print(json.dumps({"tour":list(range(n))}))
