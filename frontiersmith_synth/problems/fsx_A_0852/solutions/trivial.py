# TIER: trivial
# Ignore the probes entirely. Dump the whole budget on the first solver listed
# in the instance. The absolute simplest thing a program could do.
import sys, json

inst = json.load(sys.stdin)
heuristics = inst["heuristics"]
budget = inst["budget"]

alloc = {}
if heuristics:
    alloc[heuristics[0]["id"]] = budget

print(json.dumps({"alloc": alloc}))
