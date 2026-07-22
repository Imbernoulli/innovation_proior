# TIER: trivial
# Round-robin: assign job i to machine (i mod n_machines). Ignores job type,
# weight, size, and every machine's state entirely.
import sys, json

inst = json.load(sys.stdin)
m = inst["step"] % inst["n_machines"]
print(json.dumps({"assign": m}))
