# TIER: invalid
# Always requests a machine id far out of range -> every instance is voided
# (score 0.0) on the very first job.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"assign": inst["n_machines"] + 7}))
