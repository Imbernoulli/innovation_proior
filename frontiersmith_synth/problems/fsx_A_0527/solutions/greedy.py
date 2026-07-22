# TIER: greedy
# The obvious move: fit a per-class acceptance bar to the PUBLISHED early-regime
# means and apply it flat -- ignore how full the berth is (rem_bucket) and ignore
# the drift probe (sig_bucket).  "Admit any vessel whose fee beats its class's
# usual fee."  This spends credits on the cheap early water and, on a drifting
# tide, has no berth left when the scarce deep-sea class finally shows up.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]; R = inst["R_buckets"]; S = inst["S_buckets"]
prior = inst["prior_mu"]

bars = [[[float(prior[c]) for _ in range(S)] for _ in range(R)] for c in range(K)]
print(json.dumps({"bars": bars}))
