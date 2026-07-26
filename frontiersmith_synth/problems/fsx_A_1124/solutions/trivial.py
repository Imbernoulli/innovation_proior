# TIER: trivial
# Mean-field forecast desk: track ONE pooled backlog counter W, drained at rate C
# per unit time since the last update. Quote exactly the forecast (no strategic
# slack whatsoever): delay = ceil(W / C) + service. Admit iff that forecast fits
# within patience. This reproduces the evaluator's own weak reference policy, so
# it scores ~0.1 on every instance by construction.
import sys, json, math

inst = json.load(sys.stdin)
C = inst["capacity"]
jobs = inst["jobs"]

decisions = []
W = 0.0
t_ref = 0
for job in jobs:
    elapsed = job["t"] - t_ref
    W = max(0.0, W - elapsed * C)
    t_ref = job["t"]
    wait = math.ceil(W / C) if C > 0 else 0
    delay = wait + job["service"]
    if delay <= job["patience"]:
        decisions.append({"action": "quote", "delay": int(delay)})
        W += job["service"]
    else:
        decisions.append({"action": "reject"})

print(json.dumps({"decisions": decisions}))
