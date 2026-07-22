# TIER: trivial
# Order-up-to the flat historical mean, identically for every phase, with no trend
# and no reactive term.  This is exactly the evaluator's internal reference policy
# (the "do nothing smart" baseline), so it reproduces the anchor score (~0.1) on
# every trace and every instance.
import sys, json

inst = json.load(sys.stdin)
period = inst["period"]

policies = []
for tr in inst["traces"]:
    hist = tr["history"]
    mean = sum(hist) / float(len(hist))
    policies.append({
        "trace_id": tr["trace_id"],
        "level": [mean] * period,
        "trend": 0.0,
        "react": 0.0,
    })

print(json.dumps({"policies": policies}))
