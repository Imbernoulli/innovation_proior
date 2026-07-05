# TIER: trivial
# The do-nothing component: the identity activation g(x)=x.  With g(x)=x the
# random-feature surrogate collapses to a plain linear ridge on the standardized
# telemetry, capturing only the linear part of each inverter response.  This is
# EXACTLY the evaluator's baseline, so every instance normalizes to ~0.1.
import sys, json

json.load(sys.stdin)   # read (and ignore) the public instance
print(json.dumps({"components": [{"base": "identity", "a": 1.0, "b": 0.0, "w": 1.0}]}))
