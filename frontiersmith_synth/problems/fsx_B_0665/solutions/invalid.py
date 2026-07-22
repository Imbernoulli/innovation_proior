# TIER: invalid
# Missing required weight keys -> the evaluator's answer-schema check rejects it
# on every instance.
import sys, json

json.load(sys.stdin)
print(json.dumps({"w0": 0.0, "w1": 1.0}))
