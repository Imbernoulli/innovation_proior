# TIER: trivial
"""Do nothing. Every module STAYs forever, so the final configuration is just the start
configuration -- reproduces the evaluator's do-nothing baseline exactly (ratio 0.1 on
every instance)."""
import sys, json

json.load(sys.stdin)   # public instance is unused
print(json.dumps({"table": {}, "default": "STAY"}))
