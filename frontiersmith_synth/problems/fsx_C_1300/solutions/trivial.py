# TIER: trivial
# Constant "always level 1" for the whole future window. Ignores both the
# reported-case series and the leading indicator entirely -- a flat,
# unconditional light restriction. This reproduces the evaluator's weak
# baseline reference exactly, so it scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
FUT = inst["future_days"]

print(json.dumps({"levels": [1] * FUT}))
