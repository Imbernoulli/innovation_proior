# TIER: trivial
# Guess that the seam is a single constant term equal to the very first reading:
# f(x) ~= s_0 (exponent 0, coefficient s_0). This is exactly the evaluator's
# baseline construction. By construction it reproduces exactly the k=0 reading
# and (generically) none of the others -- fit_count = 1 -> ratio ~0.1.
import sys, json

inst = json.load(sys.stdin)
s0 = inst["s"][0]
print(json.dumps({"terms": [[0, s0]]}))
