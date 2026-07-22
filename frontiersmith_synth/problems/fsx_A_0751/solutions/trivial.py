# TIER: trivial
# Route every parcel by its raw tracking code mod B -- no hash parameters at
# all.  Expressed in the (a, c, M) contract as a=1, c=0, M=B, which collapses
# to bucket(x) = x mod B, exactly reproducing the evaluator's own weak
# baseline.  Scores ~0.1 on every instance by construction.
import sys, json

inst = json.load(sys.stdin)
B = inst["B"]
print(json.dumps({"a": 1, "c": 0, "M": B}))
