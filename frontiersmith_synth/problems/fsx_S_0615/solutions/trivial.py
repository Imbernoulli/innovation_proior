# TIER: trivial
# Admit nothing: the gate rejects every job.  The server stays idle, no value is
# earned and no penalty is paid, so J = 0 -- exactly the evaluator's do-nothing
# reference, scoring 0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"admit": [0] * N}))
