# TIER: invalid
# Deliberately malformed: wrong row length and an out-of-range / non-integer
# entry -- must be rejected by the evaluator's strict shape/range checks and
# score 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
T, N, P = inst["T"], inst["N"], inst["P"]

rows = []
for t in range(T):
    if t == 0:
        rows.append([1] * (N + 1))          # wrong length
    else:
        rows.append([P + 999] * N)          # out-of-range patient id

print(json.dumps({"assign": rows}))
