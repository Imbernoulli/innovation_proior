# TIER: trivial
# Decoupled policy: quote zero service time (S_i = 0) at every node. Every node then
# holds safety stock covering its OWN full lead time -- the naive "stock the part
# everywhere" plan. It is always feasible (tau_i = T_i >= 0, S_leaf = 0 <= s_max) and
# is exactly the evaluator's baseline, so it scores ~0.1. No pooling, no engineering.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"S": [0] * inst["n"]}))
