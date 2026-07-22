# TIER: invalid
# Dump every operation id (from every machine) into machine 0's order and leave every
# other machine empty.  machine_order[0] is not a permutation of machine_ops[0] (it has
# foreign ids and duplicates relative to what belongs there) and every other machine is
# missing its own operations -- the feasibility check rejects this and the evaluator
# scores it 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
n_machines = inst["n_machines"]
all_ids = [o["id"] for o in inst["ops"]]
machine_order = [all_ids] + [[] for _ in range(n_machines - 1)]
print(json.dumps({"machine_order": machine_order}))
