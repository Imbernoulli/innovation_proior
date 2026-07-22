# TIER: trivial
# Do nothing clever: order every machine's operations by ascending global id, which is
# exactly (job index, then position within the job).  This is the evaluator's own
# reference construction, so it always scores exactly 0.1.
import sys, json

inst = json.load(sys.stdin)
machine_order = [sorted(ids) for ids in inst["machine_ops"]]
print(json.dumps({"machine_order": machine_order}))
