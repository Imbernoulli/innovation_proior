# TIER: invalid
# On every epoch call, emits a shipment entry with t == epoch_end (one
# step past the end of the range this epoch call is allowed to use) and an
# out-of-range depot index. The evaluator requires every shipment entry's
# t to fall within [epoch_start, epoch_end) for the epoch that emitted it,
# so this is rejected as infeasible -> scores 0.0 everywhere.
import sys, json

inst = json.load(sys.stdin)
t_bad = inst["epoch_end"]
print(json.dumps({"shipments": [[t_bad, 0, 99, 10.0]]}))
