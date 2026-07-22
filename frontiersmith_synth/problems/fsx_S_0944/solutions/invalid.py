# TIER: invalid
# Load every parcel onto truck 0. Total parcel weight always exceeds a single
# truck's capacity in this family, so truck 0 overflows -> the timeline is
# infeasible at the very first overflowing parcel -> the evaluator scores this
# instance 0.0, for every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]

print(json.dumps({"placements": [0] * N, "moves": []}))
