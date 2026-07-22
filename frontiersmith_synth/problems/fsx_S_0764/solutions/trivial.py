# TIER: trivial
# Do nothing: request zero charge/discharge on both stores every tick. Generation
# still serves load directly whenever gen_t >= load_t (no storage is needed for
# that), but any drought is entirely unhedged. This reproduces the evaluator's own
# do-nothing baseline() exactly, so it always scores 0.1.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
z = [0.0] * T
print(json.dumps({"bc": z, "bd": z, "fc": z, "fd": z}))
