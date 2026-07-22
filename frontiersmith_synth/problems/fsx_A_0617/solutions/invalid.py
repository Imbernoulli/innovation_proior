# TIER: invalid
# Place every block at offset 0.  In every instance many blocks are alive at the
# same time (overlapping lifetimes), so their byte ranges [0, size) collide ->
# the layout is infeasible -> the evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
M = inst["n"]

print(json.dumps({"offset": [0] * M}))
