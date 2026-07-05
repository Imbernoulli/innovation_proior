# TIER: invalid
# Dump every request into block 0. Since the queues have total duration far above one
# block's capacity, block 0 overflows and the evaluator rejects the packing -> 0.0.
import sys, json

inst = json.load(sys.stdin)
items = inst["items"]
print(json.dumps({"assign": [0] * len(items)}))
