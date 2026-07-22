# TIER: trivial
# Fresh-slab / bump allocator: give every block its own private byte range,
# stacked in input order, never reusing a single freed byte.  Always valid
# (no two ranges ever overlap), high-water mark = sum of all sizes = the
# evaluator's fresh-slab baseline -> scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
blocks = inst["blocks"]

off = []
cur = 0
for bl in blocks:
    off.append(cur)
    cur += bl["size"]

print(json.dumps({"offset": off}))
