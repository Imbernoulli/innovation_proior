# TIER: greedy
# The obvious allocator: process blocks in ARRIVAL (birth) order and drop each
# at the LOWEST address that does not collide with any currently-alive block --
# classic online first-fit.  It reuses freed space, so it beats the fresh slab,
# but it is completely lifetime-blind: it never looks at death times, so a
# just-arrived long-lived block grabs whatever low hole a dying cohort opened and
# then permanently splits the address space.  Fragments exactly where the planted
# cohort/spine structure hurts most.
import sys, json

inst = json.load(sys.stdin)
blocks = inst["blocks"]
M = len(blocks)

order = sorted(range(M), key=lambda i: (blocks[i]["birth"], i))

placed = []          # (birth, death, off, size)
off = [0] * M
for i in order:
    s = blocks[i]["size"]
    b = blocks[i]["birth"]
    d = blocks[i]["death"]
    occ = []
    for (pb, pd, po, ps) in placed:
        if pb < d and b < pd:            # time overlap
            occ.append((po, po + ps))
    occ.sort()
    o = 0
    for (lo, hi) in occ:
        if o + s <= lo:
            break
        if o < hi:
            o = hi
    off[i] = o
    placed.append((b, d, o, s))

print(json.dumps({"offset": off}))
